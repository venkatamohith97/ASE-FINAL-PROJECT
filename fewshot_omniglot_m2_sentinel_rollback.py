"""
Milestone-2 (Fixed): Omniglot Few-Shot with Sentinel + Rollback

Fixes vs old M2:
1) latency_total_ms includes embedding extraction + method + sentinel overhead (comparable to M1)
2) Sentinel uses RELATIVE checks (ridge vs prototype) so it won't rollback 100% randomly

Rollback logic (correct for episodic few-shot):
- If primary method is unsafe, fallback to prototype FOR THE SAME EPISODE.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Backbone (Conv4)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv4Backbone(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),   # 84 -> 42
            ConvBlock(64, 64),  # 42 -> 21
            ConvBlock(64, 64),  # 21 -> 10
            ConvBlock(64, 64),  # 10 -> 5
        )
        self.fc = nn.Linear(64 * 5 * 5, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# -----------------------------
# Episode sampling
# -----------------------------
def build_class_to_indices(dataset) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        y = int(y)
        class_to_indices.setdefault(y, []).append(idx)
    return class_to_indices


@dataclass
class Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def sample_episode(dataset, class_to_indices, ways: int, shots: int, queries: int) -> Episode:
    eligible = [c for c, idxs in class_to_indices.items() if len(idxs) >= (shots + queries)]
    if len(eligible) < ways:
        raise ValueError(f"Not enough eligible classes. Need {ways}, got {len(eligible)}.")

    chosen = random.sample(eligible, ways)
    label_map = {cls: i for i, cls in enumerate(chosen)}

    sx, sy, qx, qy = [], [], [], []
    for cls in chosen:
        idxs = random.sample(class_to_indices[cls], shots + queries)
        s_idxs = idxs[:shots]
        q_idxs = idxs[shots:]

        for si in s_idxs:
            x, _ = dataset[si]
            sx.append(x)
            sy.append(label_map[cls])

        for qi in q_idxs:
            x, _ = dataset[qi]
            qx.append(x)
            qy.append(label_map[cls])

    return Episode(
        support_x=torch.stack(sx, 0),
        support_y=torch.tensor(sy, dtype=torch.long),
        query_x=torch.stack(qx, 0),
        query_y=torch.tensor(qy, dtype=torch.long),
    )


# -----------------------------
# Methods: Prototype + Ridge
# -----------------------------
def prototype_logits(support_emb, support_y, query_emb, ways: int) -> torch.Tensor:
    protos = []
    for c in range(ways):
        protos.append(support_emb[support_y == c].mean(dim=0))
    protos = torch.stack(protos, dim=0)

    x2 = (query_emb ** 2).sum(dim=1, keepdim=True)
    p2 = (protos ** 2).sum(dim=1).unsqueeze(0)
    xp = query_emb @ protos.t()
    d2 = x2 + p2 - 2 * xp
    return -d2


def ridge_logits(support_emb, support_y, query_emb, ways: int, ridge_lambda: float) -> torch.Tensor:
    X = support_emb  # [S, D]
    S, D = X.shape
    Y = F.one_hot(support_y, num_classes=ways).float()  # [S, ways]

    XtX = X.t() @ X
    reg = ridge_lambda * torch.eye(D, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, X.t() @ Y)  # [D, ways]
    return query_emb @ W


def accuracy_from_logits(logits, y_true) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y_true).float().mean().item())


# -----------------------------
# Sentinel helpers
# -----------------------------
def mean_entropy(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1).clamp(min=1e-12)
    ent = -(p * p.log()).sum(dim=1).mean()
    return float(ent.item())


def mean_confidence(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1)
    conf = p.max(dim=1).values.mean()
    return float(conf.item())


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Mean JS divergence between probability distributions of two logits sets."""
    pa = F.softmax(logits_a, dim=1).clamp(min=1e-12)
    pb = F.softmax(logits_b, dim=1).clamp(min=1e-12)
    m = 0.5 * (pa + pb)
    kl_a = (pa * (pa.log() - m.log())).sum(dim=1)
    kl_b = (pb * (pb.log() - m.log())).sum(dim=1)
    js = 0.5 * (kl_a + kl_b)
    return float(js.mean().item())


class DriftMonitor:
    """Simple feature drift monitor using first episode mean/std as reference."""
    def __init__(self):
        self.ref_mean = None
        self.ref_std = None
        self.eps = 1e-6

    def update_reference(self, support_emb: torch.Tensor) -> None:
        if self.ref_mean is None:
            self.ref_mean = support_emb.mean(dim=0).detach()
            self.ref_std = support_emb.std(dim=0).detach().clamp(min=self.eps)

    def drift_score(self, support_emb: torch.Tensor) -> float:
        if self.ref_mean is None or self.ref_std is None:
            return 0.0
        cur_mean = support_emb.mean(dim=0)
        z = (cur_mean - self.ref_mean) / self.ref_std
        return float(z.abs().mean().item())


# -----------------------------
# Optional pretraining
# -----------------------------
def load_backbone_checkpoint(backbone: nn.Module, ckpt_path: Path, device: torch.device) -> bool:
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "backbone_state" not in ckpt:
        return False
    backbone.load_state_dict(ckpt["backbone_state"])
    return True


def train_backbone_on_background(device, backbone, train_dataset, epochs, batch_size, lr, out_path: Path) -> None:
    if epochs <= 0:
        print("Skipping pretraining (train_epochs <= 0). Using random backbone.")
        return

    class_ids = sorted({int(train_dataset[i][1]) for i in range(len(train_dataset))})
    num_classes = len(class_ids)
    label_map = {c: i for i, c in enumerate(class_ids)}

    def mapped_collate(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(x)
            ys.append(label_map[int(y)])
        return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=mapped_collate)

    backbone.train()
    clf = LinearClassifier(in_dim=backbone.fc.out_features, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(clf.parameters()), lr=lr)

    print(f"Pretraining on Omniglot background: classes={num_classes}, epochs={epochs}")
    for ep in range(1, epochs + 1):
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            emb = backbone(xb)
            logits = clf(emb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            ep_acc += accuracy_from_logits(logits.detach(), yb.detach())
            n += 1
        print(f"  epoch {ep}/{epochs} | loss={ep_loss/max(n,1):.4f} | acc={ep_acc/max(n,1):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_state": backbone.state_dict()}, str(out_path))
    print(f"Saved backbone checkpoint: {out_path}")


# -----------------------------
# Run
# -----------------------------
def run_m2(
    device,
    backbone,
    eval_dataset,
    ways,
    shots,
    queries,
    episodes,
    primary,
    ridge_lambda,
    max_latency_ms,
    max_drift,
    max_entropy_gap,
    min_conf_primary,
    max_js,
) -> pd.DataFrame:
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    class_to_indices = build_class_to_indices(eval_dataset)
    drift_mon = DriftMonitor()

    rows = []

    for ep in range(1, episodes + 1):
        episode = sample_episode(eval_dataset, class_to_indices, ways, shots, queries)

        t0 = time.perf_counter()
        with torch.no_grad():
            sx = episode.support_x.to(device)
            qx = episode.query_x.to(device)
            sy = episode.support_y.to(device)
            qy = episode.query_y.to(device)

            # embeddings (this is the expensive part)
            s_emb = backbone(sx)
            q_emb = backbone(qx)

            drift_mon.update_reference(s_emb)
            drift = drift_mon.drift_score(s_emb)

            # Always compute prototype baseline (cheap) for sentinel comparison
            logits_proto = prototype_logits(s_emb, sy, q_emb, ways)
            ent_proto = mean_entropy(logits_proto)
            conf_proto = mean_confidence(logits_proto)

            # Primary method
            if primary == "ridge":
                logits_primary = ridge_logits(s_emb, sy, q_emb, ways, ridge_lambda=ridge_lambda)
            else:
                logits_primary = logits_proto

            ent_primary = mean_entropy(logits_primary)
            conf_primary = mean_confidence(logits_primary)
            js = js_divergence(logits_primary, logits_proto)
            ent_gap = ent_primary - ent_proto

        latency_total_ms = (time.perf_counter() - t0) * 1000.0

        # Sentinel rules (meaningful)
        alerts = []
        if latency_total_ms > max_latency_ms:
            alerts.append("SLOW_RUNTIME")
        if drift > max_drift:
            alerts.append("FEATURE_DRIFT")
        if ent_gap > max_entropy_gap:
            alerts.append("ENTROPY_WORSE_THAN_PROTO")
        if conf_primary < min_conf_primary:
            alerts.append("LOW_CONFIDENCE")
        if js > max_js:
            alerts.append("DISAGREE_WITH_PROTO")

        acc_primary = accuracy_from_logits(logits_primary, qy)
        acc_proto = accuracy_from_logits(logits_proto, qy)

        rollback_used = 0
        final_method = primary
        logits_final = logits_primary

        if len(alerts) > 0 and primary != "prototype":
            rollback_used = 1
            final_method = "prototype"
            logits_final = logits_proto

        acc_final = accuracy_from_logits(logits_final, qy)

        rows.append(
            {
                "episode": ep,
                "primary_method": primary,
                "final_method": final_method,
                "acc_primary": acc_primary,
                "acc_proto": acc_proto,
                "acc_final": acc_final,
                "latency_total_ms": latency_total_ms,
                "feature_drift": drift,
                "entropy_primary": ent_primary,
                "entropy_proto": ent_proto,
                "entropy_gap": ent_gap,
                "conf_primary": conf_primary,
                "conf_proto": conf_proto,
                "js_div": js,
                "alerts": "|".join(alerts),
                "alerts_count": len(alerts),
                "rollback_used": rollback_used,
            }
        )

        if ep % max(1, episodes // 10) == 0:
            print(
                f"Episode {ep}/{episodes} | acc_final={acc_final:.4f} | "
                f"primary={primary} -> final={final_method} | alerts={len(alerts)} | "
                f"latency_total_ms={latency_total_ms:.2f}"
            )

    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["episode"], df["acc_final"])
    plt.xlabel("Episode")
    plt.ylabel("Final Accuracy")
    plt.title("Final Accuracy (after rollback if used)")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_final.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(df["episode"], df["alerts_count"])
    plt.xlabel("Episode")
    plt.ylabel("Alerts Count")
    plt.title("Alerts per Episode")
    plt.tight_layout()
    plt.savefig(out_dir / "alerts.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(df["episode"], df["latency_total_ms"])
    plt.xlabel("Episode")
    plt.ylabel("Latency total (ms)")
    plt.title("Total Latency (Embeddings + Head + Sentinel)")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_total.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data_omniglot")
    parser.add_argument("--out_dir", type=str, default="./runs/m2_fixed")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=200)

    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_lr", type=float, default=1e-3)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--primary", type=str, default="ridge", choices=["ridge", "prototype"])
    parser.add_argument("--ridge_lambda", type=float, default=0.1)

    # Sentinel thresholds (good starting values)
    parser.add_argument("--max_latency_ms", type=float, default=650.0)     # based on your M1 ~570ms p95
    parser.add_argument("--max_drift", type=float, default=2.5)
    parser.add_argument("--max_entropy_gap", type=float, default=0.25)     # rollback only if ridge is much worse than proto
    parser.add_argument("--min_conf_primary", type=float, default=0.45)    # rollback if primary too uncertain
    parser.add_argument("--max_js", type=float, default=0.12)              # rollback if big disagreement

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=84)
    parser.add_argument("--invert", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "backbone.pt"

    tfms = [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
    if args.invert:
        tfms.append(transforms.Lambda(lambda x: 1.0 - x))
    transform = transforms.Compose(tfms)

    print("Loading Omniglot datasets...")
    bg = datasets.Omniglot(root=args.data_root, background=True, download=True, transform=transform)
    ev = datasets.Omniglot(root=args.data_root, background=False, download=True, transform=transform)
    print(f"Background size: {len(bg)} | Evaluation size: {len(ev)}")

    backbone = Conv4Backbone(embed_dim=args.embed_dim).to(device)
    if load_backbone_checkpoint(backbone, ckpt_path, device):
        print(f"Loaded backbone checkpoint: {ckpt_path}")
    else:
        train_backbone_on_background(
            device=device,
            backbone=backbone,
            train_dataset=bg,
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            lr=args.train_lr,
            out_path=ckpt_path,
        )

    print("\nRunning M2 Fixed (Sentinel + Rollback)...")
    df = run_m2(
        device=device,
        backbone=backbone,
        eval_dataset=ev,
        ways=args.ways,
        shots=args.shots,
        queries=args.queries,
        episodes=args.episodes,
        primary=args.primary,
        ridge_lambda=args.ridge_lambda,
        max_latency_ms=args.max_latency_ms,
        max_drift=args.max_drift,
        max_entropy_gap=args.max_entropy_gap,
        min_conf_primary=args.min_conf_primary,
        max_js=args.max_js,
    )

    csv_path = out_dir / "episode_metrics_m2_fixed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV: {csv_path}")

    save_plots(df, out_dir)
    print(f"Saved plots in: {out_dir}")

    print("\nSummary:")
    print(f"  episodes: {len(df)}")
    print(f"  mean acc_primary: {df['acc_primary'].mean():.4f}")
    print(f"  mean acc_final:   {df['acc_final'].mean():.4f}")
    print(f"  rollback rate:    {df['rollback_used'].mean():.4f}")
    print(f"  mean alerts:      {df['alerts_count'].mean():.4f}")
    print(f"  mean latency_total_ms: {df['latency_total_ms'].mean():.2f}")


if __name__ == "__main__":
    main()