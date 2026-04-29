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
            ConvBlock(1, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
        )
        self.fc = nn.Linear(64 * 5 * 5, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


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
# Methods
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
    _, D = X.shape
    Y = F.one_hot(support_y, num_classes=ways).float()  # [S, ways]
    XtX = X.t() @ X
    reg = ridge_lambda * torch.eye(D, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, X.t() @ Y)  # [D, ways]
    return query_emb @ W


def acc_from_logits(logits, y_true) -> float:
    return float((torch.argmax(logits, dim=1) == y_true).float().mean().item())


# -----------------------------
# Drift + sentinel
# -----------------------------
class DriftMonitor:
    def __init__(self):
        self.ref_mean = None
        self.ref_std = None
        self.eps = 1e-6

    def update_ref(self, emb: torch.Tensor) -> None:
        if self.ref_mean is None:
            self.ref_mean = emb.mean(dim=0).detach()
            self.ref_std = emb.std(dim=0).detach().clamp(min=self.eps)

    def score(self, emb: torch.Tensor) -> float:
        if self.ref_mean is None or self.ref_std is None:
            return 0.0
        cur_mean = emb.mean(dim=0)
        z = (cur_mean - self.ref_mean) / self.ref_std
        return float(z.abs().mean().item())


# -----------------------------
# CV decision signal (balanced folds: 1 sample/class per fold)
# -----------------------------
def build_balanced_folds(support_y: torch.Tensor, shots: int, seed: int) -> torch.Tensor:
    """
    Returns fold_id per support sample (len = ways*shots).
    Each fold contains exactly 1 sample per class (when folds == shots).
    """
    rng = np.random.RandomState(seed)
    fold_id = torch.zeros_like(support_y)

    ways = int(support_y.max().item()) + 1
    for c in range(ways):
        idxs = torch.where(support_y == c)[0].cpu().numpy()
        rng.shuffle(idxs)
        # assign fold 0..shots-1
        for j, idx in enumerate(idxs):
            fold_id[idx] = j  # 0..shots-1
    return fold_id


def cv_score_proto(s_emb, s_y, ways: int, fold_id: torch.Tensor, folds: int) -> float:
    accs = []
    for f in range(folds):
        tr = fold_id != f
        va = fold_id == f
        logits = prototype_logits(s_emb[tr], s_y[tr], s_emb[va], ways)
        accs.append(acc_from_logits(logits, s_y[va]))
    return float(np.mean(accs))


def cv_score_ridge(s_emb, s_y, ways: int, fold_id: torch.Tensor, folds: int, ridge_lambda: float) -> float:
    accs = []
    for f in range(folds):
        tr = fold_id != f
        va = fold_id == f
        logits = ridge_logits(s_emb[tr], s_y[tr], s_emb[va], ways, ridge_lambda=ridge_lambda)
        accs.append(acc_from_logits(logits, s_y[va]))
    return float(np.mean(accs))


# -----------------------------
# Pretrain/load checkpoint
# -----------------------------
def load_ckpt(backbone: nn.Module, path: Path, device: torch.device) -> bool:
    if not path.exists():
        return False
    ckpt = torch.load(str(path), map_location=device)
    if "backbone_state" not in ckpt:
        return False
    backbone.load_state_dict(ckpt["backbone_state"])
    return True


def pretrain_backbone(device, backbone, train_dataset, epochs, batch_size, lr, out_path: Path) -> None:
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

    print(f"Pretraining: classes={num_classes}, epochs={epochs}")
    for ep in range(1, epochs + 1):
        loss_sum, acc_sum, n = 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            emb = backbone(xb)
            logits = clf(emb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            acc_sum += acc_from_logits(logits.detach(), yb.detach())
            n += 1
        print(f"  epoch {ep}/{epochs} | loss={loss_sum/max(n,1):.4f} | acc={acc_sum/max(n,1):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_state": backbone.state_dict()}, str(out_path))
    print(f"Saved backbone: {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_omniglot")
    ap.add_argument("--out_dir", type=str, default="./runs/m3_agent_v3_cv")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ways", type=int, default=5)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=200)

    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--ridge_lambda", type=float, default=0.1)

    # Decision knobs
    ap.add_argument("--cv_margin", type=float, default=0.02, help="Choose ridge only if CV improves by this much")
    ap.add_argument("--explore_prob", type=float, default=0.10, help="When CV is close, sometimes try ridge")
    ap.add_argument("--close_band", type=float, default=0.01, help="CV close range for exploration")

    # Safety sentinel (hard)
    ap.add_argument("--max_latency_ms", type=float, default=650.0)
    ap.add_argument("--max_drift", type=float, default=2.5)

    # Pretrain
    ap.add_argument("--train_epochs", type=int, default=2)
    ap.add_argument("--train_batch_size", type=int, default=64)
    ap.add_argument("--train_lr", type=float, default=1e-3)

    # Important: reuse checkpoint to avoid retraining every run
    ap.add_argument("--ckpt_path", type=str, default="runs/m1_omniglot_proto/backbone.pt")

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--img_size", type=int, default=84)
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfms = [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
    if args.invert:
        tfms.append(transforms.Lambda(lambda x: 1.0 - x))
    transform = transforms.Compose(tfms)

    print("Loading Omniglot datasets...")
    bg = datasets.Omniglot(root=args.data_root, background=True, download=True, transform=transform)
    ev = datasets.Omniglot(root=args.data_root, background=False, download=True, transform=transform)
    print(f"Background size: {len(bg)} | Evaluation size: {len(ev)}")

    backbone = Conv4Backbone(embed_dim=args.embed_dim).to(device)

    ckpt_path = Path(args.ckpt_path)
    if load_ckpt(backbone, ckpt_path, device):
        print(f"Loaded backbone from: {ckpt_path}")
    else:
        # fallback: train and save into out_dir
        local_ckpt = out_dir / "backbone.pt"
        pretrain_backbone(device, backbone, bg, args.train_epochs, args.train_batch_size, args.train_lr, local_ckpt)

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    class_to_indices = build_class_to_indices(ev)
    drift_mon = DriftMonitor()

    rows = []
    ridge_chosen = 0
    rollback_used = 0

    for ep in range(1, args.episodes + 1):
        episode = sample_episode(ev, class_to_indices, args.ways, args.shots, args.queries)

        t0 = time.perf_counter()
        with torch.no_grad():
            sx = episode.support_x.to(device)
            qx = episode.query_x.to(device)
            sy = episode.support_y.to(device)
            qy = episode.query_y.to(device)

            # embeddings
            s_emb = backbone(sx)
            q_emb = backbone(qx)

            drift_mon.update_ref(s_emb)
            drift = drift_mon.score(s_emb)

            # CV scores (only if shots >= 2)
            if args.shots >= 2:
                fold_id = build_balanced_folds(sy, shots=args.shots, seed=args.seed + ep)
                folds = args.shots  # 1-per-class per fold
                cv_proto = cv_score_proto(s_emb, sy, args.ways, fold_id, folds)
                cv_ridge = cv_score_ridge(s_emb, sy, args.ways, fold_id, folds, ridge_lambda=args.ridge_lambda)
            else:
                cv_proto = np.nan
                cv_ridge = np.nan

            # Decide
            chosen = "prototype"
            if args.shots >= 2:
                if cv_ridge > cv_proto + args.cv_margin:
                    chosen = "ridge"
                else:
                    # exploration only when close (for demonstrating agent behavior)
                    if abs(cv_ridge - cv_proto) <= args.close_band and random.random() < args.explore_prob:
                        chosen = "ridge"

            # Compute logits for query using chosen method
            logits_proto_q = prototype_logits(s_emb, sy, q_emb, args.ways)
            if chosen == "ridge":
                logits_ridge_q = ridge_logits(s_emb, sy, q_emb, args.ways, ridge_lambda=args.ridge_lambda)
                logits_chosen = logits_ridge_q
            else:
                logits_chosen = logits_proto_q

            latency_total_ms = (time.perf_counter() - t0) * 1000.0

            # Safety sentinel
            alerts = []
            if latency_total_ms > args.max_latency_ms:
                alerts.append("SLOW_RUNTIME")
            if drift > args.max_drift:
                alerts.append("FEATURE_DRIFT")

            final_method = chosen
            logits_final = logits_chosen
            if chosen == "ridge" and len(alerts) > 0:
                final_method = "prototype"
                logits_final = logits_proto_q
                rollback_used += 1

            if chosen == "ridge":
                ridge_chosen += 1

            acc_proto = acc_from_logits(logits_proto_q, qy)
            acc_final = acc_from_logits(logits_final, qy)

        rows.append(
            {
                "episode": ep,
                "shots": args.shots,
                "cv_proto": cv_proto,
                "cv_ridge": cv_ridge,
                "chosen_method": chosen,
                "final_method": final_method,
                "acc_proto": acc_proto,
                "acc_final": acc_final,
                "feature_drift": drift,
                "latency_total_ms": latency_total_ms,
                "alerts": "|".join(alerts),
                "alerts_count": len(alerts),
                "rollback_used": int(final_method != chosen),
            }
        )

        if ep % max(1, args.episodes // 10) == 0:
            print(
                f"Episode {ep}/{args.episodes} | acc_final={acc_final:.4f} | "
                f"cv_proto={cv_proto:.3f} cv_ridge={cv_ridge:.3f} | chosen={chosen} final={final_method} | "
                f"latency={latency_total_ms:.2f}ms"
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "episode_metrics_m3_agent_v3_cv.csv"
    df.to_csv(csv_path, index=False)

    print("\nSaved:", csv_path)
    print("\nSummary:")
    print("  episodes:", len(df))
    print("  mean acc_final:", f"{df['acc_final'].mean():.4f}")
    print("  ridge_chosen_rate:", f"{(df['chosen_method'].eq('ridge').mean()):.4f}")
    print("  rollback_rate_over_ridge_chosen:", f"{(df[df['chosen_method']=='ridge']['rollback_used'].mean() if (df['chosen_method']=='ridge').any() else 0.0):.4f}")
    print("  mean latency_total_ms:", f"{df['latency_total_ms'].mean():.2f}")


if __name__ == "__main__":
    main()