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
def compute_prototypes(support_emb: torch.Tensor, support_y: torch.Tensor, ways: int) -> torch.Tensor:
    protos = []
    for c in range(ways):
        protos.append(support_emb[support_y == c].mean(dim=0))
    return torch.stack(protos, dim=0)  # [ways, D]


def logits_from_prototypes(query_emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    x2 = (query_emb ** 2).sum(dim=1, keepdim=True)
    p2 = (protos ** 2).sum(dim=1).unsqueeze(0)
    xp = query_emb @ protos.t()
    d2 = x2 + p2 - 2 * xp
    return -d2


def ridge_fit(support_emb: torch.Tensor, support_y: torch.Tensor, ways: int, ridge_lambda: float) -> torch.Tensor:
    X = support_emb  # [S, D]
    _, D = X.shape
    Y = F.one_hot(support_y, num_classes=ways).float()  # [S, ways]
    XtX = X.t() @ X
    reg = ridge_lambda * torch.eye(D, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, X.t() @ Y)  # [D, ways]
    return W


def acc_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    return float((torch.argmax(logits, dim=1) == y_true).float().mean().item())


def mean_confidence(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1)
    return float(p.max(dim=1).values.mean().item())


# -----------------------------
# Drift monitor
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
# Checkpoint loader
# -----------------------------
def load_ckpt(backbone: nn.Module, path: Path, device: torch.device) -> bool:
    if not path.exists():
        return False
    ckpt = torch.load(str(path), map_location=device)
    if "backbone_state" not in ckpt:
        return False
    backbone.load_state_dict(ckpt["backbone_state"])
    return True


# -----------------------------
# Holdout split (1 sample/class default)
# -----------------------------
def make_holdout_masks(
    support_y: torch.Tensor,
    ways: int,
    shots: int,
    val_per_class: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if shots <= val_per_class:
        raise ValueError(f"shots ({shots}) must be > val_per_class ({val_per_class})")

    rng = np.random.RandomState(seed)
    val_mask = torch.zeros_like(support_y, dtype=torch.bool)

    for c in range(ways):
        idxs = torch.where(support_y == c)[0].cpu().numpy()
        rng.shuffle(idxs)
        val_idxs = idxs[:val_per_class]
        val_mask[val_idxs] = True

    train_mask = ~val_mask
    return train_mask, val_mask


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_omniglot")
    ap.add_argument("--out_dir", type=str, default="./runs/m3_agent_v4_fast_holdout")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ways", type=int, default=5)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=200)

    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--ridge_lambda", type=float, default=0.1)

    # Agent decision
    ap.add_argument("--val_per_class", type=int, default=1)
    ap.add_argument("--cv_margin", type=float, default=0.02)     # choose ridge only if holdout acc better by this
    ap.add_argument("--explore_prob", type=float, default=0.10)   # sometimes try ridge even if close (for demo)
    ap.add_argument("--close_band", type=float, default=0.01)

    # Skip ridge eval if prototype already confident on query (unlabeled signal)
    ap.add_argument("--proto_conf_high", type=float, default=0.92)

    # Safety sentinel
    ap.add_argument("--max_latency_total_ms", type=float, default=900.0)  # higher because includes decision time
    ap.add_argument("--max_drift", type=float, default=2.5)

    # Reuse your good backbone (no retrain)
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
    ev = datasets.Omniglot(root=args.data_root, background=False, download=True, transform=transform)
    print(f"Evaluation size: {len(ev)}")

    backbone = Conv4Backbone(embed_dim=args.embed_dim).to(device)
    ckpt_path = Path(args.ckpt_path)
    if not load_ckpt(backbone, ckpt_path, device):
        raise FileNotFoundError(f"Backbone checkpoint not found at: {ckpt_path}")

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    class_to_indices = build_class_to_indices(ev)
    drift_mon = DriftMonitor()

    rows = []
    ridge_tried = 0
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

            # ---- embeddings (dominant cost)
            s_emb = backbone(sx)
            q_emb = backbone(qx)

            drift_mon.update_ref(s_emb)
            drift = drift_mon.score(s_emb)

            # ---- prototype using FULL support (for query + fallback)
            protos_full = compute_prototypes(s_emb, sy, args.ways)
            logits_proto_q = logits_from_prototypes(q_emb, protos_full)
            conf_proto_q = mean_confidence(logits_proto_q)

            chosen = "prototype"
            final_method = "prototype"

            # ---- Decide whether to evaluate ridge (save compute if proto is already confident)
            try_ridge = (conf_proto_q < args.proto_conf_high) or (random.random() < args.explore_prob)

            acc_hold_proto = np.nan
            acc_hold_ridge = np.nan
            logits_ridge_q = None

            if try_ridge:
                ridge_tried += 1

                train_mask, val_mask = make_holdout_masks(
                    support_y=sy, ways=args.ways, shots=args.shots,
                    val_per_class=args.val_per_class, seed=args.seed + ep
                )

                # Holdout proto
                protos_tr = compute_prototypes(s_emb[train_mask], sy[train_mask], args.ways)
                logits_val_proto = logits_from_prototypes(s_emb[val_mask], protos_tr)
                acc_hold_proto = acc_from_logits(logits_val_proto, sy[val_mask])

                # Holdout ridge
                W = ridge_fit(s_emb[train_mask], sy[train_mask], args.ways, ridge_lambda=args.ridge_lambda)
                logits_val_ridge = s_emb[val_mask] @ W
                acc_hold_ridge = acc_from_logits(logits_val_ridge, sy[val_mask])

                # choose ridge if it wins by margin, or if very close and exploration triggers
                if (acc_hold_ridge > acc_hold_proto + args.cv_margin) or (
                    abs(acc_hold_ridge - acc_hold_proto) <= args.close_band and random.random() < args.explore_prob
                ):
                    chosen = "ridge"
                    ridge_chosen += 1
                    logits_ridge_q = q_emb @ W  # reuse W for query

            # ---- Apply chosen method (query)
            logits_chosen = logits_proto_q if chosen == "prototype" else logits_ridge_q

        latency_total_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Safety sentinel (hard)
        alerts = []
        if latency_total_ms > args.max_latency_total_ms:
            alerts.append("SLOW_RUNTIME")
        if drift > args.max_drift:
            alerts.append("FEATURE_DRIFT")

        # ---- Rollback (only if ridge chosen and unsafe)
        logits_final = logits_chosen
        final_method = chosen
        if chosen == "ridge" and len(alerts) > 0:
            logits_final = logits_proto_q
            final_method = "prototype"
            rollback_used += 1

        acc_proto = acc_from_logits(logits_proto_q, qy)
        acc_final = acc_from_logits(logits_final, qy)

        rows.append(
            {
                "episode": ep,
                "shots": args.shots,
                "try_ridge": int(try_ridge),
                "chosen_method": chosen,
                "final_method": final_method,
                "acc_proto": acc_proto,
                "acc_final": acc_final,
                "holdout_acc_proto": acc_hold_proto,
                "holdout_acc_ridge": acc_hold_ridge,
                "proto_conf_query": conf_proto_q,
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
                f"try_ridge={int(try_ridge)} chosen={chosen} final={final_method} | "
                f"latency={latency_total_ms:.2f}ms"
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "episode_metrics_m3_agent_v4_fast_holdout.csv"
    df.to_csv(csv_path, index=False)

    print("\nSaved:", csv_path)
    print("\nSummary:")
    print(f"  episodes: {len(df)}")
    print(f"  mean acc_final: {df['acc_final'].mean():.4f}")
    print(f"  ridge_tried_rate: {df['try_ridge'].mean():.4f}")
    print(f"  ridge_chosen_rate: {df['chosen_method'].eq('ridge').mean():.4f}")
    ridge_rows = df[df["chosen_method"] == "ridge"]
    print(f"  rollback_rate_over_ridge_chosen: {(ridge_rows['rollback_used'].mean() if len(ridge_rows) else 0.0):.4f}")
    print(f"  mean latency_total_ms: {df['latency_total_ms'].mean():.2f}")


if __name__ == "__main__":
    main()