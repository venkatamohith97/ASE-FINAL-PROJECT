"""
Milestone-1: Omniglot Few-Shot Baseline (Prototypical Networks)

What this script does:
1) Loads Omniglot background split (for optional quick supervised pretraining).
2) Loads Omniglot evaluation split (for few-shot episodes).
3) Trains a small Conv4 backbone (optional).
4) Runs N-way K-shot episodes using a prototype head (no training in episode).
5) Saves:
   - CSV with per-episode metrics
   - PNG plots (accuracy, latency, etc.)

No Streamlit. CPU-friendly. Works on laptop.
"""

from __future__ import annotations

import argparse
import os
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
# Simple Conv4 backbone (common in few-shot)
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
    """
    Input: 1x84x84 (Omniglot grayscale)
    Output: embedding vector
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),   # 84 -> 42
            ConvBlock(64, 64),  # 42 -> 21
            ConvBlock(64, 64),  # 21 -> 10
            ConvBlock(64, 64),  # 10 -> 5
        )
        # 64 channels * 5 * 5 = 1600
        self.fc = nn.Linear(64 * 5 * 5, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # helps prototype distance
        return x


# -----------------------------
# Supervised pretraining head (only for background training)
# -----------------------------
class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# -----------------------------
# Episode sampling helpers
# -----------------------------
def build_class_to_indices(dataset) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        y = int(y)
        if y not in class_to_indices:
            class_to_indices[y] = []
        class_to_indices[y].append(idx)
    return class_to_indices


@dataclass
class Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    chosen_classes: List[int]


def sample_episode(
    dataset,
    class_to_indices: Dict[int, List[int]],
    ways: int,
    shots: int,
    queries: int,
) -> Episode:
    # choose classes that have enough samples
    eligible = [c for c, idxs in class_to_indices.items() if len(idxs) >= (shots + queries)]
    if len(eligible) < ways:
        raise ValueError(f"Not enough eligible classes. Need {ways}, got {len(eligible)}.")

    chosen = random.sample(eligible, ways)

    support_x_list, support_y_list = [], []
    query_x_list, query_y_list = [], []

    # Map real class id -> episodic label 0..ways-1
    episodic_label_map = {cls: i for i, cls in enumerate(chosen)}

    for cls in chosen:
        idxs = random.sample(class_to_indices[cls], shots + queries)
        support_idxs = idxs[:shots]
        query_idxs = idxs[shots:]

        for si in support_idxs:
            x, _ = dataset[si]
            support_x_list.append(x)
            support_y_list.append(episodic_label_map[cls])

        for qi in query_idxs:
            x, _ = dataset[qi]
            query_x_list.append(x)
            query_y_list.append(episodic_label_map[cls])

    support_x = torch.stack(support_x_list, dim=0)
    support_y = torch.tensor(support_y_list, dtype=torch.long)
    query_x = torch.stack(query_x_list, dim=0)
    query_y = torch.tensor(query_y_list, dtype=torch.long)

    return Episode(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        chosen_classes=chosen,
    )


# -----------------------------
# Prototype head (no training)
# -----------------------------
def prototype_logits(
    support_emb: torch.Tensor,
    support_y: torch.Tensor,
    query_emb: torch.Tensor,
    ways: int,
) -> torch.Tensor:
    """
    support_emb: [ways*shots, D]
    support_y:   [ways*shots]
    query_emb:   [ways*queries, D]
    returns logits: [Qtotal, ways]
    """
    prototypes = []
    for c in range(ways):
        cls_emb = support_emb[support_y == c]
        prototypes.append(cls_emb.mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)  # [ways, D]

    # negative squared euclidean distance as logits
    # dist(x,p) = ||x - p||^2 = x^2 + p^2 - 2xp
    x2 = (query_emb ** 2).sum(dim=1, keepdim=True)         # [Q,1]
    p2 = (prototypes ** 2).sum(dim=1).unsqueeze(0)         # [1,ways]
    xp = query_emb @ prototypes.t()                        # [Q,ways]
    d2 = x2 + p2 - 2 * xp
    logits = -d2
    return logits


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y_true).float().mean().item())


# -----------------------------
# Training (optional pretrain)
# -----------------------------
def train_backbone_on_background(
    device: torch.device,
    backbone: nn.Module,
    train_dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    out_path: Path,
) -> None:
    if epochs <= 0:
        print("Skipping pretraining (train_epochs <= 0). Using random backbone.")
        return

    class_ids = sorted({int(train_dataset[i][1]) for i in range(len(train_dataset))})
    num_classes = len(class_ids)

    # Omniglot labels are already contiguous in torchvision, but we keep it safe:
    # build mapping label -> [0..C-1]
    label_map = {c: i for i, c in enumerate(class_ids)}

    def mapped_collate(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(x)
            ys.append(label_map[int(y)])
        return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=mapped_collate,
    )

    backbone.train()
    clf = LinearClassifier(in_dim=backbone.fc.out_features, num_classes=num_classes).to(device)

    params = list(backbone.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    print(f"Pretraining on Omniglot background: classes={num_classes}, epochs={epochs}")
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        ep_acc = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            emb = backbone(xb)
            logits = clf(emb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())
            ep_acc += accuracy_from_logits(logits.detach(), yb.detach())
            n_batches += 1

        print(f"  epoch {ep}/{epochs} | loss={ep_loss/max(n_batches,1):.4f} | acc={ep_acc/max(n_batches,1):.4f}")

    # Save backbone only (few-shot uses only embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_state": backbone.state_dict()}, str(out_path))
    print(f"Saved backbone checkpoint: {out_path}")


def load_backbone_checkpoint(backbone: nn.Module, ckpt_path: Path, device: torch.device) -> bool:
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "backbone_state" not in ckpt:
        return False
    backbone.load_state_dict(ckpt["backbone_state"])
    return True


# -----------------------------
# Main experiment runner
# -----------------------------
def run_fewshot_episodes(
    device: torch.device,
    backbone: nn.Module,
    eval_dataset,
    ways: int,
    shots: int,
    queries: int,
    episodes: int,
) -> pd.DataFrame:
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    class_to_indices = build_class_to_indices(eval_dataset)

    rows = []
    for ep in range(1, episodes + 1):
        episode = sample_episode(eval_dataset, class_to_indices, ways, shots, queries)

        t0 = time.perf_counter()
        with torch.no_grad():
            sx = episode.support_x.to(device)
            qx = episode.query_x.to(device)
            sy = episode.support_y.to(device)
            qy = episode.query_y.to(device)

            s_emb = backbone(sx)
            q_emb = backbone(qx)

            logits = prototype_logits(s_emb, sy, q_emb, ways=ways)
            acc = accuracy_from_logits(logits, qy)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "episode": ep,
                "ways": ways,
                "shots": shots,
                "queries": queries,
                "acc": acc,
                "latency_ms": latency_ms,
            }
        )

        if ep % max(1, episodes // 10) == 0:
            print(f"Episode {ep}/{episodes} | acc={acc:.4f} | latency_ms={latency_ms:.2f}")

    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accuracy over episodes
    plt.figure()
    plt.plot(df["episode"].values, df["acc"].values)
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Few-shot Accuracy (Prototype Head)")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_over_episodes.png", dpi=160)
    plt.close()

    # 2) Latency over episodes
    plt.figure()
    plt.plot(df["episode"].values, df["latency_ms"].values)
    plt.xlabel("Episode")
    plt.ylabel("Latency (ms)")
    plt.title("Per-episode Latency")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_over_episodes.png", dpi=160)
    plt.close()

    # 3) Histogram of accuracy
    plt.figure()
    plt.hist(df["acc"].values, bins=20)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.title("Accuracy Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_hist.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data_omniglot", help="Where to download/store Omniglot")
    parser.add_argument("--out_dir", type=str, default="./runs/m1_omniglot_proto", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    # Few-shot settings
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=200)

    # Pretraining (optional)
    parser.add_argument("--train_epochs", type=int, default=2, help="0 to skip pretraining")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_lr", type=float, default=1e-3)

    # Model
    parser.add_argument("--embed_dim", type=int, default=256)

    # Device
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # Image transform options
    parser.add_argument("--img_size", type=int, default=84)
    parser.add_argument("--invert", action="store_true", help="Invert Omniglot colors (recommended)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "backbone.pt"

    # Transforms
    tfms = [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
    if args.invert:
        tfms.append(transforms.Lambda(lambda x: 1.0 - x))
    transform = transforms.Compose(tfms)

    data_root = args.data_root

    # Load datasets
    print("Loading Omniglot datasets...")
    bg = datasets.Omniglot(root=data_root, background=True, download=True, transform=transform)
    ev = datasets.Omniglot(root=data_root, background=False, download=True, transform=transform)
    print(f"Background size: {len(bg)} | Evaluation size: {len(ev)}")

    # Build model
    backbone = Conv4Backbone(embed_dim=args.embed_dim).to(device)

    # Load checkpoint if exists; else train if requested
    loaded = load_backbone_checkpoint(backbone, ckpt_path, device)
    if loaded:
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

    # Few-shot evaluation
    print("\nRunning few-shot episodes on evaluation split...")
    df = run_fewshot_episodes(
        device=device,
        backbone=backbone,
        eval_dataset=ev,
        ways=args.ways,
        shots=args.shots,
        queries=args.queries,
        episodes=args.episodes,
    )

    # Save outputs
    csv_path = out_dir / "episode_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV: {csv_path}")

    save_plots(df, out_dir)
    print(f"Saved plots in: {out_dir}")

    # Print summary
    print("\nSummary:")
    print(f"  episodes: {len(df)}")
    print(f"  mean acc: {df['acc'].mean():.4f}")
    print(f"  mean latency_ms: {df['latency_ms'].mean():.2f}")
    print(f"  p50 acc: {df['acc'].median():.4f}")
    print(f"  p95 latency_ms: {df['latency_ms'].quantile(0.95):.2f}")


if __name__ == "__main__":
    main()