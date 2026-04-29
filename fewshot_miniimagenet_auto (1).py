from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import kagglehub


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mean_confidence(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1)
    return float(probs.max(dim=1).values.mean().item())


def mean_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1).clamp(min=1e-12)
    return float((-(probs * probs.log()).sum(dim=1)).mean().item())


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    pa = F.softmax(logits_a, dim=1).clamp(min=1e-12)
    pb = F.softmax(logits_b, dim=1).clamp(min=1e-12)
    m = 0.5 * (pa + pb)
    kl_a = (pa * (pa.log() - m.log())).sum(dim=1)
    kl_b = (pb * (pb.log() - m.log())).sum(dim=1)
    js = 0.5 * (kl_a + kl_b)
    return float(js.mean().item())


def acc_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    return float((torch.argmax(logits, dim=1) == y_true).float().mean().item())


# =============================================================================
# Kaggle Download
# =============================================================================

def download_dataset(dataset_handle: str, download_dir: str, force_download: bool) -> Path:
    output_dir = Path(download_dir)
    ensure_dir(output_dir)

    print(f"Downloading dataset from Kaggle: {dataset_handle}")
    try:
        local_path = kagglehub.dataset_download(
            dataset_handle,
            output_dir=str(output_dir),
            force_download=force_download,
        )
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed.\n"
            "Please configure Kaggle auth first.\n"
            "You can use one of these:\n"
            "1) kagglehub.login()\n"
            "2) ~/.kaggle/kaggle.json\n"
            "3) ~/.kaggle/access_token\n"
            "4) KAGGLE_API_TOKEN environment variable\n"
            f"\nOriginal error: {exc}"
        ) from exc

    local_path = Path(local_path)
    print(f"Downloaded path: {local_path}")
    return local_path


# =============================================================================
# Detect flat class-folder root
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def folder_has_images(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob("*"))


def count_valid_class_dirs(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0

    count = 0
    for sub in folder.iterdir():
        if sub.is_dir() and sub.name != ".complete" and folder_has_images(sub):
            count += 1
    return count


def find_flat_class_root(base_path: Path) -> Path:
    """
    Finds a directory containing many class folders directly under it, like:
        root/
            n01532829/
            n01558993/
            ...
    """
    candidates = [base_path] + [p for p in base_path.rglob("*") if p.is_dir()]
    best_candidate = None
    best_count = 0

    for cand in candidates:
        class_count = count_valid_class_dirs(cand)
        if class_count > best_count:
            best_candidate = cand
            best_count = class_count

    if best_candidate is None or best_count < 20:
        print("\nDEBUG: top-level folders found:\n")
        if base_path.exists():
            for p in sorted(base_path.iterdir()):
                print(p)
        raise FileNotFoundError(
            f"Could not find flat class-folder dataset root under: {base_path}"
        )

    print(f"Detected flat class-folder root: {best_candidate} ({best_count} classes)")
    return best_candidate


# =============================================================================
# Dataset
# =============================================================================

class FlatClassImageDataset(Dataset):
    """
    Expects flat class folders like:
        root/
            n01532829/
            n01558993/
            ...

    Each dataset instance receives a specific list of class folder paths.
    Labels are remapped locally from 0...num_classes-1.
    """

    def __init__(self, class_dirs: List[Path], transform=None):
        self.class_dirs = sorted(class_dirs, key=lambda p: p.name)
        self.transform = transform

        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[Path, int]] = []

        for class_idx, class_dir in enumerate(self.class_dirs):
            self.class_to_idx[class_dir.name] = class_idx

            for img_path in sorted(class_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise ValueError("No images found in selected class directories.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_all_class_dirs(root: Path) -> List[Path]:
    class_dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name != ".complete" and folder_has_images(p):
            class_dirs.append(p)
    return class_dirs


def split_class_dirs(
    class_dirs: List[Path],
    seed: int,
    train_classes: int = 64,
    val_classes: int = 16,
    test_classes: int = 20,
) -> Tuple[List[Path], List[Path], List[Path]]:
    total_needed = train_classes + val_classes + test_classes
    if len(class_dirs) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} classes, but found only {len(class_dirs)}."
        )

    rng = random.Random(seed)
    shuffled = class_dirs.copy()
    rng.shuffle(shuffled)

    train_dirs = shuffled[:train_classes]
    val_dirs = shuffled[train_classes:train_classes + val_classes]
    test_dirs = shuffled[train_classes + val_classes:train_classes + val_classes + test_classes]

    return train_dirs, val_dirs, test_dirs


def build_class_to_indices(dataset) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


# =============================================================================
# Backbone
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv4Backbone(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
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


class PretrainNet(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 256):
        super().__init__()
        self.backbone = Conv4Backbone(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.classifier(emb)


def load_backbone_checkpoint(backbone: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    if "backbone_state" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

    backbone.load_state_dict(checkpoint["backbone_state"])


# =============================================================================
# Supervised pretraining
# =============================================================================

def evaluate_supervised(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += float(loss.item()) * x.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == y).sum().item())
            total_count += x.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def train_backbone_if_missing(
    train_dataset: Dataset,
    val_dataset: Dataset,
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
    pretrain_epochs: int,
    learning_rate: float,
    num_workers: int,
    embed_dim: int,
) -> None:
    if ckpt_path.exists():
        print(f"Checkpoint already exists, skipping training: {ckpt_path}")
        return

    print("Checkpoint not found. Starting automatic backbone training...")

    num_train_classes = len(train_dataset.class_to_idx)
    model = PretrainNet(num_classes=num_train_classes, embed_dim=embed_dim).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            running_correct += int((torch.argmax(logits, dim=1) == y).sum().item())
            running_count += x.size(0)

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        val_loss, val_acc = evaluate_supervised(model, val_loader, device)

        print(
            f"[Pretrain] Epoch {epoch}/{pretrain_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "backbone_state": model.backbone.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }

    if best_state is None:
        raise RuntimeError("Training completed but no checkpoint was saved.")

    ensure_dir(ckpt_path.parent)
    torch.save(best_state, ckpt_path)
    print(f"Best pretrained backbone saved to: {ckpt_path}")


# =============================================================================
# Few-shot episode sampling
# =============================================================================

@dataclass
class Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def sample_episode(dataset, class_to_indices, ways: int, shots: int, queries: int) -> Episode:
    eligible = [c for c, idxs in class_to_indices.items() if len(idxs) >= (shots + queries)]
    if len(eligible) < ways:
        raise ValueError(f"Not enough eligible classes. Need {ways}, found {len(eligible)}.")

    chosen = random.sample(eligible, ways)
    label_map = {cls: i for i, cls in enumerate(chosen)}

    support_x, support_y, query_x, query_y = [], [], [], []

    for cls in chosen:
        idxs = random.sample(class_to_indices[cls], shots + queries)
        support_indices = idxs[:shots]
        query_indices = idxs[shots:]

        for idx in support_indices:
            x, _ = dataset[idx]
            support_x.append(x)
            support_y.append(label_map[cls])

        for idx in query_indices:
            x, _ = dataset[idx]
            query_x.append(x)
            query_y.append(label_map[cls])

    return Episode(
        support_x=torch.stack(support_x, dim=0),
        support_y=torch.tensor(support_y, dtype=torch.long),
        query_x=torch.stack(query_x, dim=0),
        query_y=torch.tensor(query_y, dtype=torch.long),
    )


# =============================================================================
# Few-shot methods
# =============================================================================

def compute_prototypes(support_emb: torch.Tensor, support_y: torch.Tensor, ways: int) -> torch.Tensor:
    protos = []
    for c in range(ways):
        protos.append(support_emb[support_y == c].mean(dim=0))
    return torch.stack(protos, dim=0)


def logits_from_prototypes(query_emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    x2 = (query_emb ** 2).sum(dim=1, keepdim=True)
    p2 = (protos ** 2).sum(dim=1).unsqueeze(0)
    xp = query_emb @ protos.t()
    d2 = x2 + p2 - 2 * xp
    return -d2


def ridge_fit(support_emb: torch.Tensor, support_y: torch.Tensor, ways: int, ridge_lambda: float) -> torch.Tensor:
    X = support_emb
    _, embed_dim = X.shape
    Y = F.one_hot(support_y, num_classes=ways).float()
    XtX = X.t() @ X
    reg = ridge_lambda * torch.eye(embed_dim, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, X.t() @ Y)
    return W


# =============================================================================
# Fault injection
# =============================================================================

def inject_faults(emb: torch.Tensor, sigma: float, drop_prob: float, seed: int) -> torch.Tensor:
    if sigma <= 0.0 and drop_prob <= 0.0:
        return emb

    generator = torch.Generator(device=emb.device)
    generator.manual_seed(seed)

    out = emb

    if sigma > 0.0:
        noise = torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * sigma
        out = out + noise

    if drop_prob > 0.0:
        mask = (torch.rand(out.shape, generator=generator, device=out.device) > drop_prob).to(out.dtype)
        out = out * mask

    out = F.normalize(out, p=2, dim=1)
    return out


# =============================================================================
# Agent + sentinel
# =============================================================================

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


@dataclass
class AgentConfig:
    val_per_class: int = 1
    cv_margin: float = 0.02
    explore_prob: float = 0.10
    close_band: float = 0.01
    proto_conf_high: float = 0.92


@dataclass
class SentinelConfig:
    max_latency_ms: float = 900.0
    max_entropy: float = 1.35
    min_conf: float = 0.35
    max_js: float = 0.25


# =============================================================================
# Strategy runner
# =============================================================================

def run_strategy(
    strategy_name: str,
    device: torch.device,
    backbone: nn.Module,
    dataset,
    class_to_indices: Dict[int, List[int]],
    ways: int,
    shots: int,
    queries: int,
    episodes: int,
    ridge_lambda: float,
    fault_sigma: float,
    fault_drop_prob: float,
    agent_cfg: AgentConfig,
    sentinel_cfg: SentinelConfig,
    enable_sentinel: bool,
    seed: int,
) -> pd.DataFrame:
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    rows = []

    for ep in range(1, episodes + 1):
        episode = sample_episode(dataset, class_to_indices, ways, shots, queries)
        t0 = time.perf_counter()

        with torch.no_grad():
            sx = episode.support_x.to(device)
            qx = episode.query_x.to(device)
            sy = episode.support_y.to(device)
            qy = episode.query_y.to(device)

            s_emb = backbone(sx)
            q_emb = backbone(qx)

            s_emb_f = inject_faults(
                s_emb,
                sigma=fault_sigma,
                drop_prob=fault_drop_prob,
                seed=seed + ep * 11 + 1,
            )
            q_emb_f = inject_faults(
                q_emb,
                sigma=fault_sigma,
                drop_prob=fault_drop_prob,
                seed=seed + ep * 11 + 2,
            )

            protos_full = compute_prototypes(s_emb_f, sy, ways)
            logits_proto = logits_from_prototypes(q_emb_f, protos_full)

            logits_chosen = logits_proto
            chosen = "prototype"
            final = "prototype"
            alerts = []
            decision_hold_proto = np.nan
            decision_hold_ridge = np.nan

            if strategy_name == "proto":
                logits_chosen = logits_proto

            elif strategy_name == "ridge":
                chosen = "ridge"
                W_full = ridge_fit(s_emb_f, sy, ways, ridge_lambda=ridge_lambda)
                logits_chosen = q_emb_f @ W_full

            elif strategy_name in ("agent", "agent_sentinel"):
                conf_proto_q = mean_confidence(logits_proto)
                try_ridge = (conf_proto_q < agent_cfg.proto_conf_high) or (
                    random.random() < agent_cfg.explore_prob
                )

                if try_ridge and shots >= 2:
                    train_mask, val_mask = make_holdout_masks(
                        support_y=sy,
                        ways=ways,
                        shots=shots,
                        val_per_class=agent_cfg.val_per_class,
                        seed=seed + ep,
                    )

                    protos_tr = compute_prototypes(s_emb_f[train_mask], sy[train_mask], ways)
                    logits_val_proto = logits_from_prototypes(s_emb_f[val_mask], protos_tr)
                    hold_proto = acc_from_logits(logits_val_proto, sy[val_mask])

                    W_tr = ridge_fit(
                        s_emb_f[train_mask],
                        sy[train_mask],
                        ways,
                        ridge_lambda=ridge_lambda,
                    )
                    logits_val_ridge = s_emb_f[val_mask] @ W_tr
                    hold_ridge = acc_from_logits(logits_val_ridge, sy[val_mask])

                    decision_hold_proto = float(hold_proto)
                    decision_hold_ridge = float(hold_ridge)

                    if (hold_ridge > hold_proto + agent_cfg.cv_margin) or (
                        abs(hold_ridge - hold_proto) <= agent_cfg.close_band
                        and random.random() < agent_cfg.explore_prob
                    ):
                        chosen = "ridge"
                        logits_chosen = q_emb_f @ W_tr
                    else:
                        chosen = "prototype"
                        logits_chosen = logits_proto
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if enable_sentinel:
            ent = mean_entropy(logits_chosen)
            conf = mean_confidence(logits_chosen)
            js = js_divergence(logits_chosen, logits_proto) if logits_chosen is not logits_proto else 0.0

            if latency_ms > sentinel_cfg.max_latency_ms:
                alerts.append("SLOW_RUNTIME")
            if ent > sentinel_cfg.max_entropy:
                alerts.append("HIGH_ENTROPY")
            if conf < sentinel_cfg.min_conf:
                alerts.append("LOW_CONF")
            if js > sentinel_cfg.max_js:
                alerts.append("DISAGREE_WITH_PROTO")

            if chosen == "ridge" and len(alerts) > 0:
                final = "prototype"
                logits_final = logits_proto
            else:
                final = chosen
                logits_final = logits_chosen
        else:
            final = chosen
            logits_final = logits_chosen
            ent = mean_entropy(logits_final)
            conf = mean_confidence(logits_final)
            js = js_divergence(logits_final, logits_proto) if logits_final is not logits_proto else 0.0

        acc_final = acc_from_logits(logits_final, qy)
        acc_proto = acc_from_logits(logits_proto, qy)

        rows.append(
            {
                "episode": ep,
                "strategy": strategy_name,
                "ways": ways,
                "shots": shots,
                "queries": queries,
                "fault_sigma": fault_sigma,
                "fault_drop_prob": fault_drop_prob,
                "chosen_method": chosen,
                "final_method": final,
                "rollback_used": int(final != chosen),
                "acc_final": float(acc_final),
                "acc_proto": float(acc_proto),
                "latency_ms": float(latency_ms),
                "entropy_final": float(ent),
                "conf_final": float(conf),
                "js_vs_proto": float(js),
                "alerts": "|".join(alerts),
                "alerts_count": int(len(alerts)),
                "holdout_acc_proto": decision_hold_proto,
                "holdout_acc_ridge": decision_hold_ridge,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Reporting
# =============================================================================

def summarize_runs(all_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["shots", "fault_sigma", "fault_drop_prob", "strategy"]
    grouped = all_df.groupby(group_cols, dropna=False)

    summary = grouped.agg(
        episodes=("episode", "count"),
        mean_acc=("acc_final", "mean"),
        p50_acc=("acc_final", "median"),
        mean_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda s: float(np.quantile(s.values, 0.95))),
        rollback_rate=("rollback_used", "mean"),
        mean_alerts=("alerts_count", "mean"),
        ridge_chosen_rate=("chosen_method", lambda s: float(np.mean(s.values == "ridge"))),
    ).reset_index()

    for col in [
        "mean_acc",
        "p50_acc",
        "mean_latency_ms",
        "p95_latency_ms",
        "rollback_rate",
        "mean_alerts",
        "ridge_chosen_rate",
    ]:
        summary[col] = summary[col].astype(float).round(4)

    return summary.sort_values(
        ["shots", "fault_sigma", "fault_drop_prob", "strategy"]
    ).reset_index(drop=True)


def save_plots(summary_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)

    for shots in sorted(summary_df["shots"].unique()):
        sdf = summary_df[summary_df["shots"] == shots].copy()

        plt.figure()
        for strat in sorted(sdf["strategy"].unique()):
            ss = sdf[sdf["strategy"] == strat].sort_values("fault_sigma")
            plt.plot(ss["fault_sigma"].values, ss["mean_acc"].values, marker="o", label=strat)
        plt.xlabel("Fault sigma")
        plt.ylabel("Mean accuracy")
        plt.title(f"miniImageNet: Mean Accuracy vs Fault Level (shots={shots})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"miniimagenet_mean_acc_vs_fault_shots_{shots}.png", dpi=160)
        plt.close()

    for shots in sorted(summary_df["shots"].unique()):
        sdf = summary_df[
            (summary_df["shots"] == shots)
            & (summary_df["fault_sigma"] == 0.0)
            & (summary_df["fault_drop_prob"] == 0.0)
        ].copy()

        if len(sdf) == 0:
            continue

        sdf = sdf.sort_values("strategy")
        plt.figure()
        plt.bar(sdf["strategy"].values, sdf["mean_latency_ms"].values)
        plt.xlabel("Strategy")
        plt.ylabel("Mean latency (ms)")
        plt.title(f"miniImageNet: Latency Comparison (shots={shots}, no fault)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"miniimagenet_latency_compare_shots_{shots}.png", dpi=160)
        plt.close()

    for shots in sorted(summary_df["shots"].unique()):
        sdf = summary_df[
            (summary_df["shots"] == shots)
            & (summary_df["strategy"] == "agent_sentinel")
        ].sort_values("fault_sigma")

        if len(sdf) == 0:
            continue

        plt.figure()
        plt.plot(sdf["fault_sigma"].values, sdf["rollback_rate"].values, marker="o")
        plt.xlabel("Fault sigma")
        plt.ylabel("Rollback rate")
        plt.title(f"miniImageNet: Rollback Rate vs Fault Level (agent_sentinel, shots={shots})")
        plt.tight_layout()
        plt.savefig(out_dir / f"miniimagenet_rollback_vs_fault_shots_{shots}.png", dpi=160)
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Download
    parser.add_argument("--dataset_handle", type=str, default="arjunashok33/miniimagenet")
    parser.add_argument("--download_dir", type=str, default="./miniimagenet_download")
    parser.add_argument("--force_download", action="store_true")

    # Checkpoint
    parser.add_argument("--ckpt_path", type=str, default="runs/m1_miniimagenet_proto/backbone.pt")

    # Pretraining
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=256)

    # Evaluation
    parser.add_argument("--out_dir", type=str, default="./final_runs_miniimagenet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots_list", type=str, default="1,5")
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--ridge_lambda", type=float, default=0.1)
    parser.add_argument("--fault_sigmas", type=str, default="0,0.05,0.10")
    parser.add_argument("--fault_drop_prob", type=float, default=0.0)
    parser.add_argument("--strategies", type=str, default="proto,ridge,agent,agent_sentinel")

    # Agent config
    parser.add_argument("--val_per_class", type=int, default=1)
    parser.add_argument("--cv_margin", type=float, default=0.02)
    parser.add_argument("--explore_prob", type=float, default=0.10)
    parser.add_argument("--close_band", type=float, default=0.01)
    parser.add_argument("--proto_conf_high", type=float, default=0.92)

    # Sentinel config
    parser.add_argument("--max_latency_ms", type=float, default=900.0)
    parser.add_argument("--max_entropy", type=float, default=1.35)
    parser.add_argument("--min_conf", type=float, default=0.35)
    parser.add_argument("--max_js", type=float, default=0.25)

    parser.add_argument("--img_size", type=int, default=84)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 1. Download
    downloaded_path = download_dataset(
        dataset_handle=args.dataset_handle,
        download_dir=args.download_dir,
        force_download=args.force_download,
    )

    # 2. Detect flat class-folder root
    dataset_root = find_flat_class_root(downloaded_path)

    # 3. Split classes into train/val/test
    all_class_dirs = get_all_class_dirs(dataset_root)
    train_class_dirs, val_class_dirs, test_class_dirs = split_class_dirs(
        class_dirs=all_class_dirs,
        seed=args.seed,
        train_classes=64,
        val_classes=16,
        test_classes=20,
    )

    print(f"Train classes: {len(train_class_dirs)}")
    print(f"Val classes:   {len(val_class_dirs)}")
    print(f"Test classes:  {len(test_class_dirs)}")

    # 4. Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # 5. Build datasets
    train_ds = FlatClassImageDataset(train_class_dirs, transform=train_tfms)
    val_ds = FlatClassImageDataset(val_class_dirs, transform=eval_tfms)
    test_ds = FlatClassImageDataset(test_class_dirs, transform=eval_tfms)

    print(f"Train images: {len(train_ds)}")
    print(f"Val images:   {len(val_ds)}")
    print(f"Test images:  {len(test_ds)}")

    # 6. Train backbone if checkpoint missing
    ckpt_path = Path(args.ckpt_path)
    train_backbone_if_missing(
        train_dataset=train_ds,
        val_dataset=val_ds,
        ckpt_path=ckpt_path,
        device=device,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        embed_dim=args.embed_dim,
    )

    # 7. Load trained backbone
    backbone = Conv4Backbone(embed_dim=args.embed_dim).to(device)
    load_backbone_checkpoint(backbone, ckpt_path, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    print(f"Loaded backbone checkpoint: {ckpt_path}")

    # 8. Few-shot evaluation on test classes
    class_to_indices = build_class_to_indices(test_ds)
    print(f"Eligible evaluation classes: {len(class_to_indices)}")

    shots_list = [int(x.strip()) for x in args.shots_list.split(",") if x.strip()]
    fault_sigmas = [float(x.strip()) for x in args.fault_sigmas.split(",") if x.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    agent_cfg = AgentConfig(
        val_per_class=args.val_per_class,
        cv_margin=args.cv_margin,
        explore_prob=args.explore_prob,
        close_band=args.close_band,
        proto_conf_high=args.proto_conf_high,
    )

    sentinel_cfg = SentinelConfig(
        max_latency_ms=args.max_latency_ms,
        max_entropy=args.max_entropy,
        min_conf=args.min_conf,
        max_js=args.max_js,
    )

    run_dir = Path(args.out_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    ensure_dir(run_dir)

    all_frames = []

    for shots in shots_list:
        for sigma in fault_sigmas:
            for strategy in strategies:
                enable_sentinel = strategy == "agent_sentinel"

                print(f"\n=== Running: strategy={strategy} | shots={shots} | fault_sigma={sigma} ===")

                df = run_strategy(
                    strategy_name=strategy,
                    device=device,
                    backbone=backbone,
                    dataset=test_ds,
                    class_to_indices=class_to_indices,
                    ways=args.ways,
                    shots=shots,
                    queries=args.queries,
                    episodes=args.episodes,
                    ridge_lambda=args.ridge_lambda,
                    fault_sigma=sigma,
                    fault_drop_prob=args.fault_drop_prob,
                    agent_cfg=agent_cfg,
                    sentinel_cfg=sentinel_cfg,
                    enable_sentinel=enable_sentinel,
                    seed=args.seed,
                )

                output_csv = run_dir / f"metrics_{strategy}_shots{shots}_sigma{sigma}.csv"
                df.to_csv(output_csv, index=False)
                print(f"Saved: {output_csv}")
                all_frames.append(df)

    all_df = pd.concat(all_frames, ignore_index=True)
    all_path = run_dir / "ALL_EPISODES.csv"
    all_df.to_csv(all_path, index=False)

    summary = summarize_runs(all_df)
    summary_path = run_dir / "SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    print("\n================ FINAL SUMMARY ================")
    print(summary.to_string(index=False))
    print("================================================")
    print(f"\nSaved ALL_EPISODES: {all_path}")
    print(f"Saved SUMMARY:      {summary_path}")

    plots_dir = run_dir / "plots"
    save_plots(summary, plots_dir)
    print(f"Saved plots in: {plots_dir}")


if __name__ == "__main__":
    main()