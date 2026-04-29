from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mean_confidence(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1)
    return float(p.max(dim=1).values.mean().item())


def mean_entropy(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1).clamp(min=1e-12)
    return float((-(p * p.log()).sum(dim=1)).mean().item())


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
# Backbone
# =============================================================================

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


def load_backbone_ckpt(backbone: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "backbone_state" not in ckpt:
        raise ValueError(f"Invalid checkpoint format (missing backbone_state): {ckpt_path}")
    backbone.load_state_dict(ckpt["backbone_state"])


# =============================================================================
# Episode sampling
# =============================================================================

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


# =============================================================================
# Methods: Prototype + Ridge
# =============================================================================

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
    X = support_emb
    _, D = X.shape
    Y = F.one_hot(support_y, num_classes=ways).float()
    XtX = X.t() @ X
    reg = ridge_lambda * torch.eye(D, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, X.t() @ Y)  # [D, ways]
    return W


# =============================================================================
# Fault injection (simulating edge noise / quant drift)
# =============================================================================

def inject_faults(
    emb: torch.Tensor,
    sigma: float,
    drop_prob: float,
    seed: int,
) -> torch.Tensor:
    """
    sigma: gaussian noise std
    drop_prob: feature dropout probability (simulates missing/unstable dims)
    """
    if sigma <= 0.0 and drop_prob <= 0.0:
        return emb

    g = torch.Generator(device=emb.device)
    g.manual_seed(seed)

    out = emb
    if sigma > 0.0:
        noise = torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype) * sigma
        out = out + noise

    if drop_prob > 0.0:
        mask = (torch.rand(out.shape, generator=g, device=out.device) > drop_prob).to(out.dtype)
        out = out * mask

    # renormalize (important because backbone outputs normalized vectors)
    out = F.normalize(out, p=2, dim=1)
    return out


# =============================================================================
# Agent decision (fast holdout)
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
# Strategies
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
    """
    strategy_name:
      - "proto"
      - "ridge"
      - "agent"            (fast holdout decision; no sentinel)
      - "agent_sentinel"   (fast holdout decision + sentinel rollback to proto)
    """
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

            # embeddings
            s_emb = backbone(sx)
            q_emb = backbone(qx)

            # fault injection (simulate edge noise)
            s_emb_f = inject_faults(s_emb, sigma=fault_sigma, drop_prob=fault_drop_prob, seed=seed + ep * 11 + 1)
            q_emb_f = inject_faults(q_emb, sigma=fault_sigma, drop_prob=fault_drop_prob, seed=seed + ep * 11 + 2)

            # always compute proto logits (cheap, also used for fallback)
            protos_full = compute_prototypes(s_emb_f, sy, ways)
            logits_proto = logits_from_prototypes(q_emb_f, protos_full)

            # compute ridge (if needed)
            logits_ridge = None
            W_full = None

            chosen = "prototype"
            final = "prototype"
            alerts = []

            # ---- decide / execute
            if strategy_name == "proto":
                chosen = "prototype"
                logits_chosen = logits_proto

            elif strategy_name == "ridge":
                chosen = "ridge"
                W_full = ridge_fit(s_emb_f, sy, ways, ridge_lambda=ridge_lambda)
                logits_ridge = q_emb_f @ W_full
                logits_chosen = logits_ridge

            elif strategy_name in ("agent", "agent_sentinel"):
                # quick early skip: if proto on query is already very confident, do not waste decision compute
                conf_proto_q = mean_confidence(logits_proto)
                try_ridge = (conf_proto_q < agent_cfg.proto_conf_high) or (random.random() < agent_cfg.explore_prob)

                hold_proto = np.nan
                hold_ridge = np.nan

                chosen = "prototype"
                logits_chosen = logits_proto

                if try_ridge and shots >= 2:
                    # holdout split within support
                    train_mask, val_mask = make_holdout_masks(
                        support_y=sy, ways=ways, shots=shots, val_per_class=agent_cfg.val_per_class, seed=seed + ep
                    )

                    # holdout proto
                    protos_tr = compute_prototypes(s_emb_f[train_mask], sy[train_mask], ways)
                    logits_val_proto = logits_from_prototypes(s_emb_f[val_mask], protos_tr)
                    hold_proto = acc_from_logits(logits_val_proto, sy[val_mask])

                    # holdout ridge
                    W_tr = ridge_fit(s_emb_f[train_mask], sy[train_mask], ways, ridge_lambda=ridge_lambda)
                    logits_val_ridge = s_emb_f[val_mask] @ W_tr
                    hold_ridge = acc_from_logits(logits_val_ridge, sy[val_mask])

                    # choose ridge if better
                    if (hold_ridge > hold_proto + agent_cfg.cv_margin) or (
                        abs(hold_ridge - hold_proto) <= agent_cfg.close_band and random.random() < agent_cfg.explore_prob
                    ):
                        chosen = "ridge"
                        logits_ridge = q_emb_f @ W_tr
                        logits_chosen = logits_ridge

                # store decision diagnostics
                decision_hold_proto = float(hold_proto) if np.isfinite(hold_proto) else np.nan
                decision_hold_ridge = float(hold_ridge) if np.isfinite(hold_ridge) else np.nan
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # ---- sentinel checks (only used for agent_sentinel or if explicitly enabled)
        if enable_sentinel:
            # Use chosen logits vs proto as reference
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

            # rollback rule: if ridge chosen and alerts exist -> fallback to proto
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

        acc_final = acc_from_logits(logits_final, episode.query_y.to(device))
        acc_proto = acc_from_logits(logits_proto, episode.query_y.to(device))

        row = {
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
        }

        if strategy_name in ("agent", "agent_sentinel"):
            # attach decision metrics only if they exist (safe)
            row["holdout_acc_proto"] = decision_hold_proto
            row["holdout_acc_ridge"] = decision_hold_ridge

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Reporting (summary table + plots)
# =============================================================================

def summarize_runs(all_df: pd.DataFrame) -> pd.DataFrame:
    # group by (shots, fault_sigma, fault_drop_prob, strategy)
    grp_cols = ["shots", "fault_sigma", "fault_drop_prob", "strategy"]
    g = all_df.groupby(grp_cols, dropna=False)

    out = g.agg(
        episodes=("episode", "count"),
        mean_acc=("acc_final", "mean"),
        p50_acc=("acc_final", "median"),
        mean_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda s: float(np.quantile(s.values, 0.95))),
        rollback_rate=("rollback_used", "mean"),
        mean_alerts=("alerts_count", "mean"),
        ridge_chosen_rate=("chosen_method", lambda s: float(np.mean(s.values == "ridge"))),
    ).reset_index()

    # clean rounding for readability
    for c in ["mean_acc", "p50_acc", "mean_latency_ms", "p95_latency_ms", "rollback_rate", "mean_alerts", "ridge_chosen_rate"]:
        out[c] = out[c].astype(float).round(4 if "acc" in c else 4)

    return out.sort_values(["shots", "fault_sigma", "fault_drop_prob", "strategy"]).reset_index(drop=True)


def save_plots(summary_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)

    # Plot 1: mean accuracy vs fault_sigma for each strategy (per shots)
    for shots in sorted(summary_df["shots"].unique()):
        sdf = summary_df[summary_df["shots"] == shots].copy()
        plt.figure()
        for strat in sorted(sdf["strategy"].unique()):
            ss = sdf[sdf["strategy"] == strat].sort_values("fault_sigma")
            plt.plot(ss["fault_sigma"].values, ss["mean_acc"].values, marker="o", label=strat)
        plt.xlabel("Fault sigma (Gaussian noise)")
        plt.ylabel("Mean accuracy")
        plt.title(f"Mean Accuracy vs Fault Level (shots={shots})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"mean_acc_vs_fault_shots_{shots}.png", dpi=160)
        plt.close()

    # Plot 2: mean latency vs strategy (per shots, no fault)
    for shots in sorted(summary_df["shots"].unique()):
        sdf = summary_df[(summary_df["shots"] == shots) & (summary_df["fault_sigma"] == 0.0) & (summary_df["fault_drop_prob"] == 0.0)].copy()
        if len(sdf) == 0:
            continue
        sdf = sdf.sort_values("strategy")
        plt.figure()
        plt.bar(sdf["strategy"].values, sdf["mean_latency_ms"].values)
        plt.xlabel("Strategy")
        plt.ylabel("Mean latency (ms)")
        plt.title(f"Latency Comparison (no fault, shots={shots})")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"latency_compare_shots_{shots}.png", dpi=160)
        plt.close()

    # Plot 3: rollback rate vs fault_sigma (agent_sentinel only)
    for shots in sorted(summary_df["shots"].unique()):
        sdf = summary_df[(summary_df["shots"] == shots) & (summary_df["strategy"] == "agent_sentinel")].sort_values("fault_sigma")
        if len(sdf) == 0:
            continue
        plt.figure()
        plt.plot(sdf["fault_sigma"].values, sdf["rollback_rate"].values, marker="o")
        plt.xlabel("Fault sigma")
        plt.ylabel("Rollback rate")
        plt.title(f"Rollback Rate vs Fault Level (agent_sentinel, shots={shots})")
        plt.tight_layout()
        plt.savefig(out_dir / f"rollback_vs_fault_agent_sentinel_shots_{shots}.png", dpi=160)
        plt.close()


# =============================================================================
# Main suite
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_omniglot")
    ap.add_argument("--ckpt_path", type=str, default="runs/m1_omniglot_proto/backbone.pt")
    ap.add_argument("--out_dir", type=str, default="./final_runs")

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ways", type=int, default=5)
    ap.add_argument("--shots_list", type=str, default="1,5")
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=200)

    ap.add_argument("--ridge_lambda", type=float, default=0.1)

    # faults: comma-separated sigma list (0 means no fault)
    ap.add_argument("--fault_sigmas", type=str, default="0,0.05,0.10")
    ap.add_argument("--fault_drop_prob", type=float, default=0.0)

    # strategies: proto, ridge, agent, agent_sentinel
    ap.add_argument("--strategies", type=str, default="proto,ridge,agent,agent_sentinel")

    # agent config (fast holdout)
    ap.add_argument("--val_per_class", type=int, default=1)
    ap.add_argument("--cv_margin", type=float, default=0.02)
    ap.add_argument("--explore_prob", type=float, default=0.10)
    ap.add_argument("--close_band", type=float, default=0.01)
    ap.add_argument("--proto_conf_high", type=float, default=0.92)

    # sentinel config
    ap.add_argument("--max_latency_ms", type=float, default=900.0)
    ap.add_argument("--max_entropy", type=float, default=1.35)
    ap.add_argument("--min_conf", type=float, default=0.35)
    ap.add_argument("--max_js", type=float, default=0.25)

    ap.add_argument("--img_size", type=int, default=84)
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    # output run folder
    run_dir = Path(args.out_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    ensure_dir(run_dir)

    # dataset
    tfms = [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
    if args.invert:
        tfms.append(transforms.Lambda(lambda x: 1.0 - x))
    transform = transforms.Compose(tfms)

    print("Loading Omniglot (evaluation split only)...")
    ev = datasets.Omniglot(root=args.data_root, background=False, download=True, transform=transform)
    class_to_indices = build_class_to_indices(ev)
    print(f"Evaluation size: {len(ev)}")

    # backbone
    backbone = Conv4Backbone(embed_dim=256).to(device)
    load_backbone_ckpt(backbone, Path(args.ckpt_path), device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"Loaded backbone: {args.ckpt_path}")

    # parse lists
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

    all_frames = []
    for shots in shots_list:
        for sigma in fault_sigmas:
            for strat in strategies:
                enable_sentinel = (strat == "agent_sentinel")
                print(f"\n=== Running: strategy={strat} | shots={shots} | fault_sigma={sigma} ===")
                df = run_strategy(
                    strategy_name=strat,
                    device=device,
                    backbone=backbone,
                    dataset=ev,
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
                out_csv = run_dir / f"metrics_{strat}_shots{shots}_sigma{sigma}.csv"
                df.to_csv(out_csv, index=False)
                print(f"Saved: {out_csv}")
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

    # plots
    plots_dir = run_dir / "plots"
    save_plots(summary, plots_dir)
    print(f"Saved plots in: {plots_dir}")


if __name__ == "__main__":
    main()