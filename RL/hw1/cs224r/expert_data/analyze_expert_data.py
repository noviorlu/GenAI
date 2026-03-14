#!/usr/bin/env python3
"""Analyze CS224R expert data pickle files.

Usage examples:
  python cs224r/expert_data/analyze_expert_data.py
  python cs224r/expert_data/analyze_expert_data.py --details
  python cs224r/expert_data/analyze_expert_data.py --plot
  python cs224r/expert_data/analyze_expert_data.py cs224r/expert_data/expert_data_Ant-v4.pkl --plot
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; change to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

REQUIRED_KEYS = [
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminal",
    "image_obs",
]


def _safe_shape(x: Any) -> Tuple[int, ...] | None:
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return None


def _is_path_dict(x: Any) -> bool:
    return isinstance(x, dict) and all(k in x for k in REQUIRED_KEYS)


def _summarize_paths(paths: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traj_count = len(paths)
    if traj_count == 0:
        return {
            "num_trajectories": 0,
            "timesteps_total": 0,
            "traj_len_min": 0,
            "traj_len_mean": 0.0,
            "traj_len_max": 0,
            "return_mean": 0.0,
            "return_std": 0.0,
            "return_min": 0.0,
            "return_max": 0.0,
            "obs_shape_example": None,
            "act_shape_example": None,
            "consistent_obs_dim": True,
            "consistent_act_dim": True,
        }

    traj_lens: List[int] = []
    traj_returns: List[float] = []
    obs_shapes: List[Tuple[int, ...] | None] = []
    act_shapes: List[Tuple[int, ...] | None] = []

    for path in paths:
        rewards = np.asarray(path["reward"])
        traj_lens.append(int(len(rewards)))
        traj_returns.append(float(np.sum(rewards)))
        obs_shapes.append(_safe_shape(path["observation"]))
        act_shapes.append(_safe_shape(path["action"]))

    obs_shape_ref = obs_shapes[0]
    act_shape_ref = act_shapes[0]

    return {
        "num_trajectories": traj_count,
        "timesteps_total": int(sum(traj_lens)),
        "traj_len_min": int(min(traj_lens)),
        "traj_len_mean": float(mean(traj_lens)),
        "traj_len_max": int(max(traj_lens)),
        "return_mean": float(mean(traj_returns)),
        "return_std": float(pstdev(traj_returns) if len(traj_returns) > 1 else 0.0),
        "return_min": float(min(traj_returns)),
        "return_max": float(max(traj_returns)),
        "obs_shape_example": obs_shape_ref,
        "act_shape_example": act_shape_ref,
        "consistent_obs_dim": all(s == obs_shape_ref for s in obs_shapes),
        "consistent_act_dim": all(s == act_shape_ref for s in act_shapes),
    }


def analyze_object(data: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "top_level_type": type(data).__name__,
    }

    if isinstance(data, list):
        info["top_level_len"] = len(data)
        if len(data) > 0:
            info["first_item_type"] = type(data[0]).__name__

        if all(_is_path_dict(item) for item in data):
            info["format"] = "list_of_trajectory_dicts"
            info.update(_summarize_paths(data))
            info["keys"] = list(data[0].keys()) if len(data) > 0 else []
        else:
            info["format"] = "generic_list"

    elif isinstance(data, tuple):
        info["top_level_len"] = len(data)
        info["tuple_item_types"] = [type(x).__name__ for x in data]
        info["format"] = "tuple"

        # Helpful fallback for tuple(list_of_paths, envsteps)
        if len(data) >= 1 and isinstance(data[0], list) and all(_is_path_dict(item) for item in data[0]):
            info["tuple_first_item_format"] = "list_of_trajectory_dicts"
            info.update(_summarize_paths(data[0]))
            info["keys"] = list(data[0][0].keys()) if len(data[0]) > 0 else []

    elif isinstance(data, dict):
        info["format"] = "dict"
        info["keys"] = list(data.keys())

    else:
        info["format"] = "other"

    return info


def format_row(cols: Sequence[str], widths: Sequence[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "file",
        "format",
        "n_traj",
        "steps_total",
        "len_mean",
        "ret_mean",
        "ret_std",
        "obs_shape",
        "act_shape",
    ]

    table: List[List[str]] = []
    for r in rows:
        table.append([
            os.path.basename(r["file"]),
            str(r.get("format", "")),
            str(r.get("num_trajectories", "")),
            str(r.get("timesteps_total", "")),
            f"{r.get('traj_len_mean', 0.0):.2f}" if "traj_len_mean" in r else "",
            f"{r.get('return_mean', 0.0):.2f}" if "return_mean" in r else "",
            f"{r.get('return_std', 0.0):.2f}" if "return_std" in r else "",
            str(r.get("obs_shape_example", "")),
            str(r.get("act_shape_example", "")),
        ])

    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]

    print(format_row(headers, widths))
    print("-+-".join("-" * w for w in widths))
    for row in table:
        print(format_row(row, widths))


def resolve_files(args_files: Sequence[str], default_dir: str) -> List[str]:
    if args_files:
        return list(args_files)
    pattern = os.path.join(default_dir, "expert_data_*.pkl")
    return sorted(glob.glob(pattern))


def plot_all(results: List[Dict[str, Any]], raw_paths: Dict[str, List[Dict[str, Any]]], out_dir: str) -> None:
    """Generate a multi-panel figure per dataset and a cross-env comparison figure."""
    os.makedirs(out_dir, exist_ok=True)
    envs = [os.path.basename(r["file"]).replace("expert_data_", "").replace(".pkl", "") for r in results]

    # ── 1. Cross-environment comparison bar chart ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Expert Data: Cross-Environment Comparison", fontsize=13, fontweight="bold")

    ret_means  = [r.get("return_mean", 0)  for r in results]
    ret_stds   = [r.get("return_std", 0)   for r in results]
    obs_dims   = [r["obs_shape_example"][1] if r.get("obs_shape_example") else 0 for r in results]
    act_dims   = [r["act_shape_example"][1] if r.get("act_shape_example") else 0 for r in results]
    steps      = [r.get("timesteps_total", 0) for r in results]
    colors     = plt.cm.Set2(np.linspace(0, 1, len(envs)))

    ax = axes[0]
    bars = ax.bar(envs, ret_means, yerr=ret_stds, color=colors, capsize=5, edgecolor="k", linewidth=0.7)
    ax.set_title("Expert Return (mean ± std)")
    ax.set_ylabel("Episode Return")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=15, ha="right")
    for bar, v in zip(bars, ret_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ret_stds) * 0.05,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    x = np.arange(len(envs))
    w = 0.35
    ax.bar(x - w/2, obs_dims, width=w, label="obs dim", color=colors, edgecolor="k", linewidth=0.7)
    ax.bar(x + w/2, act_dims, width=w, label="act dim", color=colors, alpha=0.5, edgecolor="k", linewidth=0.7)
    ax.set_title("Observation & Action Dimensions")
    ax.set_ylabel("Dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.legend()
    for i, (od, ad) in enumerate(zip(obs_dims, act_dims)):
        ax.text(i - w/2, od + 0.5, str(od), ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, ad + 0.5, str(ad), ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    ax.bar(envs, steps, color=colors, edgecolor="k", linewidth=0.7)
    ax.set_title("Total Timesteps in Dataset")
    ax.set_ylabel("Timesteps")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=15, ha="right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── 2. Per-environment detailed panel ───────────────────────────────────
    for env_name, result in zip(envs, results):
        paths = raw_paths.get(result["file"], [])
        if not paths:
            continue

        n_traj = len(paths)
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"{env_name}  —  Expert Data Analysis", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # (a) Reward curve per trajectory
        ax_rew = fig.add_subplot(gs[0, :2])
        for i, path in enumerate(paths):
            ax_rew.plot(path["reward"], alpha=0.8, label=f"traj {i}")
        ax_rew.set_title("Reward per Step")
        ax_rew.set_xlabel("Timestep")
        ax_rew.set_ylabel("Reward")
        ax_rew.legend(fontsize=8)

        # (b) Cumulative return curve
        ax_cum = fig.add_subplot(gs[0, 2])
        for i, path in enumerate(paths):
            ax_cum.plot(np.cumsum(path["reward"]), alpha=0.8, label=f"traj {i}")
        ax_cum.set_title("Cumulative Return")
        ax_cum.set_xlabel("Timestep")
        ax_cum.set_ylabel("Cum. Return")
        ax_cum.legend(fontsize=8)

        # (c) Observation statistics heatmap (mean across time) per traj
        ax_obs = fig.add_subplot(gs[1, :])
        obs_means = np.stack([p["observation"].mean(axis=0) for p in paths])  # (n_traj, obs_dim)
        im = ax_obs.imshow(obs_means, aspect="auto", cmap="RdBu_r")
        ax_obs.set_title(f"Observation Mean per Dimension  (shape: {obs_means.shape})")
        ax_obs.set_xlabel("Obs Dimension")
        ax_obs.set_ylabel("Trajectory")
        ax_obs.set_yticks(range(n_traj))
        ax_obs.set_yticklabels([f"traj {i}" for i in range(n_traj)])
        fig.colorbar(im, ax=ax_obs, shrink=0.6)

        # (d) Action distribution: box-plot across all steps, all trajs
        ax_act = fig.add_subplot(gs[2, :2])
        all_actions = np.concatenate([p["action"] for p in paths], axis=0)  # (total_steps, act_dim)
        act_dim = all_actions.shape[1]
        ax_act.boxplot(all_actions, tick_labels=[f"a{j}" for j in range(act_dim)], whis=1.5)
        ax_act.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax_act.set_title(f"Action Distribution  (act_dim={act_dim})")
        ax_act.set_xlabel("Action Dimension")
        ax_act.set_ylabel("Value")

        # (e) Terminal flags
        ax_term = fig.add_subplot(gs[2, 2])
        for i, path in enumerate(paths):
            terminals = path["terminal"]
            n_done = int(np.sum(terminals))
            ax_term.bar(i, n_done, label=f"traj {i}", color=colors[i % len(colors)])
        ax_term.set_title("Terminal=1 Count")
        ax_term.set_xlabel("Trajectory")
        ax_term.set_ylabel("Count")
        ax_term.set_xticks(range(n_traj))
        ax_term.set_xticklabels([f"traj {i}" for i in range(n_traj)])

        out_path = os.path.join(out_dir, f"{env_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze expert_data pickle files")
    parser.add_argument("files", nargs="*", help="Specific .pkl files to analyze")
    parser.add_argument("--details", action="store_true", help="Print per-file detailed info")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib figures")
    parser.add_argument("--plot_dir", default="expert_data_plots", help="Output directory for plots")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    files = resolve_files(args.files, this_dir)

    if not files:
        raise SystemExit("No .pkl files found to analyze.")

    results: List[Dict[str, Any]] = []
    raw_paths: Dict[str, List[Any]] = {}
    for file_path in files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        info = analyze_object(data)
        info["file"] = file_path
        results.append(info)
        # store the raw list-of-paths for plotting
        if isinstance(data, list):
            raw_paths[file_path] = data
        elif isinstance(data, tuple) and isinstance(data[0], list):
            raw_paths[file_path] = data[0]

    print("\n=== Expert Data Summary ===")
    print_summary_table(results)

    # Also print a pandas DataFrame summary
    df = pd.DataFrame([
        {
            "env": os.path.basename(r["file"]).replace("expert_data_", "").replace(".pkl", ""),
            "n_traj": r.get("num_trajectories"),
            "steps_total": r.get("timesteps_total"),
            "traj_len": r.get("traj_len_mean"),
            "return_mean": round(r.get("return_mean", 0), 2),
            "return_std": round(r.get("return_std", 0), 2),
            "return_min": round(r.get("return_min", 0), 2),
            "return_max": round(r.get("return_max", 0), 2),
            "obs_dim": r["obs_shape_example"][1] if r.get("obs_shape_example") else None,
            "act_dim": r["act_shape_example"][1] if r.get("act_shape_example") else None,
        }
        for r in results
    ])
    print("\n=== Pandas DataFrame ===")
    print(df.to_string(index=False))

    if args.details:
        print("\n=== Detailed Checks ===")
        for r in results:
            print(f"\n[{r['file']}]")
            for k in [
                "top_level_type", "top_level_len", "first_item_type",
                "tuple_item_types", "keys",
                "traj_len_min", "traj_len_mean", "traj_len_max",
                "return_min", "return_mean", "return_max",
                "consistent_obs_dim", "consistent_act_dim",
            ]:
                if k in r:
                    print(f"  {k}: {r[k]}")

    if args.plot:
        out_dir = os.path.join(this_dir, args.plot_dir)
        print(f"\n=== Generating plots → {out_dir} ===")
        plot_all(results, raw_paths, out_dir)
        print("Done.")


if __name__ == "__main__":
    main()
