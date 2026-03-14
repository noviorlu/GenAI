#!/usr/bin/env python3
"""Analyze CS224R expert data pickle files.

Usage examples:
  python cs224r/expert_data/analyze_expert_data.py
  python cs224r/expert_data/analyze_expert_data.py --details
  python cs224r/expert_data/analyze_expert_data.py cs224r/expert_data/expert_data_Ant-v4.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze expert_data pickle files")
    parser.add_argument("files", nargs="*", help="Specific .pkl files to analyze")
    parser.add_argument("--details", action="store_true", help="Print per-file detailed info")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    files = resolve_files(args.files, this_dir)

    if not files:
        raise SystemExit("No .pkl files found to analyze.")

    results: List[Dict[str, Any]] = []
    for file_path in files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        info = analyze_object(data)
        info["file"] = file_path
        results.append(info)

    print("\n=== Expert Data Summary ===")
    print_summary_table(results)

    if args.details:
        print("\n=== Detailed Checks ===")
        for r in results:
            print(f"\n[{r['file']}]")
            for k in [
                "top_level_type",
                "top_level_len",
                "first_item_type",
                "tuple_item_types",
                "keys",
                "traj_len_min",
                "traj_len_mean",
                "traj_len_max",
                "return_min",
                "return_mean",
                "return_max",
                "consistent_obs_dim",
                "consistent_act_dim",
            ]:
                if k in r:
                    print(f"  {k}: {r[k]}")


if __name__ == "__main__":
    main()
