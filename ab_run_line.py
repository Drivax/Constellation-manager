"""
ab_run_line.py — A/B rollout-horizon experiment
================================================
Isolates whether the gain from rollout_horizon=64 came from the longer
horizon itself or simply from seeing more environment steps.

Three conditions compared:

  Arm A  (horizon-32, 100 iters)  – identical iteration budget to current,
                                     half the total env steps.
  Arm B  (horizon-32, 200 iters)  – step-matched to current (32×200 == 64×100).
  Baseline (horizon-64, 100 iters) – current best, read from outputs/step2
                                     if already present; otherwise skipped.

Results land in:
  outputs/ab_horizon32_iter100/
  outputs/ab_horizon32_iter200/

A summary table is printed at the end and written to
  outputs/ab_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from config_line import ConfigLine
from environment_line import StraightLineEnv
from train import train_mappo
from utils.visualization import plot_training_metrics


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
ARM_SPECS: list[dict] = [
    {
        "name": "horizon32_iter100",
        "label": "Arm A  (horizon=32, 100 iters — old budget)",
        "overrides": {"rollout_horizon": 32, "train_iterations": 100},
        "output_subdir": "ab_horizon32_iter100",
    },
    {
        "name": "horizon32_iter200",
        "label": "Arm B  (horizon=32, 200 iters — step-matched)",
        "overrides": {"rollout_horizon": 32, "train_iterations": 200},
        "output_subdir": "ab_horizon32_iter200",
    },
]

BASELINE_EVAL_PATH = Path("outputs/step2/line_evaluation_metrics.json")
BASELINE_LABEL = "Baseline (horizon=64, 100 iters — current)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rollout-horizon A/B experiments for the straight-line constellation "
            "with optional multi-seed aggregation."
        )
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds to run, e.g. '42,1337,2025'.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[spec["name"] for spec in ARM_SPECS],
        default=None,
        help="Subset of arms to run. Defaults to all defined arms.",
    )
    parser.add_argument(
        "--baseline-eval-path",
        default=str(BASELINE_EVAL_PATH),
        help="Path to baseline evaluation JSON. Use empty string to disable baseline loading.",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/ab_summary.json",
        help="Where to write the A/B summary JSON.",
    )
    return parser.parse_args()


def parse_seed_list(raw_seeds: str) -> list[int]:
    seeds: list[int] = []
    for token in raw_seeds.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid seed '{token}'. Seeds must be integers.") from exc

    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


# ---------------------------------------------------------------------------
# Evaluation helper (mirrors main_line.evaluate_policy)
# ---------------------------------------------------------------------------
def evaluate_policy(env: StraightLineEnv, agent, cfg: ConfigLine) -> dict:
    obs, info = env.reset(seed=cfg.seed + 1)

    episode_reward = 0.0
    spacing_series = [info["spacing_error_mean"]]
    straightness_series = [info["straightness_score"]]

    for _ in range(cfg.max_steps):
        global_obs = obs.mean(axis=0).astype(np.float32)
        actions, _, _ = agent.select_action(obs, global_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions)

        episode_reward += float(reward)
        spacing_series.append(info["spacing_error_mean"])
        straightness_series.append(info["straightness_score"])

        if terminated or truncated:
            break

    return {
        "episode_reward": episode_reward,
        "spacing_error_mean": float(np.mean(spacing_series)),
        "spacing_error_final": spacing_series[-1],
        "straightness_mean": float(np.mean(straightness_series)),
        "straightness_final": straightness_series[-1],
    }


# ---------------------------------------------------------------------------
# Run one arm
# ---------------------------------------------------------------------------
def run_arm(spec: dict, base_cfg: ConfigLine, seed: int, use_seed_subdir: bool) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {spec['label']}")
    print(f"{'=' * 60}")

    # Build config for this arm
    output_dir = f"outputs/{spec['output_subdir']}"
    if use_seed_subdir:
        output_dir = f"{output_dir}/seed{seed}"
    checkpoint_dir = f"{output_dir}/checkpoints"

    cfg = replace(
        base_cfg,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        metrics_json_name="line_training_metrics.json",
        metrics_csv_name="line_training_metrics.csv",
        seed=seed,
        evaluation_json_name="line_evaluation_metrics.json",
        latest_checkpoint_name="line_mappo_latest.pt",
        best_checkpoint_name="line_mappo_best.pt",
        policy_export_name="line_policy_actor.pt",
        **spec["overrides"],
    )

    total_env_steps = cfg.num_satellites * cfg.rollout_horizon * cfg.train_iterations
    print(f"  rollout_horizon={cfg.rollout_horizon}  train_iterations={cfg.train_iterations}")
    print(f"  total env steps (all agents): {total_env_steps:,}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = StraightLineEnv(cfg)
    agent, history, artifact_paths = train_mappo(env, cfg)

    # Plot training curves
    metrics_plot = Path(output_dir) / "line_training_metrics.png"
    plot_training_metrics(artifact_paths["metrics_json"], str(metrics_plot))

    print("\n  Running deterministic evaluation...")
    eval_stats = evaluate_policy(env, agent, cfg)

    # Persist eval metrics
    eval_path = Path(output_dir) / cfg.evaluation_json_name
    eval_path.write_text(json.dumps(eval_stats, indent=2), encoding="utf-8")

    result = {
        "arm_name": spec["name"],
        "label": spec["label"],
        "seed": seed,
        "rollout_horizon": cfg.rollout_horizon,
        "train_iterations": cfg.train_iterations,
        "total_env_steps": total_env_steps,
        "final_mean_reward_train": history["mean_reward"][-1],
        "final_spacing_error_train": history["phase_error"][-1],
        **eval_stats,
        "output_dir": output_dir,
    }
    return result


def aggregate_by_arm_seed(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["arm_name"]].append(row)

    metric_keys = [
        "episode_reward",
        "spacing_error_mean",
        "spacing_error_final",
        "straightness_mean",
        "straightness_final",
        "final_mean_reward_train",
        "final_spacing_error_train",
    ]

    aggregated: list[dict] = []
    for arm_name, rows in grouped.items():
        sample = rows[0]
        agg_row = {
            "arm_name": arm_name,
            "label": f"{sample['label']} [mean over {len(rows)} seeds]",
            "rollout_horizon": sample["rollout_horizon"],
            "train_iterations": sample["train_iterations"],
            "total_env_steps": sample["total_env_steps"],
            "n_seeds": len(rows),
        }
        for key in metric_keys:
            values = [float(r[key]) for r in rows if key in r and r[key] is not None]
            if not values:
                continue
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=0))
            agg_row[key] = mean_val
            agg_row[f"{key}_std"] = std_val
        aggregated.append(agg_row)

    return sorted(aggregated, key=lambda x: x["arm_name"])


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], baseline: dict | None) -> None:
    rows = []
    if baseline is not None:
        rows.append({"label": BASELINE_LABEL, **baseline})
    rows.extend(results)

    metrics = [
        ("episode_reward",        "Eval episode reward"),
        ("spacing_error_mean",    "Eval spacing err (mean)"),
        ("spacing_error_final",   "Eval spacing err (final)"),
        ("straightness_mean",     "Eval straightness (mean)"),
        ("straightness_final",    "Eval straightness (final)"),
        ("final_mean_reward_train", "Train final mean reward"),
        ("final_spacing_error_train", "Train final spacing err"),
    ]

    col_w = max(len(r["label"]) for r in rows) + 2

    print(f"\n{'=' * (col_w + 12 * len(metrics))}")
    print("  A/B ROLLOUT-HORIZON SUMMARY")
    print(f"{'=' * (col_w + 12 * len(metrics))}")

    # Header
    header = f"{'Condition':<{col_w}}"
    for _, mname in metrics:
        header += f"  {mname[:10]:>10}"
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['label']:<{col_w}}"
        for mkey, _ in metrics:
            val = row.get(mkey)
            if val is None:
                line += f"  {'N/A':>10}"
            else:
                line += f"  {val:>10.4f}"
        print(line)

    print()
    print("Interpretation guide:")
    print("  • If Arm A ≈ Baseline  → gain is NOT from horizon; you just needed more iters")
    print("  • If Arm B ≈ Baseline  → gain IS from total env steps, not horizon structure")
    print("  • If Arm B < Baseline  → longer horizon itself provides a structural benefit")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    base_cfg = ConfigLine()  # canonical defaults

    try:
        seeds = parse_seed_list(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    selected_arm_specs = ARM_SPECS
    if args.arms:
        selected_names = set(args.arms)
        selected_arm_specs = [spec for spec in ARM_SPECS if spec["name"] in selected_names]

    multi_seed = len(seeds) > 1
    print(f"Running A/B experiment with seeds: {seeds}")
    if multi_seed:
        print("Multi-seed mode enabled: per-seed outputs are stored in seed-specific subfolders.")

    results: list[dict] = []
    for spec in selected_arm_specs:
        for seed in seeds:
            result = run_arm(spec, base_cfg, seed=seed, use_seed_subdir=multi_seed)
            results.append(result)

    summary_rows = aggregate_by_arm_seed(results) if multi_seed else results

    # Load baseline eval if available
    baseline: dict | None = None
    baseline_eval_path = Path(args.baseline_eval_path) if args.baseline_eval_path else None
    if baseline_eval_path and baseline_eval_path.exists():
        try:
            raw = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
            # Augment with total-steps annotation
            raw["total_env_steps"] = (
                base_cfg.num_satellites * 64 * 100  # horizon=64, 100 iters
            )
            baseline = raw
        except Exception as exc:
            print(f"[warn] Could not load baseline metrics: {exc}")
    elif baseline_eval_path:
        print(f"[warn] Baseline file not found at: {baseline_eval_path}")

    print_summary(summary_rows, baseline)

    if multi_seed:
        print("Per-seed run overview:")
        for row in sorted(results, key=lambda x: (x["arm_name"], x["seed"])):
            print(
                "  "
                f"{row['arm_name']} seed={row['seed']}: "
                f"eval_reward={row['episode_reward']:.4f}, "
                f"spacing_mean={row['spacing_error_mean']:.4f}, "
                f"straightness_mean={row['straightness_mean']:.4f}"
            )

    # Persist summary
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "seeds": seeds,
        "multi_seed": multi_seed,
        "baseline": {"label": BASELINE_LABEL, **(baseline or {})},
        "arms": summary_rows,
        "arm_runs": results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
