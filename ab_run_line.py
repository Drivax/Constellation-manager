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

import copy
import json
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
def run_arm(spec: dict, base_cfg: ConfigLine) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {spec['label']}")
    print(f"{'=' * 60}")

    # Build config for this arm
    output_dir = f"outputs/{spec['output_subdir']}"
    checkpoint_dir = f"outputs/{spec['output_subdir']}/checkpoints"

    cfg = replace(
        base_cfg,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        metrics_json_name="line_training_metrics.json",
        metrics_csv_name="line_training_metrics.csv",
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
        "label": spec["label"],
        "rollout_horizon": cfg.rollout_horizon,
        "train_iterations": cfg.train_iterations,
        "total_env_steps": total_env_steps,
        "final_mean_reward_train": history["mean_reward"][-1],
        "final_spacing_error_train": history["phase_error"][-1],
        **eval_stats,
        "output_dir": output_dir,
    }
    return result


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
    base_cfg = ConfigLine()  # canonical defaults

    results: list[dict] = []
    for spec in ARM_SPECS:
        result = run_arm(spec, base_cfg)
        results.append(result)

    # Load baseline eval if available
    baseline: dict | None = None
    if BASELINE_EVAL_PATH.exists():
        try:
            raw = json.loads(BASELINE_EVAL_PATH.read_text(encoding="utf-8"))
            # Augment with total-steps annotation
            raw["total_env_steps"] = (
                base_cfg.num_satellites * 64 * 100  # horizon=64, 100 iters
            )
            baseline = raw
        except Exception as exc:
            print(f"[warn] Could not load baseline metrics: {exc}")

    print_summary(results, baseline)

    # Persist summary
    summary_path = Path("outputs/ab_summary.json")
    summary_payload = {
        "baseline": {"label": BASELINE_LABEL, **(baseline or {})},
        "arms": results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
