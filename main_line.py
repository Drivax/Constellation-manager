"""
main_line.py — Step 2 entry point
==================================
Trains and evaluates the MAPPO agent on the StraightLineEnv (30 satellites
in a single orbital plane, goal: maintain equal inter-satellite spacing).

Usage
-----
python main_line.py
python main_line.py --resume-mode latest
python main_line.py --resume-checkpoint-path outputs/step2/checkpoints/line_mappo_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config_line import ConfigLine
from environment_line import StraightLineEnv
from train import train_mappo
from utils.visualization import (
    create_trajectory_gif,
    plot_line_constellation_3d,
    plot_training_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate MAPPO on the straight-line constellation."
    )
    parser.add_argument(
        "--resume-mode",
        choices=["none", "latest", "best"],
        default=None,
        help="Resume from the latest or best checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint-path",
        default=None,
        help="Resume from an explicit checkpoint path.",
    )
    return parser.parse_args()


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
        "spacing_series": [float(v) for v in spacing_series],
        "straightness_series": [float(v) for v in straightness_series],
        "trajectories": env.get_trajectories(),
        "final_positions": env.get_latest_positions(),
    }


def save_evaluation_metrics(eval_stats: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in eval_stats.items() if k not in ("trajectories", "final_positions")}
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    cfg = ConfigLine()
    args = parse_args()

    if args.resume_mode is not None:
        cfg.resume_mode = args.resume_mode
    if args.resume_checkpoint_path is not None:
        cfg.resume_checkpoint_path = args.resume_checkpoint_path

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initialising straight-line environment (30 satellites, one orbital plane)...")
    env = StraightLineEnv(cfg)

    print("Starting MAPPO training — Step 2...")
    agent, history, artifact_paths = train_mappo(env, cfg)

    print("Running deterministic evaluation...")
    eval_stats = evaluate_policy(env, agent, cfg)

    # ---- Save artefacts -------------------------------------------------
    final_plot_path = output_dir / cfg.final_plot_name
    gif_path = output_dir / cfg.gif_name
    evaluation_metrics_path = output_dir / cfg.evaluation_json_name
    metrics_plot_path = output_dir / cfg.metrics_plot_name

    plot_line_constellation_3d(eval_stats["final_positions"], str(final_plot_path), title="Final Chain State")
    create_trajectory_gif(eval_stats["trajectories"], str(gif_path))
    save_evaluation_metrics(eval_stats, evaluation_metrics_path)
    plot_training_metrics(artifact_paths["metrics_json"], str(metrics_plot_path))

    # ---- Summary --------------------------------------------------------
    print("\nTraining summary:")
    print(f"  Final mean reward   : {history['mean_reward'][-1]:.4f}")
    print(f"  Final spacing error : {history['phase_error'][-1]:.4f}")

    print("\nEvaluation summary:")
    print(f"  Episode reward      : {eval_stats['episode_reward']:.4f}")
    print(f"  Mean spacing error  : {eval_stats['spacing_error_mean']:.4f}  (normalised by desired gap)")
    print(f"  Final spacing error : {eval_stats['spacing_error_final']:.4f}")
    print(f"  Mean straightness   : {eval_stats['straightness_mean']:.4f}  (1.0 = perfect line)")
    print(f"  Final straightness  : {eval_stats['straightness_final']:.4f}")

    print("\nArtefacts:")
    print(f"  3D plot             : {final_plot_path}")
    print(f"  GIF                 : {gif_path}")
    if "resumed_from" in artifact_paths:
        print(f"  Resumed from       : {artifact_paths['resumed_from']}")
    print(f"  Latest checkpoint   : {artifact_paths['latest_checkpoint']}")
    print(f"  Best checkpoint     : {artifact_paths['best_checkpoint']}")
    print(f"  Inference policy    : {artifact_paths['policy_export']}")
    print(f"  Training metrics    : {artifact_paths['metrics_json']}")
    print(f"  Training plot       : {metrics_plot_path}")
    print(f"  Evaluation metrics  : {evaluation_metrics_path}")


if __name__ == "__main__":
    main()
