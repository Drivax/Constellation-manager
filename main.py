from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import Config
from environment import ConstellationEnv
from train import train_mappo
from utils.visualization import create_trajectory_gif, plot_constellation_3d, plot_training_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Constellation Manager MAPPO pipeline.")
    parser.add_argument(
        "--resume-mode",
        choices=["none", "latest", "best"],
        default=None,
        help="Resume training from the latest or best checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint-path",
        default=None,
        help="Resume training from an explicit checkpoint path.",
    )
    return parser.parse_args()


def evaluate_policy(env: ConstellationEnv, agent, cfg: Config) -> dict:
    obs, info = env.reset(seed=cfg.seed + 1)

    episode_reward = 0.0
    phase_series = [info["phase_error_mean"]]
    altitude_series = [info["altitude_error_mean"]]
    anomaly_series = [info["anomaly_mean"]]

    for _ in range(cfg.max_steps):
        global_obs = obs.mean(axis=0).astype(np.float32)
        actions, _, _ = agent.select_action(obs, global_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions)

        episode_reward += float(reward)
        phase_series.append(info["phase_error_mean"])
        altitude_series.append(info["altitude_error_mean"])
        anomaly_series.append(info["anomaly_mean"])

        if terminated or truncated:
            break

    return {
        "episode_reward": episode_reward,
        "phase_mean": float(np.mean(phase_series)),
        "altitude_mean": float(np.mean(altitude_series)),
        "anomaly_mean": float(np.mean(anomaly_series)),
        "phase_series": [float(value) for value in phase_series],
        "altitude_series": [float(value) for value in altitude_series],
        "anomaly_series": [float(value) for value in anomaly_series],
        "trajectories": env.get_trajectories(),
        "final_positions": env.get_latest_positions(),
    }


def save_evaluation_metrics(eval_stats: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "episode_reward": float(eval_stats["episode_reward"]),
        "phase_mean": float(eval_stats["phase_mean"]),
        "altitude_mean": float(eval_stats["altitude_mean"]),
        "anomaly_mean": float(eval_stats["anomaly_mean"]),
        "phase_series": eval_stats["phase_series"],
        "altitude_series": eval_stats["altitude_series"],
        "anomaly_series": eval_stats["anomaly_series"],
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    cfg = Config()
    args = parse_args()

    if args.resume_mode is not None:
        cfg.resume_mode = args.resume_mode
    if args.resume_checkpoint_path is not None:
        cfg.resume_checkpoint_path = args.resume_checkpoint_path

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing environment with real Starlink TLE data...")
    env = ConstellationEnv(cfg)

    print("Starting MAPPO training...")
    agent, history, artifact_paths = train_mappo(env, cfg)

    print("Running deterministic evaluation...")
    eval_stats = evaluate_policy(env, agent, cfg)

    final_plot_path = output_dir / cfg.final_plot_name
    gif_path = output_dir / cfg.gif_name
    evaluation_metrics_path = output_dir / cfg.evaluation_json_name
    metrics_plot_path = output_dir / cfg.metrics_plot_name

    plot_constellation_3d(eval_stats["final_positions"], str(final_plot_path))
    create_trajectory_gif(eval_stats["trajectories"], str(gif_path))
    save_evaluation_metrics(eval_stats, evaluation_metrics_path)
    plot_training_metrics(artifact_paths["metrics_json"], str(metrics_plot_path))

    print("\nTraining summary:")
    print(f"- Final training mean reward: {history['mean_reward'][-1]:.4f}")
    print(f"- Final training phase error: {history['phase_error'][-1]:.4f}")
    print(f"- Final training altitude error: {history['altitude_error'][-1]:.4f}")
    print(f"- Final training anomaly: {history['anomaly'][-1]:.4f}")

    print("\nEvaluation summary:")
    print(f"- Episode reward: {eval_stats['episode_reward']:.4f}")
    print(f"- Mean phase error: {eval_stats['phase_mean']:.4f}")
    print(f"- Mean altitude error: {eval_stats['altitude_mean']:.4f}")
    print(f"- Mean anomaly score: {eval_stats['anomaly_mean']:.4f}")

    print("\nArtifacts:")
    print(f"- 3D plot: {final_plot_path}")
    print(f"- GIF: {gif_path}")
    if "resumed_from" in artifact_paths:
        print(f"- Resumed from checkpoint: {artifact_paths['resumed_from']}")
    if "pruned_interval_checkpoints" in artifact_paths:
        print(f"- Pruned checkpoints: {artifact_paths['pruned_interval_checkpoints']}")
    print(f"- Latest checkpoint: {artifact_paths['latest_checkpoint']}")
    print(f"- Best checkpoint: {artifact_paths['best_checkpoint']}")
    print(f"- Inference policy: {artifact_paths['policy_export']}")
    print(f"- Training metrics JSON: {artifact_paths['metrics_json']}")
    print(f"- Training metrics CSV: {artifact_paths['metrics_csv']}")
    print(f"- Training metrics plot: {metrics_plot_path}")
    print(f"- Evaluation metrics JSON: {evaluation_metrics_path}")


if __name__ == "__main__":
    main()
