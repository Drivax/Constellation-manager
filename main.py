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
        "--num-satellites",
        type=int,
        default=None,
        help="Override the number of satellites for scale testing, for example 300 or 500.",
    )
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
    parser.add_argument(
        "--disable-fault-injection",
        action="store_true",
        help="Disable injected actuator and orbit faults for ablation runs.",
    )
    return parser.parse_args()


def evaluate_policy(env: ConstellationEnv, agent, cfg: Config) -> dict:
    obs, info = env.reset(seed=cfg.seed + 1)

    episode_reward = 0.0
    phase_series = [info["phase_error_mean"]]
    altitude_series = [info["altitude_error_mean"]]
    anomaly_series = [info["anomaly_mean"]]
    collision_series = [info.get("collision_penalty_mean", 0.0)]
    coverage_series = [info.get("coverage_penalty", 0.0)]
    fault_series = [info.get("active_fault_fraction", 0.0)]
    anomaly_event_series = [info.get("anomaly_event_fraction", 0.0)]
    min_separation_series = [info.get("min_separation_km", 0.0)]

    for _ in range(cfg.max_steps):
        global_obs = obs.mean(axis=0).astype(np.float32)
        actions, _, _ = agent.select_action(obs, global_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions)

        episode_reward += float(reward)
        phase_series.append(info["phase_error_mean"])
        altitude_series.append(info["altitude_error_mean"])
        anomaly_series.append(info["anomaly_mean"])
        collision_series.append(info.get("collision_penalty_mean", 0.0))
        coverage_series.append(info.get("coverage_penalty", 0.0))
        fault_series.append(info.get("active_fault_fraction", 0.0))
        anomaly_event_series.append(info.get("anomaly_event_fraction", 0.0))
        min_separation_series.append(info.get("min_separation_km", 0.0))

        if terminated or truncated:
            break

    return {
        "num_satellites": int(env.num_satellites),
        "episode_reward": episode_reward,
        "phase_mean": float(np.mean(phase_series)),
        "altitude_mean": float(np.mean(altitude_series)),
        "anomaly_mean": float(np.mean(anomaly_series)),
        "collision_penalty_mean": float(np.mean(collision_series)),
        "coverage_penalty_mean": float(np.mean(coverage_series)),
        "fault_fraction_mean": float(np.mean(fault_series)),
        "anomaly_event_fraction_mean": float(np.mean(anomaly_event_series)),
        "min_separation_km_min": float(np.min(min_separation_series)),
        "phase_series": [float(value) for value in phase_series],
        "altitude_series": [float(value) for value in altitude_series],
        "anomaly_series": [float(value) for value in anomaly_series],
        "collision_series": [float(value) for value in collision_series],
        "coverage_series": [float(value) for value in coverage_series],
        "fault_series": [float(value) for value in fault_series],
        "anomaly_event_series": [float(value) for value in anomaly_event_series],
        "min_separation_series": [float(value) for value in min_separation_series],
        "trajectories": env.get_trajectories(),
        "final_positions": env.get_latest_positions(),
    }


def save_evaluation_metrics(eval_stats: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value
        for key, value in eval_stats.items()
        if key not in {"trajectories", "final_positions"}
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def apply_runtime_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.num_satellites is not None:
        cfg.num_satellites = args.num_satellites
        cfg.output_dir = f"outputs/scaling_{args.num_satellites}"
        cfg.checkpoint_dir = f"{cfg.output_dir}/checkpoints"
    if args.disable_fault_injection:
        cfg.enable_fault_injection = False


def main() -> None:
    cfg = Config()
    args = parse_args()
    apply_runtime_overrides(cfg, args)

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
    print(f"- Mean collision penalty: {eval_stats['collision_penalty_mean']:.4f}")
    print(f"- Mean coverage penalty: {eval_stats['coverage_penalty_mean']:.4f}")
    print(f"- Mean active fault fraction: {eval_stats['fault_fraction_mean']:.4f}")
    print(f"- Mean anomaly event fraction: {eval_stats['anomaly_event_fraction_mean']:.4f}")
    print(f"- Minimum separation observed (km): {eval_stats['min_separation_km_min']:.2f}")

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
