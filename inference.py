from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from environment import ConstellationEnv
from models.agent import load_exported_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic inference with an exported MAPPO actor policy.")
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Path to an exported actor policy file. Defaults to outputs/policy_actor.pt.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for the maximum number of rollout steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    if args.steps is not None:
        cfg.max_steps = args.steps

    policy_path = Path(args.policy_path) if args.policy_path else Path(cfg.output_dir) / cfg.policy_export_name
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    env = ConstellationEnv(cfg)
    actor, _ = load_exported_actor(str(policy_path))

    obs, info = env.reset(seed=cfg.seed + 2)
    episode_reward = 0.0
    phase_series = [float(info["phase_error_mean"])]
    anomaly_series = [float(info["anomaly_mean"])]

    for _ in range(cfg.max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = actor(obs_t)
            actions = torch.argmax(logits, dim=-1).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(actions)
        episode_reward += float(reward)
        phase_series.append(float(info["phase_error_mean"]))
        anomaly_series.append(float(info["anomaly_mean"]))

        if terminated or truncated:
            break

    summary = {
        "policy_path": str(policy_path),
        "episode_reward": float(episode_reward),
        "mean_phase_error": float(np.mean(phase_series)),
        "mean_anomaly_score": float(np.mean(anomaly_series)),
        "steps_executed": len(phase_series) - 1,
    }

    output_path = Path(cfg.output_dir) / "inference_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Deterministic inference summary:")
    print(f"- Policy: {policy_path}")
    print(f"- Episode reward: {summary['episode_reward']:.4f}")
    print(f"- Mean phase error: {summary['mean_phase_error']:.4f}")
    print(f"- Mean anomaly score: {summary['mean_anomaly_score']:.4f}")
    print(f"- Steps executed: {summary['steps_executed']}")
    print(f"- Metrics JSON: {output_path}")


if __name__ == "__main__":
    main()