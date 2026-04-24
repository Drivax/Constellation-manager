from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from config import Config
from environment import ConstellationEnv
from models.agent import MAPPOAgent, MAPPOHyperParams


def save_checkpoint(
    agent: MAPPOAgent,
    cfg: Config,
    checkpoint_path: str | Path,
    history: Dict[str, list],
    iteration: int,
    best_reward: float,
) -> Path:
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    normalizer_state: Dict[str, object] = {}
    if hasattr(agent, "reward_rms"):
        normalizer_state["reward_rms"] = agent.reward_rms.state_dict()
    if hasattr(agent, "return_rms"):
        normalizer_state["return_rms"] = agent.return_rms.state_dict()

    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "normalizer_state": normalizer_state,
            "history": history,
            "best_reward": best_reward,
            "config": cfg.__dict__,
        },
        checkpoint_file,
    )
    return checkpoint_file


def load_checkpoint(
    agent: MAPPOAgent,
    checkpoint_path: str | Path,
) -> Dict[str, object]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=agent.device)
    agent.model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        agent.optimizer.load_state_dict(optimizer_state)

    normalizer_state = checkpoint.get("normalizer_state", {})
    if hasattr(agent, "reward_rms") and "reward_rms" in normalizer_state:
        agent.reward_rms.load_state_dict(normalizer_state["reward_rms"])
    if hasattr(agent, "return_rms") and "return_rms" in normalizer_state:
        agent.return_rms.load_state_dict(normalizer_state["return_rms"])

    return checkpoint


def export_policy(
    agent: MAPPOAgent,
    policy_path: str | Path,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> Path:
    policy_file = Path(policy_path)
    policy_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": agent.model.actor.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
        },
        policy_file,
    )
    return policy_file


def prune_interval_checkpoints(checkpoint_dir: str | Path, max_keep: int) -> list[str]:
    if max_keep < 1:
        max_keep = 1

    checkpoint_path = Path(checkpoint_dir)
    interval_files = sorted(checkpoint_path.glob("mappo_iter_*.pt"))
    if len(interval_files) <= max_keep:
        return []

    removed = []
    for old_file in interval_files[:-max_keep]:
        old_file.unlink(missing_ok=True)
        removed.append(str(old_file))
    return removed


def export_metrics(
    history: Dict[str, list],
    json_path: str | Path,
    csv_path: str | Path,
) -> Tuple[Path, Path]:
    json_file = Path(json_path)
    csv_file = Path(csv_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    num_rows = len(history.get("mean_reward", []))
    for idx in range(num_rows):
        row = {"iteration": idx + 1}
        for key, values in history.items():
            row[key] = float(values[idx])
        rows.append(row)

    with json_file.open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "rows": rows}, handle, indent=2)

    if rows:
        with csv_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return json_file, csv_file


def train_mappo(env: ConstellationEnv, cfg: Config) -> Tuple[MAPPOAgent, Dict[str, list], Dict[str, str]]:
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = obs.shape[1]
    global_obs_dim = obs_dim

    hparams = MAPPOHyperParams(
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_eps=cfg.clip_eps,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate_start=float(getattr(cfg, "learning_rate_start", getattr(cfg, "learning_rate", 3e-4))),
        learning_rate_end=float(getattr(cfg, "learning_rate_end", 1e-4)),
        adam_eps=float(getattr(cfg, "adam_eps", 1e-5)),
        value_clip_eps=float(getattr(cfg, "value_clip_eps", cfg.clip_eps)),
        target_kl=float(getattr(cfg, "target_kl", 0.02)),
        normalize_advantages=bool(getattr(cfg, "normalize_advantages", True)),
        normalize_rewards=bool(getattr(cfg, "normalize_rewards", True)),
        normalize_returns=bool(getattr(cfg, "normalize_returns", True)),
    )

    agent = MAPPOAgent(
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        action_dim=cfg.action_dim,
        actor_hidden_dim=cfg.actor_hidden_dim,
        critic_hidden_dim=int(getattr(cfg, "critic_hidden_dim", cfg.actor_hidden_dim)),
        hparams=hparams,
    )

    history: Dict[str, list] = {
        "mean_reward": [],
        "phase_error": [],
        "altitude_error": [],
        "anomaly": [],
        "collision_penalty": [],
        "coverage_penalty": [],
        "fault_fraction": [],
        "anomaly_event_fraction": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clipfrac": [],
        "explained_var": [],
        "learning_rate": [],
    }
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: Dict[str, str] = {
        "latest_checkpoint": str(checkpoint_dir / cfg.latest_checkpoint_name),
        "best_checkpoint": str(checkpoint_dir / cfg.best_checkpoint_name),
        "policy_export": str(Path(cfg.output_dir) / cfg.policy_export_name),
    }
    best_reward = float("-inf")
    start_iteration = 0

    resume_path = None
    if cfg.resume_checkpoint_path:
        resume_path = Path(cfg.resume_checkpoint_path)
    elif cfg.resume_mode == "latest":
        resume_path = checkpoint_dir / cfg.latest_checkpoint_name
    elif cfg.resume_mode == "best":
        resume_path = checkpoint_dir / cfg.best_checkpoint_name

    if resume_path is not None and resume_path.exists():
        checkpoint = load_checkpoint(agent, resume_path)
        checkpoint_history = checkpoint.get("history")
        if isinstance(checkpoint_history, dict):
            history = {
                key: list(values)
                for key, values in checkpoint_history.items()
            }
        start_iteration = int(checkpoint.get("iteration", 0))
        best_reward = float(checkpoint.get("best_reward", float("-inf")))
        artifact_paths["resumed_from"] = str(resume_path)
        print(f"Resuming training from checkpoint: {resume_path}")

    for iteration in range(start_iteration, cfg.train_iterations):
        progress = iteration / max(1, cfg.train_iterations - 1)
        learning_rate = (
            hparams.learning_rate_start
            + (hparams.learning_rate_end - hparams.learning_rate_start) * progress
        )
        agent.set_learning_rate(learning_rate)

        rollout = {
            "obs": [],
            "global_obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

        reward_acc = 0.0
        phase_acc = 0.0
        altitude_acc = 0.0
        anomaly_acc = 0.0
        collision_acc = 0.0
        coverage_acc = 0.0
        fault_acc = 0.0
        anomaly_event_acc = 0.0

        for _ in range(cfg.rollout_horizon):
            global_obs = obs.mean(axis=0).astype(np.float32)
            actions, log_probs, value = agent.select_action(obs, global_obs)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            per_agent_reward = info.get(
                "per_agent_reward",
                np.full(env.num_satellites, reward, dtype=np.float32),
            )

            rollout["obs"].append(obs.astype(np.float32))
            rollout["global_obs"].append(global_obs)
            rollout["actions"].append(actions.astype(np.int64))
            rollout["log_probs"].append(log_probs.astype(np.float32))
            rollout["rewards"].append(per_agent_reward.astype(np.float32))
            rollout["dones"].append(float(done))
            rollout["values"].append(float(value))

            reward_acc += float(reward)
            phase_acc += float(info["phase_error_mean"])
            altitude_acc += float(info["altitude_error_mean"])
            anomaly_acc += float(info["anomaly_mean"])
            collision_acc += float(info.get("collision_penalty_mean", 0.0))
            coverage_acc += float(info.get("coverage_penalty", 0.0))
            fault_acc += float(info.get("active_fault_fraction", 0.0))
            anomaly_event_acc += float(info.get("anomaly_event_fraction", 0.0))

            obs = next_obs
            if done:
                obs, _ = env.reset()

        next_global_obs = obs.mean(axis=0).astype(np.float32)
        _, _, next_value = agent.select_action(obs, next_global_obs, deterministic=True)

        rollout_np = {
            key: np.array(value)
            for key, value in rollout.items()
        }

        update_stats = agent.update(rollout_np, next_value)

        history["mean_reward"].append(reward_acc / cfg.rollout_horizon)
        history["phase_error"].append(phase_acc / cfg.rollout_horizon)
        history["altitude_error"].append(altitude_acc / cfg.rollout_horizon)
        history["anomaly"].append(anomaly_acc / cfg.rollout_horizon)
        history["collision_penalty"].append(collision_acc / cfg.rollout_horizon)
        history["coverage_penalty"].append(coverage_acc / cfg.rollout_horizon)
        history["fault_fraction"].append(fault_acc / cfg.rollout_horizon)
        history["anomaly_event_fraction"].append(anomaly_event_acc / cfg.rollout_horizon)
        history["actor_loss"].append(update_stats["actor_loss"])
        history["critic_loss"].append(update_stats["critic_loss"])
        history["entropy"].append(update_stats["entropy"])
        history["approx_kl"].append(update_stats["approx_kl"])
        history["clipfrac"].append(update_stats["clipfrac"])
        history["explained_var"].append(update_stats["explained_var"])
        history["learning_rate"].append(learning_rate)

        print(
            "[Iter {}/{}] reward={:.4f} phase={:.4f} alt={:.4f} anom={:.4f} "
            "actor={:.4f} critic={:.4f} ent={:.4f} kl={:.5f} clip={:.3f} ev={:.3f} lr={:.2e}".format(
                iteration + 1,
                cfg.train_iterations,
                history["mean_reward"][-1],
                history["phase_error"][-1],
                history["altitude_error"][-1],
                history["anomaly"][-1],
                history["actor_loss"][-1],
                history["critic_loss"][-1],
                history["entropy"][-1],
                history["approx_kl"][-1],
                history["clipfrac"][-1],
                history["explained_var"][-1],
                history["learning_rate"][-1],
            )
        )

        current_reward = history["mean_reward"][-1]
        if current_reward >= best_reward:
            best_reward = current_reward
            best_checkpoint = save_checkpoint(
                agent,
                cfg,
                checkpoint_dir / cfg.best_checkpoint_name,
                history,
                iteration + 1,
                best_reward,
            )
            artifact_paths["best_checkpoint"] = str(best_checkpoint)
        elif "best_checkpoint" not in artifact_paths:
            artifact_paths["best_checkpoint"] = str(checkpoint_dir / cfg.best_checkpoint_name)

        latest_checkpoint = save_checkpoint(
            agent,
            cfg,
            checkpoint_dir / cfg.latest_checkpoint_name,
            history,
            iteration + 1,
            best_reward,
        )
        artifact_paths["latest_checkpoint"] = str(latest_checkpoint)

        if (iteration + 1) % cfg.checkpoint_every == 0:
            interval_checkpoint = save_checkpoint(
                agent,
                cfg,
                checkpoint_dir / f"mappo_iter_{iteration + 1:03d}.pt",
                history,
                iteration + 1,
                best_reward,
            )
            artifact_paths[f"checkpoint_iter_{iteration + 1}"] = str(interval_checkpoint)
            removed_files = prune_interval_checkpoints(checkpoint_dir, cfg.max_interval_checkpoints)
            if removed_files:
                artifact_paths["pruned_interval_checkpoints"] = ", ".join(removed_files)

    metrics_json, metrics_csv = export_metrics(
        history,
        Path(cfg.output_dir) / cfg.metrics_json_name,
        Path(cfg.output_dir) / cfg.metrics_csv_name,
    )
    artifact_paths["metrics_json"] = str(metrics_json)
    artifact_paths["metrics_csv"] = str(metrics_csv)

    policy_file = export_policy(
        agent,
        Path(cfg.output_dir) / cfg.policy_export_name,
        obs_dim,
        cfg.action_dim,
        cfg.actor_hidden_dim,
    )
    artifact_paths["policy_export"] = str(policy_file)

    return agent, history, artifact_paths
