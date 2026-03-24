from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def _draw_earth(ax, radius_km: float = 6371.0) -> None:
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="#4f81bd", alpha=0.25, linewidth=0)


def plot_constellation_3d(positions: np.ndarray, output_path: str) -> None:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    _draw_earth(ax)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=12, c="#d95f02", alpha=0.9)

    max_range = float(np.max(np.linalg.norm(positions, axis=1))) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Final Constellation State")
    ax.view_init(elev=25, azim=35)

    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close(fig)


def create_trajectory_gif(
    trajectories: np.ndarray,
    output_path: str,
    max_agents_to_draw: int = 40,
    fps: int = 12,
) -> None:
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError("trajectories must have shape (T, N, 3)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    num_frames, num_agents, _ = trajectories.shape
    if num_frames < 2:
        raise ValueError("Need at least 2 frames to create a GIF")

    draw_agents = min(num_agents, max_agents_to_draw)
    indices = np.linspace(0, num_agents - 1, draw_agents, dtype=int)
    traj = trajectories[:, indices, :]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_earth(ax)

    max_range = float(np.max(np.linalg.norm(traj.reshape(-1, 3), axis=1))) * 1.15
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Constellation Trajectory")

    scatter = ax.scatter([], [], [], s=14, c="#1b9e77", alpha=0.95)

    def _update(frame: int):
        xyz = traj[frame]
        scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.view_init(elev=20 + 8 * np.sin(frame / 12), azim=frame * 2)
        ax.set_title(f"Constellation Trajectory - step {frame + 1}/{num_frames}")
        return (scatter,)

    ani = animation.FuncAnimation(fig, _update, frames=num_frames, interval=1000 // fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(output, writer=writer)
    plt.close(fig)


def plot_training_metrics(metrics_json_path: str, output_path: str) -> None:
    metrics_file = Path(metrics_json_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    history = payload.get("history", {})

    iterations = np.arange(1, len(history.get("mean_reward", [])) + 1)
    if iterations.size == 0:
        raise ValueError("No training history found in metrics JSON")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(iterations, history["mean_reward"], color="#1b9e77", linewidth=2)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Reward History")
    axes[0].grid(alpha=0.25)

    axes[1].plot(iterations, history["phase_error"], color="#d95f02", linewidth=2)
    axes[1].set_ylabel("Phase Error")
    axes[1].set_title("Training Phase Error History")
    axes[1].grid(alpha=0.25)

    axes[2].plot(iterations, history["anomaly"], color="#7570b3", linewidth=2)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Anomaly")
    axes[2].set_title("Training Anomaly History")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close(fig)
