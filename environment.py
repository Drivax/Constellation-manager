from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from sgp4.api import jday

from config import Config
from utils.tle_loader import build_satrecs, fetch_starlink_tles


class OrbitAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class ConstellationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        tle_records = fetch_starlink_tles(limit=cfg.num_satellites)
        self.sat_names = [rec["name"] for rec in tle_records]
        self.satrecs = build_satrecs(tle_records)
        self.num_satellites = len(self.satrecs)

        self.obs_dim = 7
        self.action_dim = cfg.action_dim

        self.action_space = spaces.MultiDiscrete([self.action_dim] * self.num_satellites)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_satellites, self.obs_dim),
            dtype=np.float32,
        )

        self.start_time = datetime.utcnow()
        self.step_count = 0

        self.desired_phases = np.linspace(0.0, 2.0 * np.pi, self.num_satellites, endpoint=False)
        self.phase_offsets = np.zeros(self.num_satellites, dtype=np.float32)
        self.fuel = np.ones(self.num_satellites, dtype=np.float32)

        self.target_radius = 0.0
        self.latest_positions = np.zeros((self.num_satellites, 3), dtype=np.float32)
        self.latest_velocities = np.zeros((self.num_satellites, 3), dtype=np.float32)
        self.latest_anomaly = np.zeros(self.num_satellites, dtype=np.float32)
        self.latest_collision_penalty = np.zeros(self.num_satellites, dtype=np.float32)
        self.latest_coverage_penalty = 0.0

        self.trajectories: List[np.ndarray] = []

        self.fault_start_steps = np.full(self.num_satellites, self.cfg.max_steps + 1, dtype=np.int32)
        self.fault_end_steps = np.zeros(self.num_satellites, dtype=np.int32)
        self.fault_phase_drifts = np.zeros(self.num_satellites, dtype=np.float32)
        self.fault_radial_offsets = np.zeros(self.num_satellites, dtype=np.float32)
        self.fault_actuation_scales = np.ones(self.num_satellites, dtype=np.float32)

        self.ae_input_dim = 6
        self.autoencoder = OrbitAutoencoder(
            input_dim=self.ae_input_dim,
            hidden_dim=cfg.ae_hidden_dim,
            latent_dim=cfg.ae_latent_dim,
        )
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=cfg.ae_lr)
        self.ae_mean = np.zeros(self.ae_input_dim, dtype=np.float32)
        self.ae_std = np.ones(self.ae_input_dim, dtype=np.float32)

        self._fit_autoencoder_on_nominal_data()

    def _propagate_at_time(self, current_time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        jd, fr = jday(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute,
            current_time.second + current_time.microsecond * 1e-6,
        )

        positions = np.zeros((self.num_satellites, 3), dtype=np.float64)
        velocities = np.zeros((self.num_satellites, 3), dtype=np.float64)

        for idx, sat in enumerate(self.satrecs):
            err, pos, vel = sat.sgp4(jd, fr)
            if err == 0:
                positions[idx] = np.array(pos, dtype=np.float64)
                velocities[idx] = np.array(vel, dtype=np.float64)
            else:
                if idx > 0:
                    positions[idx] = positions[idx - 1]
                    velocities[idx] = velocities[idx - 1]

        return positions, velocities

    def _raw_feature_matrix(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        radius = np.linalg.norm(positions, axis=1)
        speed = np.linalg.norm(velocities, axis=1)
        incl = np.array([sat.inclo for sat in self.satrecs], dtype=np.float64)
        ecc = np.array([sat.ecco for sat in self.satrecs], dtype=np.float64)

        features = np.stack(
            [
                radius,
                speed,
                incl,
                ecc,
                np.sin(phases),
                np.cos(phases),
            ],
            axis=1,
        )
        return features.astype(np.float32)

    def _fit_autoencoder_on_nominal_data(self) -> None:
        samples = []
        base_time = datetime.utcnow()
        zero_offsets = np.zeros(self.num_satellites, dtype=np.float32)

        for k in range(20):
            t = base_time + timedelta(seconds=k * self.cfg.dt_seconds)
            pos, vel = self._propagate_at_time(t)
            phases = np.arctan2(pos[:, 1], pos[:, 0]) + zero_offsets
            sample = self._raw_feature_matrix(pos, vel, phases)
            samples.append(sample)

        train_data = np.concatenate(samples, axis=0)
        self.ae_mean = train_data.mean(axis=0)
        self.ae_std = train_data.std(axis=0) + 1e-6
        train_norm = (train_data - self.ae_mean) / self.ae_std

        x = torch.tensor(train_norm, dtype=torch.float32)

        self.autoencoder.train()
        for _ in range(self.cfg.ae_epochs):
            recon = self.autoencoder(x)
            loss = ((x - recon) ** 2).mean()
            self.ae_optimizer.zero_grad()
            loss.backward()
            self.ae_optimizer.step()

    def _compute_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        norm = (features - self.ae_mean) / self.ae_std
        x = torch.tensor(norm, dtype=torch.float32)

        self.autoencoder.eval()
        with torch.no_grad():
            recon = self.autoencoder(x).cpu().numpy()

        err = ((norm - recon) ** 2).mean(axis=1)
        return err.astype(np.float32)

    def _sample_fault_scenarios(self) -> None:
        self.fault_start_steps.fill(self.cfg.max_steps + 1)
        self.fault_end_steps.fill(0)
        self.fault_phase_drifts.fill(0.0)
        self.fault_radial_offsets.fill(0.0)
        self.fault_actuation_scales.fill(1.0)

        if not self.cfg.enable_fault_injection:
            return

        selected = self.rng.random(self.num_satellites) < self.cfg.fault_probability
        if not np.any(selected):
            return

        max_start = max(5, self.cfg.max_steps - self.cfg.fault_min_duration)
        for idx in np.flatnonzero(selected):
            start = int(self.rng.integers(5, max_start + 1))
            duration = int(
                self.rng.integers(
                    self.cfg.fault_min_duration,
                    self.cfg.fault_max_duration + 1,
                )
            )
            end = min(self.cfg.max_steps, start + duration)
            severity = float(self.rng.uniform(0.4, 1.0))

            self.fault_start_steps[idx] = start
            self.fault_end_steps[idx] = end
            self.fault_phase_drifts[idx] = severity * self.cfg.fault_phase_drift_scale
            self.fault_radial_offsets[idx] = severity * self.cfg.fault_radial_offset_km
            self.fault_actuation_scales[idx] = 1.0 - severity * self.rng.uniform(
                self.cfg.fault_actuation_loss_min,
                self.cfg.fault_actuation_loss_max,
            )

    def _current_fault_mask(self) -> np.ndarray:
        return (self.step_count >= self.fault_start_steps) & (self.step_count < self.fault_end_steps)

    def _apply_fault_offsets(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fault_mask = self._current_fault_mask()
        if not np.any(fault_mask):
            return positions, velocities

        positions = positions.copy()
        velocities = velocities.copy()
        radius = np.linalg.norm(positions, axis=1, keepdims=True)
        unit_radius = positions / np.clip(radius, 1e-6, None)
        positions[fault_mask] = positions[fault_mask] + unit_radius[fault_mask] * self.fault_radial_offsets[fault_mask, None]
        velocities[fault_mask] = velocities[fault_mask] * self.fault_actuation_scales[fault_mask, None]
        return positions, velocities

    def _compute_collision_penalty(self, positions: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.num_satellites < 2:
            return np.zeros(self.num_satellites, dtype=np.float32), float("inf")

        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, np.inf)

        min_distances = distances.min(axis=1)
        penalty = np.clip(
            (self.cfg.collision_distance_km - min_distances) / self.cfg.collision_distance_km,
            0.0,
            None,
        )
        return penalty.astype(np.float32), float(min_distances.min())

    def _compute_coverage_penalty(self, phases: np.ndarray) -> Tuple[float, float]:
        sorted_phases = np.sort(np.mod(phases, 2.0 * np.pi))
        wrapped = np.concatenate([sorted_phases, [sorted_phases[0] + 2.0 * np.pi]])
        gaps = np.diff(wrapped)
        ideal_gap = 2.0 * np.pi / max(1, self.num_satellites)
        max_gap = float(np.max(gaps))
        penalty = float(np.clip(max_gap / ideal_gap - 1.0, 0.0, 1.0))
        return penalty, max_gap

    @staticmethod
    def _angle_wrap(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _build_observation(
        self,
        phase_error: np.ndarray,
        altitude_error: np.ndarray,
        fuel: np.ndarray,
        anomaly: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        obs = np.stack(
            [
                np.sin(phases),
                np.cos(phases),
                phase_error,
                altitude_error,
                fuel,
                anomaly,
                np.full_like(fuel, self.step_count / max(1, self.cfg.max_steps), dtype=np.float32),
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.phase_offsets = np.zeros(self.num_satellites, dtype=np.float32)
        self.fuel = np.ones(self.num_satellites, dtype=np.float32)
        self.start_time = datetime.utcnow()
        self.trajectories = []
        self._sample_fault_scenarios()

        positions, velocities = self._propagate_at_time(self.start_time)
        positions, velocities = self._apply_fault_offsets(positions, velocities)
        self.latest_positions = positions.astype(np.float32)
        self.latest_velocities = velocities.astype(np.float32)

        radius = np.linalg.norm(positions, axis=1)
        self.target_radius = float(np.median(radius))

        phases = np.arctan2(positions[:, 1], positions[:, 0]) + self.phase_offsets
        phase_error = self._angle_wrap(phases - self.desired_phases)
        altitude_error = (radius - self.target_radius) / (self.target_radius + 1e-6)

        features = self._raw_feature_matrix(positions, velocities, phases)
        anomaly = self._compute_anomaly_scores(features)
        self.latest_anomaly = anomaly
        collision_penalty, min_separation_km = self._compute_collision_penalty(positions)
        coverage_penalty, max_gap = self._compute_coverage_penalty(phases)
        self.latest_collision_penalty = collision_penalty
        self.latest_coverage_penalty = coverage_penalty
        fault_mask = self._current_fault_mask()

        obs = self._build_observation(phase_error, altitude_error, self.fuel, anomaly, phases)
        self.trajectories.append(self.latest_positions.copy())

        info = {
            "phase_error_mean": float(np.mean(np.abs(phase_error))),
            "altitude_error_mean": float(np.mean(np.abs(altitude_error))),
            "anomaly_mean": float(np.mean(anomaly)),
            "collision_penalty_mean": float(np.mean(collision_penalty)),
            "coverage_penalty": float(coverage_penalty),
            "active_fault_count": int(np.sum(fault_mask)),
            "active_fault_fraction": float(np.mean(fault_mask.astype(np.float32))),
            "anomaly_event_fraction": float(np.mean((anomaly > self.cfg.anomaly_event_threshold).astype(np.float32))),
            "min_separation_km": float(min_separation_km),
            "max_gap_rad": float(max_gap),
        }
        return obs, info

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape[0] != self.num_satellites:
            raise ValueError(
                f"Expected {self.num_satellites} actions, received shape {actions.shape}."
            )

        command = actions - 1
        command = np.clip(command, -1, 1)

        next_step = self.step_count + 1
        fault_mask = (next_step >= self.fault_start_steps) & (next_step < self.fault_end_steps)
        actuation_scale = np.ones(self.num_satellites, dtype=np.float32)
        actuation_scale[fault_mask] = self.fault_actuation_scales[fault_mask]
        effective_command = command.astype(np.float32) * actuation_scale

        self.phase_offsets += effective_command * self.cfg.phase_gain
        self.phase_offsets += np.where(fault_mask, self.fault_phase_drifts, 0.0)
        control_used = np.abs(command).astype(np.float32)
        self.fuel = np.clip(self.fuel - control_used * self.cfg.fuel_cost, 0.0, 1.0)

        self.step_count += 1
        current_time = self.start_time + timedelta(seconds=self.step_count * self.cfg.dt_seconds)
        positions, velocities = self._propagate_at_time(current_time)
        positions, velocities = self._apply_fault_offsets(positions, velocities)

        self.latest_positions = positions.astype(np.float32)
        self.latest_velocities = velocities.astype(np.float32)

        radius = np.linalg.norm(positions, axis=1)
        phases = np.arctan2(positions[:, 1], positions[:, 0]) + self.phase_offsets
        phase_error = self._angle_wrap(phases - self.desired_phases)
        altitude_error = (radius - self.target_radius) / (self.target_radius + 1e-6)

        features = self._raw_feature_matrix(positions, velocities, phases)
        anomaly = self._compute_anomaly_scores(features)
        self.latest_anomaly = anomaly
        collision_penalty, min_separation_km = self._compute_collision_penalty(positions)
        coverage_penalty, max_gap = self._compute_coverage_penalty(phases)
        self.latest_collision_penalty = collision_penalty
        self.latest_coverage_penalty = coverage_penalty
        anomaly_event_fraction = float(
            np.mean((anomaly > self.cfg.anomaly_event_threshold).astype(np.float32))
        )

        per_agent_reward = -(
            self.cfg.phase_weight * np.abs(phase_error)
            + self.cfg.altitude_weight * np.abs(altitude_error)
            + self.cfg.control_weight * control_used
            + self.cfg.anomaly_weight * anomaly
            + self.cfg.collision_weight * collision_penalty
            + self.cfg.coverage_weight * coverage_penalty
        )

        close_alignment = (np.abs(phase_error) < 0.05).astype(np.float32)
        per_agent_reward += self.cfg.alignment_bonus * close_alignment

        reward = float(np.mean(per_agent_reward))
        terminated = self.step_count >= self.cfg.max_steps
        truncated = False

        obs = self._build_observation(phase_error, altitude_error, self.fuel, anomaly, phases)
        self.trajectories.append(self.latest_positions.copy())

        info = {
            "per_agent_reward": per_agent_reward.astype(np.float32),
            "phase_error_mean": float(np.mean(np.abs(phase_error))),
            "altitude_error_mean": float(np.mean(np.abs(altitude_error))),
            "anomaly_mean": float(np.mean(anomaly)),
            "collision_penalty_mean": float(np.mean(collision_penalty)),
            "coverage_penalty": float(coverage_penalty),
            "active_fault_count": int(np.sum(fault_mask)),
            "active_fault_fraction": float(np.mean(fault_mask.astype(np.float32))),
            "anomaly_event_fraction": anomaly_event_fraction,
            "min_separation_km": float(min_separation_km),
            "max_gap_rad": float(max_gap),
            "fuel_mean": float(np.mean(self.fuel)),
            "positions": self.latest_positions.copy(),
        }

        return obs, reward, terminated, truncated, info

    def get_latest_positions(self) -> np.ndarray:
        return self.latest_positions.copy()

    def get_trajectories(self) -> np.ndarray:
        return np.array(self.trajectories, dtype=np.float32)
