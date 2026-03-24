"""
StraightLineEnv — Step 2 scenario
===================================
30 satellites are placed in a single circular orbital plane at a fixed
altitude (550 km, i = 53°, RAAN = 0), initially forming a compact *train*
(equally spaced by ~0.29° ≈ 34 km gaps).

Each satellite receives a tiny random altitude offset at reset, giving it a
slightly different natural mean-motion.  Without control the inter-satellite
gaps slowly drift apart.  The multi-agent goal is to keep the chain straight:
all 30 gaps must remain equal to the initial spacing at all times.

Observation for satellite i (7 components)
-------------------------------------------
  0  sin(current_phase)
  1  cos(current_phase)
  2  normalised gap to next satellite  (gap_ahead  − d0) / d0
  3  normalised gap from prev satellite (gap_behind − d0) / d0
  4  normalised accumulated correction  phase_correction / max_phase_correction
  5  fuel remaining  ∈ [0, 1]
  6  time fraction   step / max_steps

For the first satellite (i=0) the gap_behind field is 0.
For the last  satellite (i=N-1) the gap_ahead  field is 0.

Actions  {0, 1, 2}  →  command  {−1, 0, +1}
  −1 : small retarding manoeuvre (phase correction decreases)
   0 : hold
  +1 : small advancing manoeuvre  (phase correction increases)

Reward (mean over all agents)
------------------------------
  per_agent = −( spacing_weight * |norm_gap_error|
               + control_weight * |command| )
  + 0.1  if gap error < 1 % of desired spacing  (alignment bonus)

Compatibility with train.py
-----------------------------
The step() info dict includes:
  "phase_error_mean"    ← spacing_error_mean  (used by train.py history)
  "altitude_error_mean" ← 0.0                 (not applicable here)
  "anomaly_mean"        ← 0.0                 (no autoencoder in this env)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config_line import ConfigLine


class StraightLineEnv(gym.Env):
    """Multi-agent straight-line constellation environment."""

    metadata = {"render_modes": []}
    MU_KM3_S2: float = 3.986004418e5  # Earth gravitational parameter (km³/s²)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: ConfigLine) -> None:
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.num_satellites = cfg.num_satellites

        # Nominal circular orbit
        self.a_nominal = 6371.0 + cfg.altitude_km         # km
        self.n_nominal = np.sqrt(self.MU_KM3_S2 / self.a_nominal ** 3)  # rad/s
        self.inc = np.radians(cfg.inclination_deg)

        self.obs_dim = 7
        self.action_dim = cfg.action_dim

        self.action_space = spaces.MultiDiscrete(
            [self.action_dim] * self.num_satellites
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_satellites, self.obs_dim),
            dtype=np.float32,
        )

        # Mutable state — initialised properly in reset()
        self.initial_phases: np.ndarray = np.zeros(self.num_satellites)
        self.n_values: np.ndarray = np.full(self.num_satellites, self.n_nominal)
        self.phase_corrections: np.ndarray = np.zeros(self.num_satellites)
        self.fuel: np.ndarray = np.ones(self.num_satellites, dtype=np.float32)
        self.step_count: int = 0

        self.latest_positions: np.ndarray = np.zeros(
            (self.num_satellites, 3), dtype=np.float32
        )
        self.trajectories: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _current_phases(self) -> np.ndarray:
        """True phase of each satellite at the current step (radians)."""
        return (
            self.initial_phases
            + self.n_values * self.step_count * self.cfg.dt_seconds
            + self.phase_corrections
        )

    def _phases_to_positions(self, phases: np.ndarray) -> np.ndarray:
        """Map orbital phases to 3-D ECI positions (km).

        Assumes circular orbit, RAAN = 0, argument of perigee = 0.
        Rotation from orbital plane to ECI:
          x_ECI = r · cos(θ)
          y_ECI = r · sin(θ) · cos(i)
          z_ECI = r · sin(θ) · sin(i)
        """
        r = self.a_nominal
        x = r * np.cos(phases)
        y = r * np.sin(phases) * np.cos(self.inc)
        z = r * np.sin(phases) * np.sin(self.inc)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _gap_errors(self, phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-satellite normalised gap errors.

        gap_ahead[i]  = (phases[i+1] − phases[i]) − d0   (0 for last sat)
        gap_behind[i] = (phases[i]   − phases[i-1]) − d0  (0 for first sat)

        Both are normalised by d0 = cfg.initial_spacing_rad for scale invariance.
        """
        d0 = self.cfg.initial_spacing_rad + 1e-12
        diffs = np.diff(phases)  # shape (N-1,)

        gap_ahead = np.concatenate([diffs - self.cfg.initial_spacing_rad, [0.0]])
        gap_behind = np.concatenate([[0.0], diffs - self.cfg.initial_spacing_rad])

        return (gap_ahead / d0).astype(np.float32), (gap_behind / d0).astype(np.float32)

    def _spacing_error_mean(self, phases: np.ndarray) -> float:
        """Mean absolute spacing error normalised by desired spacing."""
        d0 = self.cfg.initial_spacing_rad + 1e-12
        diffs = np.diff(phases)
        return float(np.mean(np.abs(diffs - self.cfg.initial_spacing_rad)) / d0)

    def _straightness_score(self, phases: np.ndarray) -> float:
        """Straightness ∈ [0, 1].  1.0 = perfectly equal spacing."""
        spacings = np.diff(phases)
        mean_s = spacings.mean()
        if mean_s < 1e-12:
            return 1.0
        return float(max(0.0, 1.0 - spacings.std() / mean_s))

    def _build_observation(
        self,
        phases: np.ndarray,
        gap_ahead: np.ndarray,
        gap_behind: np.ndarray,
    ) -> np.ndarray:
        time_frac = self.step_count / max(1, self.cfg.max_steps)
        corr_norm = np.clip(
            self.phase_corrections / (self.cfg.max_phase_correction + 1e-12),
            -1.0,
            1.0,
        )
        obs = np.stack(
            [
                np.sin(phases % (2.0 * np.pi)),
                np.cos(phases % (2.0 * np.pi)),
                gap_ahead,
                gap_behind,
                corr_norm,
                self.fuel,
                np.full(self.num_satellites, time_frac, dtype=np.float32),
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

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
        self.fuel = np.ones(self.num_satellites, dtype=np.float32)
        self.phase_corrections = np.zeros(self.num_satellites, dtype=np.float64)

        # Assign slightly different altitudes → slightly different mean motions.
        altitude_offsets = self.rng.normal(
            0.0, self.cfg.altitude_noise_km, size=self.num_satellites
        )
        a_i = self.a_nominal + altitude_offsets
        self.n_values = np.sqrt(self.MU_KM3_S2 / a_i ** 3)  # rad/s, shape (N,)

        # Initial phases: equal spacing with a tiny angular jitter (~2 % of gap).
        nominal = np.arange(self.num_satellites) * self.cfg.initial_spacing_rad
        jitter = self.rng.uniform(
            -self.cfg.initial_spacing_rad * 0.02,
            self.cfg.initial_spacing_rad * 0.02,
            size=self.num_satellites,
        )
        self.initial_phases = (nominal + jitter).astype(np.float64)

        phases = self._current_phases()
        self.latest_positions = self._phases_to_positions(phases)
        self.trajectories = [self.latest_positions.copy()]

        gap_ahead, gap_behind = self._gap_errors(phases)
        obs = self._build_observation(phases, gap_ahead, gap_behind)

        spacing_err_mean = self._spacing_error_mean(phases)
        info = {
            "spacing_error_mean": spacing_err_mean,
            "straightness_score": self._straightness_score(phases),
            "phase_spread_deg": float(np.degrees(phases[-1] - phases[0])),
            # Compatibility aliases for train.py and evaluate_policy()
            "phase_error_mean": spacing_err_mean,
            "altitude_error_mean": 0.0,
            "anomaly_mean": 0.0,
        }
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape[0] != self.num_satellites:
            raise ValueError(
                f"Expected {self.num_satellites} actions, got {actions.shape}."
            )

        command = (actions - 1).astype(np.float64)  # {−1, 0, +1}

        # Apply phase corrections (clipped to avoid runaway).
        self.phase_corrections = np.clip(
            self.phase_corrections + command * self.cfg.phase_gain,
            -self.cfg.max_phase_correction,
            self.cfg.max_phase_correction,
        )

        # Consume fuel.
        control_used = np.abs(command).astype(np.float32)
        self.fuel = np.clip(self.fuel - control_used * self.cfg.fuel_cost, 0.0, 1.0)

        self.step_count += 1

        # Propagate.
        phases = self._current_phases()
        self.latest_positions = self._phases_to_positions(phases)
        self.trajectories.append(self.latest_positions.copy())

        gap_ahead, gap_behind = self._gap_errors(phases)

        # Reward: penalise unequal spacing and unnecessary thrusting.
        mean_gap_err = 0.5 * (np.abs(gap_ahead) + np.abs(gap_behind))
        per_agent_reward = -(
            self.cfg.spacing_weight * mean_gap_err
            + self.cfg.control_weight * control_used
        )
        # Small bonus when gap error is within 1 % of desired spacing.
        aligned = (mean_gap_err < 0.01).astype(np.float32)
        per_agent_reward += 0.1 * aligned
        reward = float(np.mean(per_agent_reward))

        terminated = self.step_count >= self.cfg.max_steps

        obs = self._build_observation(phases, gap_ahead, gap_behind)

        spacing_err_mean = self._spacing_error_mean(phases)
        info: Dict[str, Any] = {
            "per_agent_reward": per_agent_reward.astype(np.float32),
            "spacing_error_mean": spacing_err_mean,
            "straightness_score": self._straightness_score(phases),
            "phase_spread_deg": float(np.degrees(phases[-1] - phases[0])),
            "fuel_mean": float(np.mean(self.fuel)),
            "positions": self.latest_positions.copy(),
            # Compatibility aliases for train.py
            "phase_error_mean": spacing_err_mean,
            "altitude_error_mean": 0.0,
            "anomaly_mean": 0.0,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_latest_positions(self) -> np.ndarray:
        return self.latest_positions.copy()

    def get_trajectories(self) -> np.ndarray:
        return np.array(self.trajectories, dtype=np.float32)
