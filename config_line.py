from dataclasses import dataclass


@dataclass
class ConfigLine:
    # ---- Scenario -------------------------------------------------------
    num_satellites: int = 100
    max_steps: int = 90
    dt_seconds: int = 60
    seed: int = 42

    # ---- Orbital parameters (simplified circular orbit) -----------------
    # Mirrors Starlink shell 1: ~550 km altitude, 53° inclination
    altitude_km: float = 550.0
    inclination_deg: float = 53.0

    # Spacing between consecutive satellites at launch (radians).
    # 0.005 rad ≈ 0.29° → 30 sats span ~8.6° of orbital arc (~34 km gaps).
    initial_spacing_rad: float = 0.005

    # Each satellite receives a tiny random altitude offset at reset so that
    # their natural mean motions differ slightly.  Without control the gaps
    # slowly drift; the agent must cancel that drift.
    altitude_noise_km: float = 0.5

    # ---- Reward shaping -------------------------------------------------
    # Spacing error is normalised by initial_spacing_rad before weighting,
    # so these weights are scale-invariant.
    spacing_weight: float = 1.5
    control_weight: float = 0.05
    fuel_cost: float = 0.003

    # Maximum accumulated phase correction (rad) a satellite can hold.
    max_phase_correction: float = 0.05

    # Phase correction applied per unit action per step (rad).
    phase_gain: float = 0.0003

    # ---- Agent architecture ---------------------------------------------
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    action_dim: int = 3  # 0 = retard, 1 = hold, 2 = advance

    # ---- MAPPO training -------------------------------------------------
    train_iterations: int = 15
    rollout_horizon: int = 32
    ppo_epochs: int = 3
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate_start: float = 3e-4
    learning_rate_end: float = 1e-4
    adam_eps: float = 1e-5
    value_clip_eps: float = 0.2
    target_kl: float = 0.02
    normalize_advantages: bool = True

    # ---- Output paths (Step 2 directory) --------------------------------
    output_dir: str = "outputs/step2"
    checkpoint_dir: str = "outputs/step2/checkpoints"
    final_plot_name: str = "line_final.png"
    gif_name: str = "line_animation.gif"
    metrics_json_name: str = "line_training_metrics.json"
    metrics_csv_name: str = "line_training_metrics.csv"
    metrics_plot_name: str = "line_training_metrics.png"
    evaluation_json_name: str = "line_evaluation_metrics.json"
    latest_checkpoint_name: str = "line_mappo_latest.pt"
    best_checkpoint_name: str = "line_mappo_best.pt"
    policy_export_name: str = "line_policy_actor.pt"
    checkpoint_every: int = 5
    max_interval_checkpoints: int = 3
    resume_mode: str = "none"
    resume_checkpoint_path: str = ""
