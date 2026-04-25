from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    num_satellites: int = 100
    max_steps: int = 90
    dt_seconds: int = 60
    seed: int = 42

    # Reward shaping
    phase_weight: float = 1.0
    altitude_weight: float = 0.7
    control_weight: float = 0.05
    anomaly_weight: float = 0.3
    collision_weight: float = 0.9
    coverage_weight: float = 0.35
    phase_gain: float = 0.025
    fuel_cost: float = 0.003
    alignment_bonus: float = 0.1
    collision_distance_km: float = 25.0
    anomaly_event_threshold: float = 1.2

    # Fault injection
    enable_fault_injection: bool = True
    fault_probability: float = 0.08
    fault_min_duration: int = 12
    fault_max_duration: int = 36
    fault_phase_drift_scale: float = 0.012
    fault_radial_offset_km: float = 20.0
    fault_actuation_loss_min: float = 0.25
    fault_actuation_loss_max: float = 0.75

    # Autoencoder
    ae_hidden_dim: int = 16
    ae_latent_dim: int = 4
    ae_epochs: int = 40
    ae_lr: float = 1e-3

    # MAPPO
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    action_dim: int = 3
    train_iterations: int = 100
    rollout_horizon: int = 32
    ppo_epochs: int = 5
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.008
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate_start: float = 5e-4
    learning_rate_end: float = 5e-5
    adam_eps: float = 1e-5
    value_clip_eps: float = 0.2
    target_kl: float = 0.015
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    normalize_returns: bool = True

    # Output
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    final_plot_name: str = "constellation_final.png"
    gif_name: str = "constellation_animation.gif"
    metrics_json_name: str = "training_metrics.json"
    metrics_csv_name: str = "training_metrics.csv"
    metrics_plot_name: str = "training_metrics.png"
    evaluation_json_name: str = "evaluation_metrics.json"
    latest_checkpoint_name: str = "mappo_latest.pt"
    best_checkpoint_name: str = "mappo_best.pt"
    policy_export_name: str = "policy_actor.pt"
    checkpoint_every: int = 4
    max_interval_checkpoints: int = 3
    resume_mode: str = "none"
    resume_checkpoint_path: str = ""
