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
    anomaly_weight: float = 0.6
    phase_gain: float = 0.025
    fuel_cost: float = 0.003

    # Autoencoder
    ae_hidden_dim: int = 16
    ae_latent_dim: int = 4
    ae_epochs: int = 40
    ae_lr: float = 1e-3

    # MAPPO
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    action_dim: int = 3
    train_iterations: int = 12
    rollout_horizon: int = 32
    ppo_epochs: int = 3
    minibatch_size: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

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
