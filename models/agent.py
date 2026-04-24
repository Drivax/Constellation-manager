from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class RunningMeanStd:
    """Welford online algorithm for running mean and variance of scalar sequences."""

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        batch_count = float(x.size)
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


def _orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


def build_actor_network(obs_dim: int, action_dim: int, hidden_dim: int) -> nn.Sequential:
    actor = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, action_dim),
    )
    actor[0].apply(lambda module: _orthogonal_init(module, gain=np.sqrt(2.0)))
    actor[2].apply(lambda module: _orthogonal_init(module, gain=np.sqrt(2.0)))
    actor[4].apply(lambda module: _orthogonal_init(module, gain=0.01))
    return actor


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
    ) -> None:
        super().__init__()

        self.actor = build_actor_network(obs_dim, action_dim, actor_hidden_dim)

        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self.critic[0].apply(lambda module: _orthogonal_init(module, gain=np.sqrt(2.0)))
        self.critic[2].apply(lambda module: _orthogonal_init(module, gain=np.sqrt(2.0)))
        self.critic[4].apply(lambda module: _orthogonal_init(module, gain=1.0))

    def get_action_and_value(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(local_obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(global_obs).squeeze(-1)
        return action, log_prob, entropy, value


@dataclass
class MAPPOHyperParams:
    gamma: float
    gae_lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    ppo_epochs: int
    minibatch_size: int
    max_grad_norm: float
    learning_rate_start: float
    learning_rate_end: float
    adam_eps: float
    value_clip_eps: float
    target_kl: float
    normalize_advantages: bool
    normalize_rewards: bool
    normalize_returns: bool


class MAPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
        hparams: MAPPOHyperParams,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.hparams = hparams
        self.model = ActorCritic(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=action_dim,
            actor_hidden_dim=actor_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=hparams.learning_rate_start,
            eps=hparams.adam_eps,
        )
        self.reward_rms = RunningMeanStd()
        self.return_rms = RunningMeanStd()

    def set_learning_rate(self, learning_rate: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(learning_rate)

    def select_action(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        global_t = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model.actor(obs_t)
            dist = Categorical(logits=logits)
            if deterministic:
                actions_t = torch.argmax(logits, dim=-1)
            else:
                actions_t = dist.sample()
            log_prob_t = dist.log_prob(actions_t)
            value_t = self.model.critic(global_t).item()

        return actions_t.cpu().numpy(), log_prob_t.cpu().numpy(), float(value_t)

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.hparams.gamma * next_value * non_terminal - values[t]
            gae = delta + self.hparams.gamma * self.hparams.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: Dict[str, np.ndarray], next_value: float) -> Dict[str, float]:
        obs = rollout["obs"]
        global_obs = rollout["global_obs"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]

        mean_rewards = rewards.mean(axis=1).astype(np.float32)
        if self.hparams.normalize_rewards:
            self.reward_rms.update(mean_rewards)
            mean_rewards = mean_rewards / self.reward_rms.std

        advantages, returns = self._compute_gae(mean_rewards, dones, values, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_steps, num_agents, obs_dim = obs.shape
        flat_size = t_steps * num_agents

        obs_flat = obs.reshape(flat_size, obs_dim)
        global_flat = np.repeat(global_obs, num_agents, axis=0)
        actions_flat = actions.reshape(flat_size)
        old_log_flat = old_log_probs.reshape(flat_size)
        adv_flat = np.repeat(advantages, num_agents)
        ret_flat = np.repeat(returns, num_agents)
        old_val_flat = np.repeat(values, num_agents)

        if self.hparams.normalize_returns:
            self.return_rms.update(ret_flat)
            ret_mean = float(self.return_rms.mean)
            ret_std = self.return_rms.std
            ret_flat = (ret_flat - ret_mean) / ret_std
            old_val_flat = (old_val_flat - ret_mean) / ret_std

        actor_losses = []
        critic_losses = []
        entropies = []
        approx_kls = []
        clipfracs = []

        indices = np.arange(flat_size)
        stop_early = False

        for _ in range(self.hparams.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, flat_size, self.hparams.minibatch_size):
                end = min(start + self.hparams.minibatch_size, flat_size)
                mb_idx = indices[start:end]

                obs_b = torch.tensor(obs_flat[mb_idx], dtype=torch.float32, device=self.device)
                global_b = torch.tensor(global_flat[mb_idx], dtype=torch.float32, device=self.device)
                act_b = torch.tensor(actions_flat[mb_idx], dtype=torch.long, device=self.device)
                old_log_b = torch.tensor(old_log_flat[mb_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.tensor(adv_flat[mb_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.tensor(ret_flat[mb_idx], dtype=torch.float32, device=self.device)
                old_val_b = torch.tensor(old_val_flat[mb_idx], dtype=torch.float32, device=self.device)

                if self.hparams.normalize_advantages and adv_b.numel() > 1:
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                _, new_log_b, entropy_b, value_b = self.model.get_action_and_value(obs_b, global_b, action=act_b)

                log_ratio = new_log_b - old_log_b
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.hparams.clip_eps, 1.0 + self.hparams.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_val_b + (value_b - old_val_b).clamp(
                    -self.hparams.value_clip_eps,
                    self.hparams.value_clip_eps,
                )
                value_losses = (value_b - ret_b) ** 2
                value_losses_clipped = (value_pred_clipped - ret_b) ** 2
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                entropy = entropy_b.mean()

                loss = actor_loss + self.hparams.value_coef * critic_loss - self.hparams.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.hparams.clip_eps).float().mean()
                    approx_kls.append(float(approx_kl.item()))
                    clipfracs.append(float(clipfrac.item()))

                if self.hparams.target_kl > 0.0 and approx_kl.item() > self.hparams.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        with torch.no_grad():
            global_eval_t = torch.tensor(global_flat, dtype=torch.float32, device=self.device)
            values_pred = self.model.critic(global_eval_t).squeeze(-1).cpu().numpy()
            var_returns = np.var(ret_flat)
            explained_var = float(1.0 - np.var(ret_flat - values_pred) / (var_returns + 1e-8))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_var": explained_var,
            "mean_reward": float(np.mean(mean_rewards)),
        }


def load_exported_actor(policy_path: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(policy_path, map_location=device)
    actor = build_actor_network(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(torch.device(device))
    actor.eval()
    return actor, payload
