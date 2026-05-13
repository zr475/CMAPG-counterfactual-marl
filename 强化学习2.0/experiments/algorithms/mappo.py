"""MAPPO: Multi-Agent PPO baseline.

Standard PPO with centralized value function for multi-agent settings.
Based on Yu et al. (2022): "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games".
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from ..networks import MLP, CategoricalActor, GaussianActor, CentralizedCritic


class MAPPO:
    """Multi-Agent PPO with centralized value function."""

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        state_dim: int,
        action_dims: List[int],
        discrete_actions: bool = True,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dims = obs_dims if isinstance(obs_dims, list) else [obs_dims] * n_agents
        self.state_dim = state_dim
        self.action_dims = action_dims if isinstance(action_dims, list) else [action_dims] * n_agents
        self.discrete_actions = discrete_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        total_action_dim = sum(self.action_dims)

        self.actors = nn.ModuleList()
        for i in range(n_agents):
            if discrete_actions:
                actor = CategoricalActor(self.obs_dims[i], hidden_dim, self.action_dims[i])
            else:
                actor = GaussianActor(self.obs_dims[i], hidden_dim, self.action_dims[i])
            self.actors.append(actor)

        self.critic = CentralizedCritic(state_dim, total_action_dim, hidden_dim)
        self.target_critic = deepcopy(self.critic)

        self.optimizer = optim.Adam(
            list(self.actors.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    def to(self, device: str):
        self.device = device
        self.actors.to(device)
        self.critic.to(device)
        self.target_critic.to(device)

    def evaluate_advantages(
        self, obs_list: List[np.ndarray], state: np.ndarray, all_actions: np.ndarray
    ) -> np.ndarray:
        """Compute per-agent advantages for evaluation (uniform split for MAPPO)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        parts = []
        for i, a in enumerate(all_actions):
            if self.discrete_actions:
                a_t = torch.FloatTensor([a]).unsqueeze(0).to(self.device)
                onehot = torch.zeros(1, self.action_dims[i], device=self.device)
                onehot.scatter_(1, a_t.long(), 1)
                parts.append(onehot)
            else:
                a_t = torch.FloatTensor(a).view(1, -1).to(self.device)
                parts.append(a_t)
        actions_t = torch.cat(parts, dim=-1)

        with torch.no_grad():
            q = self.critic(state_t, actions_t).squeeze(-1)
        return (q.item() / self.n_agents) * np.ones(self.n_agents)

    def get_actions(self, obs_list: List[np.ndarray], deterministic: bool = False):
        actions = []
        log_probs = []
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            dist = self.actors[i](obs_tensor)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1) if self.discrete_actions else dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            if not self.discrete_actions:
                log_prob = log_prob.sum(dim=-1)
            actions.append(action.squeeze(0).detach().cpu().numpy())
            log_probs.append(log_prob.squeeze(0))
        return actions, log_probs

    def _one_hot(self, indices: torch.Tensor, n_classes: int) -> torch.Tensor:
        onehot = torch.zeros(indices.shape[0], n_classes, device=self.device)
        onehot.scatter_(1, indices.long(), 1)
        return onehot

    def update(
        self,
        obs_batch: List[torch.Tensor],
        actions_batch: List[torch.Tensor],
        rewards_batch: torch.Tensor,
        dones_batch: torch.Tensor,
        states_batch: torch.Tensor,
        old_log_probs_batch: List[torch.Tensor],
    ) -> Dict[str, float]:
        batch_size, seq_len = rewards_batch.shape

        parts = []
        for i, a in enumerate(actions_batch):
            a_flat = a.view(batch_size * seq_len, -1)
            if self.discrete_actions:
                parts.append(self._one_hot(a_flat, self.action_dims[i]))
            else:
                parts.append(a_flat)
        all_actions_flat = torch.cat(parts, dim=-1)
        states_flat = states_batch.view(batch_size * seq_len, -1)

        # Compute values
        with torch.no_grad():
            values = self.target_critic(states_flat, all_actions_flat)
            values = values.view(batch_size, seq_len)

        # GAE
        advantages = torch.zeros(batch_size, seq_len, device=self.device)
        returns = torch.zeros(batch_size, seq_len, device=self.device)
        for b in range(batch_size):
            gae = 0.0
            for t in reversed(range(seq_len)):
                next_val = values[b, t + 1] if t < seq_len - 1 else 0.0
                next_non_terminal = 1.0 - dones_batch[b, t].float()
                delta = rewards_batch[b, t] + self.gamma * next_val * next_non_terminal - values[b, t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[b, t] = gae
                returns[b, t] = gae + values[b, t]

        advantages_flat = advantages.view(-1)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        returns_flat = returns.view(-1)

        # Policy loss
        policy_loss = 0.0
        entropy_total = 0.0
        for i in range(self.n_agents):
            obs_i = obs_batch[i].view(batch_size * seq_len, -1)
            actions_i = actions_batch[i].view(batch_size * seq_len, -1)
            old_log_prob_i = old_log_probs_batch[i].view(batch_size * seq_len)

            dist = self.actors[i](obs_i)
            new_log_prob = dist.log_prob(
                actions_i.squeeze(-1) if self.discrete_actions else actions_i
            )
            if not self.discrete_actions:
                new_log_prob = new_log_prob.sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob_i)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_flat
            policy_loss += -torch.min(surr1, surr2).mean()
            entropy_total += dist.entropy().mean()

        policy_loss /= self.n_agents
        entropy_avg = entropy_total / self.n_agents

        # Value loss
        values_pred = self.critic(states_flat, all_actions_flat).view(-1)
        value_loss = F.mse_loss(values_pred, returns_flat)

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_avg

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actors.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # Update target
        for src, tgt in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt.data.copy_(0.005 * src.data + 0.995 * tgt.data)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_avg.item(),
            "total_loss": total_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.target_critic = deepcopy(self.critic)
