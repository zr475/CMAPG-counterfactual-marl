"""QMIX: Monotonic Value Decomposition baseline.

Based on Rashid et al. (2018): "QMIX: Monotonic Value Function Factorisation
for Deep Multi-Agent Reinforcement Learning".
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from ..networks import MLP, QMIXMixingNet


class QMIXAgent(nn.Module):
    """Individual agent Q-network."""

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.q = MLP(obs_dim, [hidden_dim, hidden_dim], n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q(obs)


class QMIX:
    """QMIX algorithm."""

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_anneal: int = 50000,
        target_update_freq: int = 200,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal = epsilon_anneal
        self.target_update_freq = target_update_freq
        self.device = device
        self.steps = 0

        obs_dims = obs_dims if isinstance(obs_dims, list) else [obs_dims] * n_agents

        # Agent Q-networks
        self.agent_qs = nn.ModuleList()
        for i in range(n_agents):
            self.agent_qs.append(QMIXAgent(obs_dims[i], hidden_dim, n_actions))

        # Target agent Q-networks
        self.target_agent_qs = deepcopy(self.agent_qs)

        # Mixing network
        self.mixer = QMIXMixingNet(state_dim, n_agents, hidden_dim)
        self.target_mixer = deepcopy(self.mixer)

        self.optimizer = optim.Adam(
            list(self.agent_qs.parameters()) + list(self.mixer.parameters()),
            lr=lr,
        )

    def to(self, device: str):
        self.device = device
        self.agent_qs.to(device)
        self.target_agent_qs.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)

    def select_actions(self, obs_list: List[np.ndarray], explore: bool = True) -> np.ndarray:
        actions = np.zeros(self.n_agents, dtype=np.int64)
        self.steps += 1

        with torch.no_grad():
            for i in range(self.n_agents):
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                q_values = self.agent_qs[i](obs_t)  # (1, n_actions)

                if explore and np.random.random() < self.epsilon:
                    actions[i] = np.random.randint(self.n_actions)
                else:
                    actions[i] = q_values.argmax(dim=-1).item()

        self._update_epsilon()
        return actions

    def _update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_anneal,
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = [batch[f"obs_{i}"] for i in range(self.n_agents)]
        next_obs = [batch[f"next_obs_{i}"] for i in range(self.n_agents)]
        actions = batch["actions"]  # (B, n_agents)
        rewards = batch["rewards"].squeeze(-1)  # (B,)
        dones = batch["dones"].squeeze(-1)  # (B,)
        states = batch["states"]  # (B, state_dim)
        next_states = batch["next_states"]  # (B, state_dim)

        batch_size = rewards.shape[0]

        # Get current Q-values per agent
        agent_qs = torch.zeros(batch_size, self.n_agents, self.n_actions, device=self.device)
        target_next_qs = torch.zeros(batch_size, self.n_agents, self.n_actions, device=self.device)

        for i in range(self.n_agents):
            agent_qs[:, i] = self.agent_qs[i](obs[i])
            target_next_qs[:, i] = self.target_agent_qs[i](next_obs[i])

        # Gather Q-values for taken actions
        chosen_qs = torch.gather(
            agent_qs, dim=-1, index=actions[:, :, None].long()
        ).squeeze(-1)  # (B, n_agents)

        # Mix to get Q_tot
        q_tot = self.mixer(chosen_qs, states)

        # Target Q_tot
        with torch.no_grad():
            # Double Q-learning: pick best action using current network
            next_actions = agent_qs.max(dim=-1)[1]  # Only used for argmax; use target for values
            max_target_next_qs = target_next_qs.max(dim=-1)[0]  # (B, n_agents)
            target_q_tot = self.target_mixer(max_target_next_qs, next_states)
            target = rewards + self.gamma * target_q_tot * (1 - dones)

        # Loss
        td_error = q_tot - target.detach()
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_qs.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10.0)
        self.optimizer.step()

        # Update target networks
        if self.steps % self.target_update_freq == 0:
            self.target_agent_qs = deepcopy(self.agent_qs)
            self.target_mixer = deepcopy(self.mixer)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str):
        torch.save({
            "agent_qs": self.agent_qs.state_dict(),
            "target_agent_qs": self.target_agent_qs.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent_qs.load_state_dict(ckpt["agent_qs"])
        self.target_agent_qs.load_state_dict(ckpt["target_agent_qs"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(ckpt["target_mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", self.epsilon)
