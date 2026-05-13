"""COMA: Counterfactual Multi-Agent Policy Gradients.

Foerster et al. (2018): "Counterfactual Multi-Agent Policy Gradients" (AAAI 2018).

COMA computes a counterfactual baseline by marginalizing over one agent's
actions using the current policy: b(s, a^{-i}) = Σ_{a'} π_i(a'|τ_i) Q(s, (a^{-i}, a')).

Unlike CMAPG, COMA enumerates all actions (exact marginalization) rather than
learning a separate Ψ network. This is O(n * |A|) per advantage computation.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from ..networks import MLP, CategoricalActor, GaussianActor, CentralizedCritic


class COMA:
    """COMA with PPO backbone for fair comparison with CMAPG/MAPPO.

    Key difference from MAPPO: uses counterfactual baseline (marginalization
    over actions) instead of uniform advantage splitting.
    """

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

        if not discrete_actions:
            raise NotImplementedError("COMA currently only supports discrete actions")

        total_action_dim = sum(self.action_dims)

        self.actors = nn.ModuleList()
        for i in range(n_agents):
            actor = CategoricalActor(self.obs_dims[i], hidden_dim, self.action_dims[i])
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
        """Compute per-agent counterfactual advantages for evaluation.

        Exact marginalization over each agent's actions.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        obs_tensors = [torch.FloatTensor(o).unsqueeze(0).to(self.device) for o in obs_list]

        with torch.no_grad():
            advantages = self._compute_counterfactual_advantages(
                state_t, obs_tensors, all_actions
            )
        return advantages.squeeze(0).cpu().numpy()

    def _one_hot_encode(self, indices: torch.Tensor, n_classes: int) -> torch.Tensor:
        onehot = torch.zeros(indices.shape[0], n_classes, device=self.device)
        onehot.scatter_(1, indices.long(), 1)
        return onehot

    def _build_action_tensor(self, action_indices: List[int]) -> torch.Tensor:
        """Build concatenated one-hot action tensor from action indices. (1, total_action_dim)."""
        parts = []
        for i, a_idx in enumerate(action_indices):
            idx_t = torch.tensor([[a_idx]], device=self.device).float()
            onehot = self._one_hot_encode(idx_t, self.action_dims[i])
            parts.append(onehot)
        return torch.cat(parts, dim=-1)

    def _compute_counterfactual_advantages(
        self,
        state: torch.Tensor,              # (B, state_dim)
        obs_list: List[torch.Tensor],     # n_agents * (B, obs_dim)
        action_indices: np.ndarray,       # (n_agents,) or (B, n_agents)
    ) -> torch.Tensor:
        """Compute COMA counterfactual advantages: A_i = Q(s,a) - Σ π_i(a') Q(s,(a^{-i},a'))."""
        batch_size = state.shape[0]

        # Build actual action one-hot encoding
        if action_indices.ndim == 1:
            action_indices = action_indices.reshape(1, -1)
        if action_indices.shape[0] == 1:
            action_indices = np.tile(action_indices, (batch_size, 1))

        actual_actions = []
        for b in range(batch_size):
            actual_actions.append(self._build_action_tensor(
                [int(action_indices[b, i]) for i in range(self.n_agents)]
            ))
        actual_actions = torch.cat(actual_actions, dim=0)  # (B, total_action_dim)

        # Q(s, a) for actual joint action
        q_actual = self.critic(state, actual_actions)  # (B, 1)

        advantages = torch.zeros(batch_size, self.n_agents, device=self.device)

        for i in range(self.n_agents):
            # Compute baseline b(s, a^{-i}) = Σ_{a'} π_i(a'|obs_i) * Q(s, (a^{-i}, a'))
            dist_i = self.actors[i](obs_list[i])  # Categorical distribution
            probs = dist_i.probs  # (B, n_actions_i)

            # Build (a^{-i}, a') for each possible a' and batch element
            baseline = torch.zeros(batch_size, device=self.device)

            for a_val in range(self.action_dims[i]):
                # Build counterfactual joint action: replace agent i's action with a_val
                cf_actions = actual_actions.clone()
                # Remove agent i's one-hot and insert a_val one-hot
                start_i = sum(self.action_dims[:i])
                end_i = start_i + self.action_dims[i]
                # Zero out agent i's action
                cf_actions[:, start_i:end_i] = 0
                # Set the counterfactual action
                for b in range(batch_size):
                    cf_actions[b, start_i + a_val] = 1.0

                q_counterfactual = self.critic(state, cf_actions).squeeze(-1)  # (B,)
                baseline += probs[:, a_val] * q_counterfactual

            advantages[:, i] = q_actual.squeeze(-1) - baseline

        return advantages  # (B, n_agents)

    def get_actions(self, obs_list: List[np.ndarray], deterministic: bool = False):
        actions = []
        log_probs = []
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            dist = self.actors[i](obs_tensor)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action.squeeze(0).detach().cpu().numpy())
            log_probs.append(log_prob.squeeze(0))
        return actions, log_probs

    def _one_hot(self, indices: torch.Tensor, n_classes: int) -> torch.Tensor:
        onehot = torch.zeros(indices.shape[0], n_classes, device=self.device)
        onehot.scatter_(1, indices.long(), 1)
        return onehot

    def update(
        self,
        obs_batch: List[torch.Tensor],       # n_agents * (B, T, obs_dim)
        actions_batch: List[torch.Tensor],   # n_agents * (B, T, action_dim)
        rewards_batch: torch.Tensor,          # (B, T)
        dones_batch: torch.Tensor,            # (B, T)
        states_batch: torch.Tensor,           # (B, T, state_dim)
        old_log_probs_batch: List[torch.Tensor],  # n_agents * (B, T)
    ) -> Dict[str, float]:
        batch_size, seq_len = rewards_batch.shape
        flat_size = batch_size * seq_len

        # Build flat action tensor
        parts = []
        for i, a in enumerate(actions_batch):
            a_flat = a.view(flat_size, -1)
            parts.append(self._one_hot(a_flat, self.action_dims[i]))
        all_actions_flat = torch.cat(parts, dim=-1)
        states_flat = states_batch.view(flat_size, -1)

        # Compute Q values and GAE
        with torch.no_grad():
            q_values = self.target_critic(states_flat, all_actions_flat)
            q_values = q_values.view(batch_size, seq_len)

        # GAE computation
        gae_adv = torch.zeros(batch_size, seq_len, device=self.device)
        returns = torch.zeros(batch_size, seq_len, device=self.device)
        for b in range(batch_size):
            gae = 0.0
            for t in reversed(range(seq_len)):
                next_val = q_values[b, t + 1] if t < seq_len - 1 else 0.0
                next_non_terminal = 1.0 - dones_batch[b, t].float()
                delta = rewards_batch[b, t] + self.gamma * next_val * next_non_terminal - q_values[b, t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                gae_adv[b, t] = gae
                returns[b, t] = gae + q_values[b, t]

        returns_flat = returns.view(-1)

        # Compute COMA counterfactual advantages (batched to avoid OOM)
        obs_flat = [o.view(flat_size, -1) for o in obs_batch]
        actions_flat_idx = []
        for i in range(self.n_agents):
            a_i = actions_batch[i].view(flat_size, -1)
            if self.discrete_actions:
                actions_flat_idx.append(a_i.squeeze(-1).cpu().numpy().astype(int))
            else:
                actions_flat_idx.append(a_i.cpu().numpy())

        actions_idx_np = np.stack(actions_flat_idx, axis=-1)  # (flat_size, n_agents)

        # Process in chunks to avoid memory issues, detach from computation graph
        chunk_size = 512
        cf_adv_raw = torch.zeros(flat_size, self.n_agents, device=self.device)
        for start in range(0, flat_size, chunk_size):
            end = min(start + chunk_size, flat_size)
            chunk_state = states_flat[start:end]
            chunk_obs = [o[start:end] for o in obs_flat]
            chunk_actions = actions_idx_np[start:end]
            cf_adv_raw[start:end] = self._compute_counterfactual_advantages(
                chunk_state, chunk_obs, chunk_actions
            )
        cf_adv_raw = cf_adv_raw.detach()

        # Normalize per-agent advantages (no gradient through advantages)
        cf_advantages_flat = torch.zeros_like(cf_adv_raw)
        for i in range(self.n_agents):
            adv_i = cf_adv_raw[:, i]
            cf_advantages_flat[:, i] = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

        # Policy loss (PPO with COMA advantages)
        policy_loss = 0.0
        entropy_total = 0.0
        for i in range(self.n_agents):
            obs_i = obs_batch[i].view(flat_size, -1)
            actions_i = actions_batch[i].view(flat_size, -1)
            old_log_prob_i = old_log_probs_batch[i].view(flat_size)

            dist = self.actors[i](obs_i)
            new_log_prob = dist.log_prob(actions_i.squeeze(-1))
            if not self.discrete_actions:
                new_log_prob = new_log_prob.sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob_i)
            adv_i = cf_advantages_flat[:, i]

            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_i
            policy_loss += -torch.min(surr1, surr2).mean()
            entropy_total += dist.entropy().mean()

        policy_loss /= self.n_agents
        entropy_avg = entropy_total / self.n_agents

        # Value loss
        q_pred = self.critic(states_flat, all_actions_flat).view(-1)
        value_loss = F.mse_loss(q_pred, returns_flat)

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
