"""SVPG: Shapley Value Policy Gradient.

Our proposed method that uses Shapley values from cooperative game theory
for credit assignment in multi-agent reinforcement learning.

Key insight: instead of learning a counterfactual baseline Ψ(s, a^{-i}) that
collapses to mirror Q(s,a), we compute Shapley values via explicit coalition
evaluation using the Q-function. Shapley values have axiomatic foundations
(efficiency, symmetry, null player, additivity) and cannot "collapse" because
they are computed, not learned.

For each agent i, the Shapley value φ_i is:
    φ_i(v) = Σ_{C⊆N\{i}} w(|C|) · [v(C∪{i}) - v(C)]
    w(k) = k!(n-k-1)! / n!

We approximate via Monte Carlo permutation sampling (K random permutations).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from ..networks import MLP, CategoricalActor, GaussianActor, CentralizedCritic


class SVPG:
    """Shapley Value Policy Gradient.

    Args:
        n_agents: Number of agents.
        obs_dims: Observation dimension per agent.
        state_dim: Global state dimension.
        action_dims: Action dimension per agent.
        discrete_actions: Whether actions are discrete.
        hidden_dim: Hidden layer dimension.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_epsilon: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Maximum gradient norm.
        K: Number of Monte Carlo permutations for Shapley estimation.
        device: Device to use.
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
        K: int = 20,
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
        self.K = K
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

        self._action_offsets = [0]
        for d in self.action_dims[:-1]:
            self._action_offsets.append(self._action_offsets[-1] + d)

    def to(self, device: str):
        self.device = device
        self.actors.to(device)
        self.critic.to(device)
        self.target_critic.to(device)

    def get_actions(
        self, obs_list: List[np.ndarray], deterministic: bool = False
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
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

    def evaluate_advantages(
        self, obs_list: List[np.ndarray], state: np.ndarray, all_actions: np.ndarray
    ) -> np.ndarray:
        """Compute per-agent Shapley value advantages for credit assignment evaluation."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions_t = self._actions_to_tensor(all_actions).unsqueeze(0)
        with torch.no_grad():
            shapley = self.compute_shapley_values(state_t, actions_t)
        return shapley.squeeze(0).cpu().numpy()

    def _actions_to_tensor(self, action_values: np.ndarray) -> torch.Tensor:
        """Convert action values (n_agents,) to tensor (total_action_dim,)."""
        if self.discrete_actions:
            return self._actions_to_onehot(action_values)
        else:
            parts = []
            for i in range(self.n_agents):
                a = np.atleast_1d(action_values[i])
                parts.append(torch.FloatTensor(a).to(self.device))
            return torch.cat(parts, dim=-1)

    def compute_shapley_values(
        self,
        states: torch.Tensor,          # (N_samples, state_dim)
        all_actions: torch.Tensor,     # (N_samples, total_action_dim)
    ) -> torch.Tensor:
        """Compute Monte Carlo Shapley values for each agent.

        For each of K random permutations of agents, we compute the marginal
        contribution of each agent when added to the coalition of agents
        preceding it in the permutation.

        Returns:
            shapley: (N_samples, n_agents) — Shapley values per agent.
        """
        N = states.shape[0]
        shapley = torch.zeros(N, self.n_agents, device=self.device)

        for _ in range(self.K):
            perm = torch.randperm(self.n_agents, device=self.device)

            # Build coalition actions for all prefixes of this permutation
            # prefix 0: empty coalition (all actions zeroed)
            # prefix 1: {perm[0]}
            # prefix 2: {perm[0], perm[1]}
            # ...
            # prefix n_agents: all agents (original actions)
            coalition_actions = []
            for prefix_len in range(self.n_agents + 1):
                coalition = set(perm[:prefix_len].tolist())
                mask = torch.zeros_like(all_actions)
                for ag_idx in coalition:
                    start = self._action_offsets[ag_idx]
                    end = start + self.action_dims[ag_idx]
                    mask[:, start:end] = 1.0
                coalition_actions.append(all_actions * mask)

            batched_actions = torch.cat(coalition_actions, dim=0)          # (N*(n+1), total_action_dim)
            batched_states = states.repeat(self.n_agents + 1, 1)           # (N*(n+1), state_dim)

            with torch.no_grad():
                q_values = self.target_critic(batched_states, batched_actions)
            q_values = q_values.view(self.n_agents + 1, N)                  # (n+1, N)

            for pos in range(self.n_agents):
                agent_idx = perm[pos].item()
                marginal = q_values[pos + 1] - q_values[pos]                # (N,)
                shapley[:, agent_idx] += marginal

        shapley /= self.K
        return shapley

    def _actions_to_onehot(self, action_indices: np.ndarray) -> torch.Tensor:
        """Convert action indices (n_agents,) to one-hot (total_action_dim,)."""
        parts = []
        for i in range(self.n_agents):
            onehot = torch.zeros(self.action_dims[i], device=self.device)
            onehot[int(action_indices[i])] = 1.0
            parts.append(onehot)
        return torch.cat(parts, dim=-1)

    def _one_hot(self, indices: torch.Tensor, n_classes: int) -> torch.Tensor:
        onehot = torch.zeros(indices.shape[0], n_classes, device=self.device)
        onehot.scatter_(1, indices.long(), 1)
        return onehot

    def _build_all_actions(self, actions_batch: List[torch.Tensor], batch_size: int, seq_len: int) -> torch.Tensor:
        parts = []
        for i, a in enumerate(actions_batch):
            a_flat = a.view(batch_size * seq_len, -1)
            if self.discrete_actions:
                parts.append(self._one_hot(a_flat, self.action_dims[i]))
            else:
                parts.append(a_flat)
        return torch.cat(parts, dim=-1)

    def update(
        self,
        obs_batch: List[torch.Tensor],         # n_agents * (B, T, obs_dim)
        actions_batch: List[torch.Tensor],     # n_agents * (B, T, action_dim)
        rewards_batch: torch.Tensor,            # (B, T)
        dones_batch: torch.Tensor,              # (B, T)
        states_batch: torch.Tensor,             # (B, T, state_dim)
        old_log_probs_batch: List[torch.Tensor],# n_agents * (B, T)
    ) -> Dict[str, float]:
        batch_size, seq_len = rewards_batch.shape
        flat_size = batch_size * seq_len

        all_actions_flat = self._build_all_actions(actions_batch, batch_size, seq_len)
        states_flat = states_batch.view(flat_size, -1)

        # ---- GAE with Q-function as value ----
        with torch.no_grad():
            q_values = self.target_critic(states_flat, all_actions_flat)
            q_values = q_values.view(batch_size, seq_len)

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

        # ---- Compute Shapley values for per-agent credit assignment ----
        with torch.no_grad():
            shapley = self.compute_shapley_values(states_flat, all_actions_flat)

        # ---- Credit assignment: Shapley-weighted GAE ----
        # Use softmax of Shapley values to distribute GAE advantage across agents.
        # Agents with higher Shapley contribution get proportionally larger advantage.
        # No per-agent normalization — preserves cross-agent credit differentiation.
        gae_flat = gae_adv.view(-1)  # (N,)
        shapley_props = torch.softmax(shapley, dim=-1)  # (N, n_agents) sums to 1
        shapley_adv = gae_flat.unsqueeze(-1) * shapley_props * self.n_agents  # (N, n_agents)

        # Global advantage normalization (shared across all agents)
        adv_mean = shapley_adv.mean()
        adv_std = shapley_adv.std()
        if adv_std > 1e-8:
            shapley_adv = (shapley_adv - adv_mean) / adv_std

        # ---- Diagnostics ----
        diag = {}
        with torch.no_grad():
            diag["shapley_mean"] = shapley.mean().item()
            diag["shapley_std"] = shapley.std().item()
            diag["shapley_max"] = shapley.max().item()
            diag["shapley_min"] = shapley.min().item()
            for i in range(self.n_agents):
                diag[f"shapley_a{i}"] = shapley[:, i].mean().item()
            # Entropy of Shapley distribution (lower = more differentiated credit)
            shapley_props_diag = torch.softmax(shapley, dim=-1)
            entropy = -(shapley_props_diag * torch.log(shapley_props_diag + 1e-8)).sum(dim=-1).mean()
            max_entropy = np.log(self.n_agents)
            diag["shapley_entropy"] = entropy.item()
            diag["shapley_entropy_norm"] = entropy.item() / max_entropy
            diag["q_pred_mean"] = q_values.mean().item()

        # ---- Policy loss (PPO with Shapley-weighted advantages) ----
        policy_loss = 0.0
        entropy_total = 0.0
        for i in range(self.n_agents):
            obs_i = obs_batch[i].view(flat_size, -1)
            actions_i = actions_batch[i].view(flat_size, -1)
            old_log_prob_i = old_log_probs_batch[i].view(flat_size)

            dist = self.actors[i](obs_i)
            new_log_prob = dist.log_prob(
                actions_i.squeeze(-1) if self.discrete_actions else actions_i
            )
            if not self.discrete_actions:
                new_log_prob = new_log_prob.sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob_i)
            adv_i = shapley_adv[:, i]

            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_i
            policy_loss += -torch.min(surr1, surr2).mean()
            entropy_total += dist.entropy().mean()

        policy_loss /= self.n_agents
        entropy_avg = entropy_total / self.n_agents

        # ---- Q-function TD loss (with coalition augmentation) ----
        q_pred = self.critic(states_flat, all_actions_flat).view(-1)
        value_loss = F.mse_loss(q_pred, returns.view(-1))

        # Coalition augmentation: train Q on random subsets of agents to improve
        # Shapley value estimates. With 30% probability, mask a random subset of agents.
        n_coalition = flat_size // 2
        if n_coalition > 0:
            coalition_idx = torch.randperm(flat_size, device=self.device)[:n_coalition]
            # Random coalition: each agent present with 50% probability
            coalition_present = torch.rand(n_coalition, self.n_agents, device=self.device) > 0.5
            # Ensure at least one agent present and at least one absent
            coalition_present[:, 0] = 1  # ensure at least one present
            if self.n_agents > 1:
                coalition_present[:, 1] = 0  # ensure at least one absent

            # Build masked actions for coalitions
            coalition_acts = all_actions_flat[coalition_idx].clone()
            for ag_idx in range(self.n_agents):
                start = self._action_offsets[ag_idx]
                end = start + self.action_dims[ag_idx]
                coalition_acts[:, start:end] *= coalition_present[:, ag_idx:ag_idx+1]

            coalition_q = self.critic(states_flat[coalition_idx], coalition_acts).view(-1)
            coalition_target = returns.view(-1)[coalition_idx]
            value_loss += 0.3 * F.mse_loss(coalition_q, coalition_target)

        # ---- Total loss ----
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_avg

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actors.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # Soft update target network
        for src, tgt in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt.data.copy_(0.005 * src.data + 0.995 * tgt.data)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_avg.item(),
            "total_loss": total_loss.item(),
            "shapley_mean": diag["shapley_mean"],
            "shapley_std": diag["shapley_std"],
            "shapley_entropy_norm": diag["shapley_entropy_norm"],
            "q_pred_mean": diag["q_pred_mean"],
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
