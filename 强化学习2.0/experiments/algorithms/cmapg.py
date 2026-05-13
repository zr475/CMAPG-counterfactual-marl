"""CMAPG: Counterfactual Multi-Agent Policy Gradient.

Our proposed method that explicitly models credit assignment through
counterfactual reasoning.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from ..networks import (
    MLP, CategoricalActor, GaussianActor, CentralizedCritic,
    CounterfactualValueNet, ClubMIEstimator,
)


class CMAPG:
    """Counterfactual Multi-Agent Policy Gradient.

    Args:
        n_agents: Number of agents.
        obs_dims: Observation dimension per agent (list or int).
        state_dim: Global state dimension.
        action_dims: Action dimension per agent (list or int, or discrete n_actions).
        discrete_actions: Whether actions are discrete.
        hidden_dim: Hidden layer dimension.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_epsilon: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient.
        lambda_q: Weight for Q-function loss.
        lambda_psi: Weight for counterfactual model loss.
        lambda_mi: Weight for mutual information regularization.
        alpha_default: Mixing coefficient for default policy (0=uniform, 1=current policy).
        target_tau: Soft update coefficient for target networks.
        value_coef: Value loss coefficient.
        max_grad_norm: Maximum gradient norm.
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
        lambda_q: float = 1.0,
        lambda_psi: float = 0.5,
        lambda_mi: float = 0.1,
        alpha_default: float = 0.5,
        target_tau: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        epsilon_anneal: int = 200000,
        entropy_decay: float = 0.995,
        adv_mode: str = "softmax_gae",
        device: str = "cpu",
    ):
        """Args:
            adv_mode: How to compute per-agent advantages.
                "softmax_gae": GAE * softmax(Q-Ψ) [current default]
                "direct_cf": Q - Ψ directly
                "normalized_cf": Q - Ψ with per-agent normalization
        """
        self.n_agents = n_agents
        self.obs_dims = obs_dims if isinstance(obs_dims, list) else [obs_dims] * n_agents
        self.state_dim = state_dim
        self.action_dims = action_dims if isinstance(action_dims, list) else [action_dims] * n_agents
        self.discrete_actions = discrete_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.base_entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.lambda_q = lambda_q
        self.lambda_psi = lambda_psi
        self.lambda_mi = lambda_mi
        self.alpha_default = alpha_default
        self.target_tau = target_tau
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal = epsilon_anneal
        self.adv_mode = adv_mode
        self.device = device
        self.steps = 0

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

        self.psi_nets = nn.ModuleList()
        self.target_psi_nets = nn.ModuleList()
        for i in range(n_agents):
            psi = CounterfactualValueNet(state_dim, self.action_dims[i], hidden_dim)
            self.psi_nets.append(psi)
            self.target_psi_nets.append(deepcopy(psi))

        self.club_estimators = nn.ModuleList()
        for i in range(n_agents):
            club = ClubMIEstimator(self.action_dims[i], 1, hidden_dim)
            self.club_estimators.append(club)

        self.optimizer = optim.Adam(
            list(self.actors.parameters())
            + list(self.critic.parameters())
            + list(self.psi_nets.parameters())
            + list(self.club_estimators.parameters()),
            lr=lr,
        )

    def to(self, device: str):
        self.device = device
        self.actors.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.psi_nets.to(device)
        self.target_psi_nets.to(device)
        self.club_estimators.to(device)

    def evaluate_advantages(
        self, obs_list: List[np.ndarray], state: np.ndarray, all_actions: np.ndarray
    ) -> np.ndarray:
        """Compute per-agent counterfactual advantages for evaluation.

        Returns A_i^{cf} = Q(s,a) - Ψ(s, a^{-i}) for each agent, shape (n_agents,).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # One-hot encode discrete actions for critic input
        parts = []
        for i in range(self.n_agents):
            if self.discrete_actions:
                a_t = torch.FloatTensor([all_actions[i]]).unsqueeze(0).to(self.device)
                onehot = torch.zeros(1, self.action_dims[i], device=self.device)
                onehot.scatter_(1, a_t.long(), 1)
                parts.append(onehot)
            else:
                a_t = torch.FloatTensor(all_actions[i]).view(1, -1).to(self.device)
                parts.append(a_t)
        actions_t = torch.cat(parts, dim=-1)

        with torch.no_grad():
            q = self.critic(state_t, actions_t).squeeze(-1)  # (1,)
            psi = self.get_counterfactual_value(state_t, actions_t)  # (1, n_agents)
            advantages = (q.unsqueeze(-1) - psi).squeeze(0)  # (n_agents,)
        return advantages.cpu().numpy()

    def get_actions(
        self, obs_list: List[np.ndarray], deterministic: bool = False
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """Sample actions from current policies with epsilon-greedy exploration."""
        self.steps += 1
        self._update_epsilon()
        actions = []
        log_probs = []
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            dist = self.actors[i](obs_tensor)
            if deterministic:
                if self.discrete_actions:
                    action = torch.argmax(dist.probs, dim=-1)
                else:
                    action = dist.mean
            else:
                if self.discrete_actions and np.random.random() < self.epsilon:
                    action = torch.randint(0, self.action_dims[i], (1,), device=self.device)
                else:
                    action = dist.sample()
            log_prob = dist.log_prob(action)
            if not self.discrete_actions:
                log_prob = log_prob.sum(dim=-1)
            actions.append(action.squeeze(0).detach().cpu().numpy())
            log_probs.append(log_prob.squeeze(0))
        return actions, log_probs

    def _update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_anneal,
        )

    def get_counterfactual_value(
        self,
        states: torch.Tensor,       # (B, state_dim)
        all_actions: torch.Tensor,  # (B, total_action_dim)
    ) -> torch.Tensor:
        """Compute counterfactual values for all agents.

        Returns Ψ(s, a^{-i}) for each agent i, shape (B, n_agents).
        """
        batch_size = states.shape[0]
        psi_values = torch.zeros(batch_size, self.n_agents, device=self.device)
        action_offset = 0

        for i in range(self.n_agents):
            a_dim = self.action_dims[i]
            a_i = all_actions[:, action_offset:action_offset + a_dim]

            # Construct a^{-i}: all actions except agent i's
            other_actions_list = []
            for j in range(self.n_agents):
                if j != i:
                    offset_j = sum(self.action_dims[:j])
                    other_actions_list.append(all_actions[:, offset_j:offset_j + self.action_dims[j]])

            if other_actions_list:
                other_actions = torch.stack(other_actions_list, dim=1)  # (B, n-1, a_dim_j)
            else:
                other_actions = torch.zeros(batch_size, 0, 1, device=self.device)

            psi = self.psi_nets[i](states, other_actions)
            psi_values[:, i] = psi
            action_offset += a_dim

        return psi_values

    def _one_hot(self, indices: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Convert discrete action indices to one-hot encoding."""
        onehot = torch.zeros(indices.shape[0], n_classes, device=self.device)
        onehot.scatter_(1, indices.long(), 1)
        return onehot

    def _build_all_actions(self, actions_batch: List[torch.Tensor], batch_size: int, seq_len: int) -> torch.Tensor:
        """Build concatenated action tensor, one-hot encoding discrete actions."""
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
        obs_batch: List[torch.Tensor],       # n_agents * (B, T, obs_dim)
        actions_batch: List[torch.Tensor],   # n_agents * (B, T, action_dim)
        rewards_batch: torch.Tensor,          # (B, T)
        dones_batch: torch.Tensor,            # (B, T)
        states_batch: torch.Tensor,           # (B, T, state_dim)
        old_log_probs_batch: List[torch.Tensor],  # n_agents * (B, T)
    ) -> Dict[str, float]:
        """Perform one CMAPG update on a batch of trajectories."""
        batch_size, seq_len = rewards_batch.shape

        all_actions_flat = self._build_all_actions(actions_batch, batch_size, seq_len)
        states_flat = states_batch.view(batch_size * seq_len, -1)

        # Compute Q-values
        with torch.no_grad():
            q_values = self.target_critic(states_flat, all_actions_flat)
            q_values = q_values.view(batch_size, seq_len)
            psi_values = self.get_counterfactual_value(states_flat, all_actions_flat)
            psi_values = psi_values.view(batch_size, seq_len, self.n_agents)

        # Standard GAE with Q as value function
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

        # Normalize GAE advantages
        gae_adv_flat = gae_adv.view(-1)
        gae_adv_flat = (gae_adv_flat - gae_adv_flat.mean()) / (gae_adv_flat.std() + 1e-8)

        # Counterfactual per-agent advantages: A_i = Q(s,a) - Ψ(s,a^{-i})
        cf_adv = q_values.unsqueeze(-1).expand(-1, -1, self.n_agents) - psi_values  # (B, T, n_agents)
        cf_adv_flat = cf_adv.view(-1, self.n_agents)

        # Compute per-agent advantages based on adv_mode
        if self.adv_mode == "softmax_gae":
            # Original: redistribute GAE by softmax(cf_adv) weights
            # NOTE: destroys sign information — all agents get same-sign advantage
            cf_weights = torch.softmax(cf_adv_flat, dim=-1)
            advantages_flat = gae_adv_flat.unsqueeze(-1) * cf_weights
        elif self.adv_mode == "direct_cf":
            # Direct counterfactual advantages (matching paper theory)
            advantages_flat = cf_adv_flat
        elif self.adv_mode == "normalized_cf":
            # Per-agent normalized counterfactual advantages
            advantages_flat = torch.zeros_like(cf_adv_flat)
            for i in range(self.n_agents):
                adv_i = cf_adv_flat[:, i]
                advantages_flat[:, i] = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
        else:
            raise ValueError(f"Unknown adv_mode: {self.adv_mode}")

        # Diagnostics: log advantage distribution stats
        diag = {}
        with torch.no_grad():
            diag["cf_adv_mean"] = cf_adv_flat.mean().item()
            diag["cf_adv_std"] = cf_adv_flat.std().item()
            diag["cf_adv_max"] = cf_adv_flat.max().item()
            diag["cf_adv_min"] = cf_adv_flat.min().item()
            # Per-agent cf_adv means
            for i in range(self.n_agents):
                diag[f"cf_adv_a{i}"] = cf_adv_flat[:, i].mean().item()
            # Correlation between Q and Ψ (should be high if Ψ copies Q)
            diag["q_psi_corr"] = torch.corrcoef(torch.stack([
                q_values.view(-1), psi_values.mean(dim=-1).view(-1)
            ]))[0, 1].item() if q_values.numel() > 1 else 0.0
            # Sign agreement: do cf_adv and GAE agree on sign?
            cf_sign = cf_adv_flat.mean(dim=-1).sign()
            gae_sign = gae_adv_flat.sign()
            diag["sign_agreement"] = (cf_sign == gae_sign).float().mean().item()
            # softmax entropy (measure of uniformness)
            if self.adv_mode == "softmax_gae":
                sm_entropy = -(cf_weights * torch.log(cf_weights + 1e-8)).sum(dim=-1).mean().item()
                diag["softmax_entropy"] = sm_entropy

        # ---- Policy loss ----
        policy_loss = 0.0
        entropy_total = 0.0
        for i in range(self.n_agents):
            obs_i = obs_batch[i].view(batch_size * seq_len, -1)
            actions_i = actions_batch[i].view(batch_size * seq_len, -1)
            old_log_prob_i = old_log_probs_batch[i].view(batch_size * seq_len)

            dist = self.actors[i](obs_i)
            new_log_prob = dist.log_prob(actions_i.squeeze(-1) if self.discrete_actions else actions_i)
            if not self.discrete_actions:
                new_log_prob = new_log_prob.sum(dim=-1)

            ratio = torch.exp(new_log_prob - old_log_prob_i)
            adv_i = advantages_flat[:, i]

            # PPO clipped objective
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_i
            policy_loss += -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy_total += dist.entropy().mean()

        policy_loss = policy_loss / self.n_agents
        entropy_avg = entropy_total / self.n_agents

        # ---- Q-function loss ----
        q_pred = self.critic(states_flat, all_actions_flat).view(batch_size, seq_len)
        q_loss = F.mse_loss(q_pred, returns)

        # ---- Counterfactual model loss ----
        psi_loss = 0.0
        for i in range(self.n_agents):
            a_dim = self.action_dims[i]
            action_offset_i = sum(self.action_dims[:i])

            # Construct other-agent actions
            other_actions_list = []
            for j in range(self.n_agents):
                if j != i:
                    offset_j = sum(self.action_dims[:j])
                    other_actions_list.append(
                        all_actions_flat[:, offset_j:offset_j + self.action_dims[j]]
                    )

            if other_actions_list:
                other_actions = torch.stack(other_actions_list, dim=1)
            else:
                other_actions = torch.zeros(batch_size * seq_len, 0, 1, device=self.device)

            psi_pred = self.psi_nets[i](states_flat, other_actions)

            # Target: use Q-network to compute counterfactual target
            with torch.no_grad():
                # Generate default actions from mixture policy
                if self.discrete_actions:
                    uniform_probs = torch.ones(
                        batch_size * seq_len, a_dim, device=self.device
                    ) / a_dim

                    obs_i_flat = obs_batch[i].view(batch_size * seq_len, -1)
                    current_dist = self.actors[i](obs_i_flat)

                    # Mixture distribution: (1-α) * uniform + α * current policy
                    mixed_probs = (1 - self.alpha_default) * uniform_probs + \
                                  self.alpha_default * current_dist.probs

                    # Sample default actions from mixture
                    default_actions = torch.multinomial(mixed_probs, 1).float()
                else:
                    obs_i_flat = obs_batch[i].view(batch_size * seq_len, -1)
                    current_dist = self.actors[i](obs_i_flat)
                    default_actions = current_dist.sample()

                # Build full action with default for agent i
                full_default = all_actions_flat.clone()
                if self.discrete_actions:
                    default_onehot = self._one_hot(default_actions, a_dim)
                    full_default[:, action_offset_i:action_offset_i + a_dim] = default_onehot
                else:
                    full_default[:, action_offset_i:action_offset_i + a_dim] = default_actions

                psi_target = self.target_critic(states_flat, full_default).detach()

            psi_loss += F.mse_loss(psi_pred, psi_target.squeeze(-1))
        psi_loss = psi_loss / self.n_agents

        # ---- MI regularization ----
        mi_loss = 0.0
        for i in range(self.n_agents):
            action_offset_i = sum(self.action_dims[:i])
            a_i = all_actions_flat[:, action_offset_i:action_offset_i + self.action_dims[i]]

            # Compute Ψ(s, a^{-i}) for agent i
            other_actions_list = []
            for j in range(self.n_agents):
                if j != i:
                    offset_j = sum(self.action_dims[:j])
                    other_actions_list.append(
                        all_actions_flat[:, offset_j:offset_j + self.action_dims[j]]
                    )
            other_actions = torch.stack(other_actions_list, dim=1) if other_actions_list else \
                torch.zeros(batch_size * seq_len, 0, 1, device=self.device)

            with torch.no_grad():
                psi_i = self.psi_nets[i](states_flat, other_actions).unsqueeze(-1)

            mi = self.club_estimators[i].mi_upper_bound(psi_i, a_i)
            mi_loss += mi
        mi_loss = mi_loss / self.n_agents

        # ---- Total loss ----
        current_entropy_coef = max(0.001, self.entropy_coef * (self.entropy_decay ** self.steps))
        total_loss = (
            policy_loss
            + self.lambda_q * q_loss
            + self.lambda_psi * psi_loss
            + self.lambda_mi * mi_loss
            - current_entropy_coef * entropy_avg
        )

        # ---- Optimization ----
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actors.parameters())
            + list(self.critic.parameters())
            + list(self.psi_nets.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # ---- Update target networks ----
        self._soft_update(self.critic, self.target_critic)
        for i in range(self.n_agents):
            self._soft_update(self.psi_nets[i], self.target_psi_nets[i])

        return {
            "policy_loss": policy_loss.item(),
            "q_loss": q_loss.item(),
            "psi_loss": psi_loss.item(),
            "mi_loss": mi_loss.item(),
            "entropy": entropy_avg.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "total_loss": total_loss.item(),
            "q_psi_corr": diag.get("q_psi_corr", 0.0),
            "sign_agreement": diag.get("sign_agreement", 0.0),
            "cf_adv_mean": diag.get("cf_adv_mean", 0.0),
            "cf_adv_std": diag.get("cf_adv_std", 0.0),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.target_tau * src_param.data + (1 - self.target_tau) * tgt_param.data
            )

    def save(self, path: str):
        torch.save({
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "psi_nets": self.psi_nets.state_dict(),
            "club_estimators": self.club_estimators.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
            "entropy_coef": self.entropy_coef,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])
        self.psi_nets.load_state_dict(ckpt["psi_nets"])
        self.club_estimators.load_state_dict(ckpt["club_estimators"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.entropy_coef = ckpt.get("entropy_coef", self.entropy_coef)
        self.target_critic = deepcopy(self.critic)
        for i in range(self.n_agents):
            self.target_psi_nets[i] = deepcopy(self.psi_nets[i])
