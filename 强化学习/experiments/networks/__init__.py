"""Network modules for CMAPG and baseline algorithms."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CategoricalActor(nn.Module):
    """Discrete action policy network."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_actions: int,
    ):
        super().__init__()
        self.base = MLP(obs_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        x = F.relu(self.base(obs))
        logits = self.logits(x)
        return torch.distributions.Categorical(logits=logits)

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.base(obs))
        return self.logits(x)


class GaussianActor(nn.Module):
    """Continuous action policy network."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.base = MLP(obs_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        x = F.relu(self.base(obs))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)


class CentralizedCritic(nn.Module):
    """Centralized Q-function that takes global state + all actions."""

    def __init__(
        self,
        state_dim: int,
        total_action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.q = MLP(state_dim + total_action_dim, [hidden_dim, hidden_dim], 1)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, actions], dim=-1)
        return self.q(x)


class CounterfactualValueNet(nn.Module):
    """Counterfactual value network Ψ(s, a^{-i}) for agent i.

    Estimates the expected return when agent i's action is marginalized out.
    Uses a permutation-invariant architecture over other agents' actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_agents: int = 20,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # State encoder
        self.state_encoder = MLP(state_dim, [hidden_dim], hidden_dim)

        # Per-agent action encoder (shared)
        self.action_encoder = MLP(action_dim, [hidden_dim], hidden_dim)

        # Aggregation network (processes sum of other agents' encodings)
        self.aggregator = MLP(hidden_dim * 2, [hidden_dim, hidden_dim], 1)

        # Learnable "absence" embedding for the removed agent
        self.absence_emb = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(
        self,
        state: torch.Tensor,          # (B, state_dim)
        other_actions: torch.Tensor,  # (B, n-1, action_dim)
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        n_others = other_actions.shape[1]

        # Encode state
        state_enc = self.state_encoder(state)  # (B, H)

        # Encode other agents' actions and aggregate
        if n_others > 0:
            other_actions_flat = other_actions.view(-1, self.action_dim)
            action_enc = self.action_encoder(other_actions_flat)
            action_enc = action_enc.view(batch_size, n_others, -1)
            action_agg = action_enc.sum(dim=1) / (n_others ** 0.5)
        else:
            action_agg = self.absence_emb.expand(batch_size, -1)

        # Combine and predict value
        combined = torch.cat([state_enc, action_agg], dim=-1)
        value = self.aggregator(combined)
        return value.squeeze(-1)


class QMIXMixingNet(nn.Module):
    """QMIX monotonic mixing network with hypernetworks."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        hidden_dim: int,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = MLP(state_dim, [hidden_dim], embed_dim * n_agents)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = MLP(state_dim, [hidden_dim], embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,  # (B, n_agents)
        state: torch.Tensor,     # (B, state_dim)
    ) -> torch.Tensor:
        batch_size = agent_qs.shape[0]

        # First layer: state-dependent weights (must be positive for monotonicity)
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(batch_size, 1, -1)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # (B, 1, E)
        hidden = F.elu(hidden)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, -1, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class ClubMIEstimator(nn.Module):
    """CLUB estimator for mutual information regularization.

    Estimates MI(A_i; Ψ(s, A^{-i})) to regularize the counterfactual model.
    """

    def __init__(self, action_dim: int, psi_dim: int, hidden_dim: int):
        super().__init__()
        # q_ξ(ψ | a_i): variational approximation of p(ψ | a_i)
        self.q_mu = MLP(action_dim, [hidden_dim, hidden_dim], psi_dim)
        self.q_logvar = MLP(action_dim, [hidden_dim, hidden_dim], psi_dim)

    def forward(
        self, psi: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log q_ξ(ψ | a_i)."""
        mu = self.q_mu(actions)
        logvar = self.q_logvar(actions)
        # Negative log-likelihood under diagonal Gaussian
        var = torch.exp(logvar)
        nll = 0.5 * (torch.log(2 * np.pi * var) + (psi - mu) ** 2 / var)
        return -nll.sum(dim=-1)  # log q(ψ | a)

    def mi_upper_bound(
        self,
        psi: torch.Tensor,      # (B, psi_dim)
        actions: torch.Tensor,  # (B, action_dim)
    ) -> torch.Tensor:
        """Compute CLUB upper bound: E_p(a,ψ)[log q(ψ|a)] - E_p(a)p(ψ)[log q(ψ|a)]."""
        # Joint expectation
        log_q_joint = self.forward(psi, actions)

        # Marginal expectation (shuffle ψ to break dependency)
        psi_shuffled = psi[torch.randperm(psi.shape[0])]
        log_q_marg = self.forward(psi_shuffled, actions)

        mi = log_q_joint.mean() - log_q_marg.mean()
        return mi
