"""Generate real counterfactual value heatmaps from a trained CMAPG model.

Replaces the fake simulated heatmaps (plot.py:plot_visualization) with actual
Ψ(s, a^{-i}) values computed from a trained network.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.algorithms.cmapg import CMAPG
from experiments.envs import SequentialKeyLockEnv

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_trained_cmapg(checkpoint_path: str, env: SequentialKeyLockEnv) -> CMAPG:
    """Load a trained CMAPG model from checkpoint."""
    sample_obs = env.observation_space[f"agent_0"]
    obs_dim = sample_obs.shape[0] if hasattr(sample_obs, "shape") else sample_obs.n
    n_actions = env.action_space[f"agent_0"].n
    state_dim = obs_dim * env.n_agents

    cmapg = CMAPG(
        n_agents=env.n_agents,
        obs_dims=[obs_dim] * env.n_agents,
        state_dim=state_dim,
        action_dims=[n_actions] * env.n_agents,
        discrete_actions=True,
        hidden_dim=128,
        device="cpu",
    )
    cmapg.to("cpu")
    cmapg.load(checkpoint_path)
    cmapg.actors.eval()
    cmapg.psi_nets.eval()
    cmapg.critic.eval()
    return cmapg


def compute_psi_heatmap(
    cmapg: CMAPG,
    agent_idx: int,
    base_state: np.ndarray,
    base_actions: np.ndarray,
    vary_agent: int,
    vary_position: float,
    n_agents: int,
    n_actions: int,
    obs_dim: int,
) -> np.ndarray:
    """Compute Ψ for agent_idx varying one other agent's action."""
    if vary_agent == agent_idx:
        return np.zeros(n_actions)

    psi_values = np.zeros(n_actions)
    for a_val in range(n_actions):
        modified_obs_list = []
        for i in range(n_agents):
            obs = base_state[i * obs_dim:(i + 1) * obs_dim].copy()
            if i == vary_agent:
                obs[0] = vary_position
            modified_obs_list.append(obs)

        state = np.concatenate(modified_obs_list)
        state_t = torch.FloatTensor(state).unsqueeze(0)

        all_actions_onehot = []
        for i in range(n_agents):
            onehot = np.zeros(cmapg.action_dims[i])
            onehot[int(base_actions[i]) if i != vary_agent else a_val] = 1.0
            all_actions_onehot.append(onehot)
        actions_t = torch.FloatTensor(np.concatenate(all_actions_onehot)).unsqueeze(0)

        with torch.no_grad():
            psi = cmapg.get_counterfactual_value(state_t, actions_t)
            psi_values[a_val] = psi[0, agent_idx].item()

    return psi_values


def generate_psi_grid(
    cmapg: CMAPG,
    agent_idx: int,
    other_agent: int,
    state_template: np.ndarray,
    action_template: np.ndarray,
    grid_size: int = 20,
    x_range: tuple = (-5, 5),
    y_range: tuple = (-5, 5),
    obs_dim: int = 2,
) -> np.ndarray:
    """Generate 2D grid of Ψ values as a function of other_agent's position."""
    n_agents = cmapg.n_agents
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    psi_grid = np.zeros((grid_size, grid_size))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            modified_state = state_template.copy()
            modified_state[other_agent * obs_dim] = x
            modified_state[other_agent * obs_dim + 1] = y

            state_t = torch.FloatTensor(modified_state).unsqueeze(0)

            # Build one-hot actions
            all_actions = []
            for ag in range(n_agents):
                onehot = np.zeros(cmapg.action_dims[ag])
                onehot[int(action_template[ag])] = 1.0
                all_actions.append(onehot)
            actions_t = torch.FloatTensor(np.concatenate(all_actions)).unsqueeze(0)

            with torch.no_grad():
                psi = cmapg.get_counterfactual_value(state_t, actions_t)
                psi_grid[i, j] = psi[0, agent_idx].item()

    return xs, ys, psi_grid


def plot_real_psi_heatmaps(cmapg: CMAPG, env: SequentialKeyLockEnv):
    """Generate and save real Ψ heatmaps for all agent pairs."""
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n_agents = cmapg.n_agents

    # Get a sample state and actions from the environment
    obs, _ = env.reset()
    obs_list = [obs[f"agent_{i}"] for i in range(n_agents)]
    state = np.concatenate(obs_list)
    obs_dim = obs_list[0].shape[0]

    # Get actions from the trained policy
    actions, _ = cmapg.get_actions(obs_list, deterministic=True)
    all_actions = np.array([int(a) for a in actions])

    grid_size = 30
    # Use agent positions range from env
    x_range = (-env.line_length / 2, env.line_length / 2) if hasattr(env, 'line_length') else (-5, 5)
    y_offsets = np.linspace(-2, 2, grid_size)  # vary y position as proxy

    # For 1D Key-Lock, create 1D heatmaps: vary each other agent's position
    n_others = n_agents - 1
    others = [i for i in range(n_agents) if i != 0]

    fig, axes = plt.subplots(1, min(3, n_others), figsize=(9.0, 3.2))
    if n_others == 1:
        axes = [axes]

    for ax_idx, other_agent in enumerate(others[:3]):
        ax = axes[ax_idx]
        xs = np.linspace(x_range[0], x_range[1], grid_size)

        # For each possible action of the target agent, compute Ψ as f(other_agent_pos)
        n_actions_target = cmapg.action_dims[0]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_actions_target))

        for a_val in range(n_actions_target):
            psi_curve = np.zeros(grid_size)
            for k, x_pos in enumerate(xs):
                # Modify state: move other_agent to x_pos
                mod_state = state.copy()
                mod_state[other_agent * obs_dim] = x_pos

                # Build action tensor with varied action for agent 0
                mod_actions = all_actions.copy()
                mod_actions[0] = a_val

                all_actions_onehot = []
                for ag in range(n_agents):
                    onehot = np.zeros(cmapg.action_dims[ag])
                    onehot[int(mod_actions[ag])] = 1.0
                    all_actions_onehot.append(onehot)

                state_t = torch.FloatTensor(mod_state).unsqueeze(0)
                actions_t = torch.FloatTensor(np.concatenate(all_actions_onehot)).unsqueeze(0)

                with torch.no_grad():
                    psi = cmapg.get_counterfactual_value(state_t, actions_t)
                    psi_curve[k] = psi[0, 0].item()

            action_names = ["Left", "Right", "Stay", "Press"]
            ax.plot(xs, psi_curve, color=colors[a_val], lw=1.5,
                   label=f"$a_0$={action_names[a_val] if a_val < len(action_names) else a_val}")

        ax.set_xlabel(f"Agent {other_agent+1} position")
        ax.set_ylabel(r"$\Psi_1(s, \mathbf{a}^{-1})$")
        ax.set_title(f"Ψ₁ vs Agent {other_agent+1} Position")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Real Learned Counterfactual Values (Trained CMAPG on Key-Lock)",
                fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "real_psi_visualization.pdf"))
    fig.savefig(os.path.join(fig_dir, "real_psi_visualization.png"))
    plt.close(fig)
    print(f"Saved real Ψ visualization to {fig_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained CMAPG checkpoint (.pt file)")
    parser.add_argument("--n_agents", type=int, default=5)
    args = parser.parse_args()

    env = SequentialKeyLockEnv(n_agents=args.n_agents, max_steps=200, seed=42)
    cmapg = load_trained_cmapg(args.checkpoint, env)
    plot_real_psi_heatmaps(cmapg, env)
    env.close()


if __name__ == "__main__":
    main()
