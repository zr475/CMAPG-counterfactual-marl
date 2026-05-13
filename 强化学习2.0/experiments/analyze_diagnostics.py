"""SVPG diagnostic analysis: Shapley value distribution vs ground-truth credit.

Evaluates a trained SVPG model to determine whether Shapley values correctly
identify which agents contribute more to the team reward. Generates diagnostic
plots comparing Shapley values with actual agent contributions.
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

from experiments.algorithms.svpg import SVPG
from experiments.envs import SequentialKeyLockEnv

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_trained_svpg(checkpoint_path: str, env: SequentialKeyLockEnv) -> SVPG:
    sample_obs = env.observation_space["agent_0"]
    obs_dim = sample_obs.shape[0] if hasattr(sample_obs, "shape") else sample_obs.n
    n_actions = env.action_space["agent_0"].n
    state_dim = obs_dim * env.n_agents

    svpg = SVPG(
        n_agents=env.n_agents,
        obs_dims=[obs_dim] * env.n_agents,
        state_dim=state_dim,
        action_dims=[n_actions] * env.n_agents,
        discrete_actions=True,
        hidden_dim=128,
        K=20,
        device="cpu",
    )
    svpg.to("cpu")
    svpg.load(checkpoint_path)
    for actor in svpg.actors:
        actor.eval()
    svpg.critic.eval()
    return svpg


def collect_diagnostic_data(svpg: SVPG, env: SequentialKeyLockEnv, n_steps: int = 200):
    """Collect Shapley values and ground-truth credit over an episode."""
    obs, _ = env.reset()
    obs_list = [obs[f"agent_{i}"] for i in range(env.n_agents)]

    shapley_history = []
    gt_credit_history = []
    actions_history = []
    q_values_history = []
    rewards_history = []

    for step in range(n_steps):
        actions, _ = svpg.get_actions(obs_list, deterministic=False)
        action_dict = {f"agent_{i}": actions[i] for i in range(env.n_agents)}
        next_obs, reward, terminated, truncated, info = env.step(action_dict)

        # Compute Shapley values
        state = np.concatenate(obs_list)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        actions_t = svpg._actions_to_onehot(np.array(actions)).unsqueeze(0)

        with torch.no_grad():
            shapley = svpg.compute_shapley_values(state_t, actions_t)
            q_val = svpg.target_critic(
                state_t,
                torch.cat([svpg._one_hot(
                    torch.tensor([[a]], dtype=torch.float32), svpg.action_dims[i]
                ) for i, a in enumerate(actions)], dim=-1)
            )

        shapley_history.append(shapley.squeeze(0).cpu().numpy())
        actions_history.append(actions)
        q_values_history.append(q_val.item())
        rewards_history.append(reward)

        if hasattr(env, "get_step_credit"):
            gt_credit_history.append(env.get_step_credit())

        obs_list = [next_obs[f"agent_{i}"] for i in range(env.n_agents)]

        if terminated or truncated:
            obs, _ = env.reset()
            obs_list = [obs[f"agent_{i}"] for i in range(env.n_agents)]

    return {
        "shapley": np.array(shapley_history),
        "gt_credit": np.array(gt_credit_history) if gt_credit_history else None,
        "actions": np.array(actions_history),
        "q_values": np.array(q_values_history),
        "rewards": np.array(rewards_history),
    }


def plot_diagnostics(data: dict, n_agents: int, save: bool = True):
    """Generate diagnostic plots from collected data."""
    shapley = data["shapley"]          # (T, n_agents)
    gt = data["gt_credit"]             # (T, n_agents) or None
    actions = data["actions"]          # (T, n_agents)
    q_vals = data["q_values"]          # (T,)
    rewards = data["rewards"]          # (T,)
    T = shapley.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # 1. Shapley values over time
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    for i in range(n_agents):
        ax.plot(shapley[:, i], color=colors[i], lw=0.8, label=f"Agent {i+1}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Shapley Value φ_i")
    ax.set_title("Shapley Values Over Time")
    ax.legend(fontsize=7, ncol=n_agents)
    ax.grid(True, alpha=0.2)

    # 2. Mean Shapley distribution
    ax = axes[0, 1]
    mean_shapley = shapley.mean(axis=0)
    std_shapley = shapley.std(axis=0)
    bars = ax.bar(range(n_agents), mean_shapley, yerr=std_shapley, capsize=4,
                 color=colors)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Shapley Value")
    ax.set_title("Mean Shapley Distribution")
    ax.axhline(y=0, color="gray", linestyle="--", lw=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    # 3. Shapley vs Ground Truth scatter
    ax = axes[0, 2]
    if gt is not None:
        for i in range(n_agents):
            ax.scatter(shapley[:, i], gt[:, i], alpha=0.5, s=8, color=colors[i],
                      label=f"Agent {i+1}")
        # Compute correlation
        shapley_flat = shapley.flatten()
        gt_flat = gt.flatten()
        mask = ~(np.isnan(shapley_flat) | np.isnan(gt_flat))
        if mask.sum() > 2:
            corr = np.corrcoef(shapley_flat[mask], gt_flat[mask])[0, 1]
            ax.text(0.05, 0.95, f"Corr = {corr:.3f}", transform=ax.transAxes,
                   fontsize=9, verticalalignment="top")
    ax.set_xlabel("Shapley Value φ_i")
    ax.set_ylabel("Ground Truth Credit")
    ax.set_title("Shapley vs Ground Truth")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)

    # 4. Q-values over time
    ax = axes[1, 0]
    ax.plot(q_vals, color="blue", lw=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Q(s, a)")
    ax.set_title("Q-values Over Time")
    ax.grid(True, alpha=0.2)

    # 5. Rewards over time
    ax = axes[1, 1]
    ax.plot(rewards, color="green", lw=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    ax.grid(True, alpha=0.2)

    # 6. Shapley entropy over time
    ax = axes[1, 2]
    shapley_centered = shapley - shapley.mean(axis=-1, keepdims=True)
    shapley_softmax = np.exp(shapley_centered) / np.exp(shapley_centered).sum(axis=-1, keepdims=True)
    entropy = -(shapley_softmax * np.log(shapley_softmax + 1e-8)).sum(axis=-1)
    max_entropy = np.log(n_agents)
    ax.plot(entropy, color="purple", lw=0.8)
    ax.axhline(y=max_entropy, color="gray", linestyle="--", lw=0.5, alpha=0.5,
              label=f"Max entropy ({max_entropy:.2f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Shapley Entropy")
    ax.set_title("Shapley Distribution Entropy\n(lower = more differentiated)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    fig.suptitle("SVPG Diagnostic Analysis: Shapley Value Credit Assignment",
                fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "diagnostics.pdf"))
        fig.savefig(os.path.join(FIG_DIR, "diagnostics.png"))
        print(f"Saved diagnostics to {FIG_DIR}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SVPG Diagnostic Analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained SVPG checkpoint (.pt file)")
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=200)
    args = parser.parse_args()

    env = SequentialKeyLockEnv(n_agents=args.n_agents, max_steps=200, seed=42)
    svpg = load_trained_svpg(args.checkpoint, env)

    print(f"Collecting diagnostic data ({args.n_steps} steps)...")
    data = collect_diagnostic_data(svpg, env, args.n_steps)

    print(f"Shapley values: mean={data['shapley'].mean():.4f}, std={data['shapley'].std():.4f}")
    for i in range(args.n_agents):
        print(f"  Agent {i+1}: mean φ = {data['shapley'][:, i].mean():.4f} ± {data['shapley'][:, i].std():.4f}")

    if data["gt_credit"] is not None:
        shapley_flat = data["shapley"].flatten()
        gt_flat = data["gt_credit"].flatten()
        corr = np.corrcoef(shapley_flat, gt_flat)[0, 1]
        print(f"Shapley-GT correlation: {corr:.4f}")

    plot_diagnostics(data, args.n_agents)
    env.close()


if __name__ == "__main__":
    main()
