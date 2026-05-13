"""Plotting utilities for SVPG paper.

Generates all figures for the paper:
  - training_curves: SVPG vs baselines with error bars (3 seeds)
  - shapley_diagnostics: Shapley value distribution and entropy
  - ablation_K: Performance vs number of Monte Carlo permutations
  - overview: Conceptual diagram of SVPG
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color scheme
COLORS = {
    "SVPG": "#2196F3",      # Blue
    "MAPPO": "#FF9800",     # Orange
    "CMAPG": "#4CAF50",     # Green
    "COMA": "#9C27B0",      # Purple
}

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def load_seed_metrics(algo: str, env: str, n_agents: int, seeds: list):
    """Load metrics from multiple seeds, return as list of dicts."""
    all_runs = []
    for seed in seeds:
        log_dir = os.path.join(LOG_DIR, f"{algo}_{env}_n{n_agents}_s{seed}")
        metrics_path = os.path.join(log_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                all_runs.append(json.load(f))
    return all_runs


def compute_mean_std_curve(runs: list, key_x: str, key_y: str):
    """Compute mean and std across runs, interpolating to common x grid."""
    if not runs:
        return np.array([]), np.array([]), np.array([])

    # Find common x range
    all_x = []
    for run in runs:
        if key_x in run and key_y in run:
            x_vals = np.array(run[key_x])
            y_vals = np.array(run[key_y])
            if len(x_vals) > 0:
                all_x.append(x_vals)

    if not all_x:
        return np.array([]), np.array([]), np.array([])

    # Common grid: union of all eval points
    common_x = sorted(set(np.concatenate(all_x)))
    if len(common_x) > 100:
        common_x = np.linspace(min(common_x), max(common_x), 50)

    interpolated = []
    for i, run in enumerate(runs):
        if key_x in run and key_y in run:
            x_vals = np.array(run[key_x])
            y_vals = np.array(run[key_y])
            if len(x_vals) >= 2:
                interp = np.interp(common_x, x_vals, y_vals)
                interpolated.append(interp)

    if not interpolated:
        return np.array([]), np.array([]), np.array([])

    mean_y = np.mean(interpolated, axis=0)
    std_y = np.std(interpolated, axis=0)
    return np.array(common_x), mean_y, std_y


def plot_training_curves(env: str = "key_lock", n_agents: int = 5, seeds: list = None,
                          algorithms: list = None, save: bool = True):
    """Plot training curves with error bars (mean ± std across seeds)."""
    if seeds is None:
        seeds = [42, 123, 456]
    if algorithms is None:
        algorithms = ["SVPG", "MAPPO", "CMAPG", "COMA"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: main training curves
    for algo_upper in algorithms:
        algo_lower = algo_upper.lower()
        runs = load_seed_metrics(algo_lower, env, n_agents, seeds)
        x, mean_y, std_y = compute_mean_std_curve(runs, "eval_step", "eval_return_mean")

        if len(x) > 0:
            color = COLORS.get(algo_upper, "#333333")
            ax1.plot(x / 1000, mean_y, color=color, lw=1.8, label=algo_upper)
            ax1.fill_between(x / 1000, mean_y - std_y, mean_y + std_y,
                            color=color, alpha=0.15)

    ax1.set_xlabel("Environment Steps (×1000)")
    ax1.set_ylabel("Episode Return")
    ax1.set_title(f"Training Curves on {env.replace('_',' ').title()}")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    # Right: peak performance bar chart
    peaks = {}
    errors = {}
    for algo_upper in algorithms:
        algo_lower = algo_upper.lower()
        runs = load_seed_metrics(algo_lower, env, n_agents, seeds)
        if not runs:
            continue
        peak_vals = []
        for run in runs:
            if "eval_return_mean" in run and len(run["eval_return_mean"]) > 0:
                peak_vals.append(np.max(run["eval_return_mean"]))
        if peak_vals:
            peaks[algo_upper] = np.mean(peak_vals)
            errors[algo_upper] = np.std(peak_vals)

    if peaks:
        bars = ax2.bar(peaks.keys(), peaks.values(),
                      yerr=errors.values(), capsize=4,
                      color=[COLORS.get(k, "#333") for k in peaks.keys()],
                      edgecolor="white", linewidth=0.5)
        ax2.set_ylabel("Peak Return")
        ax2.set_title("Peak Performance")
        ax2.grid(True, alpha=0.2, axis="y")
        for bar, peak in zip(bars, peaks.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{peak:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"SVPG vs Baselines on {env.replace('_',' ').title()} (N={n_agents}, {len(seeds)} seeds)",
                fontweight="bold")
    fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        prefix = f"training_curves_{env}"
        fig.savefig(os.path.join(FIG_DIR, f"{prefix}.pdf"))
        fig.savefig(os.path.join(FIG_DIR, f"{prefix}.png"))
        print(f"Saved {prefix} to {FIG_DIR}")
    plt.close(fig)


def plot_shapley_diagnostics(env: str = "key_lock", n_agents: int = 5, seeds: list = None,
                              save: bool = True):
    """Plot Shapley value diagnostics from SVPG training logs."""
    if seeds is None:
        seeds = [42, 123, 456]

    # Load SVPG diagnostic data
    algo = "svpg"
    all_runs = load_seed_metrics(algo, env, n_agents, seeds)
    if not all_runs:
        print("No SVPG metrics found for Shapley diagnostics")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    # 1. Shapley mean over time
    ax = axes[0, 0]
    for seed_idx, run in enumerate(all_runs):
        if "shapley_mean" in run and "eval_step" in run:
            x = np.arange(len(run["shapley_mean"]))
            ax.plot(x, run["shapley_mean"], lw=0.5, alpha=0.7,
                   label=f"seed {seeds[seed_idx]}")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Mean Shapley Value")
    ax.set_title("Shapley Value Mean")
    ax.grid(True, alpha=0.2)

    # 2. Shapley std over time
    ax = axes[0, 1]
    for seed_idx, run in enumerate(all_runs):
        if "shapley_std" in run:
            x = np.arange(len(run["shapley_std"]))
            ax.plot(x, run["shapley_std"], lw=0.5, alpha=0.7,
                   label=f"seed {seeds[seed_idx]}")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Shapley Value Std")
    ax.set_title("Shapley Value Standard Deviation")
    ax.grid(True, alpha=0.2)

    # 3. Shapley entropy (normalized) over time
    ax = axes[0, 2]
    for seed_idx, run in enumerate(all_runs):
        if "shapley_entropy_norm" in run:
            x = np.arange(len(run["shapley_entropy_norm"]))
            ax.plot(x, run["shapley_entropy_norm"], lw=0.8, alpha=0.7,
                   label=f"seed {seeds[seed_idx]}")
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=0.5, label="Uniform (entropy=1)")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("Shapley Distribution Entropy\n(1.0 = uniform, <1.0 = differentiated)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)

    # 4. Per-agent Shapley means (bar chart from last 10% of training)
    ax = axes[1, 0]
    agent_means = defaultdict(list)
    for run in all_runs:
        n_updates = len(run.get("shapley_mean", []))
        if n_updates < 10:
            continue
        last_start = max(0, n_updates - n_updates // 10)
        for i in range(n_agents):
            key = f"shapley_a{i}"
            if key in run:
                agent_means[i].append(np.mean(run[key][last_start:]))

    if agent_means:
        agent_labels = [f"Agent {i+1}" for i in range(n_agents)]
        agent_avg = [np.mean(agent_means.get(i, [0])) for i in range(n_agents)]
        agent_std = [np.std(agent_means.get(i, [0])) for i in range(n_agents)]
        colors_bar = plt.cm.viridis(np.linspace(0.1, 0.9, n_agents))
        ax.bar(agent_labels, agent_avg, yerr=agent_std, capsize=4,
              color=colors_bar, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Mean Shapley Value")
        ax.set_title("Per-Agent Shapley Values (Late Training)")
        ax.grid(True, alpha=0.2, axis="y")

    # 5. Q_pred_mean over time
    ax = axes[1, 1]
    for seed_idx, run in enumerate(all_runs):
        if "q_pred_mean" in run:
            x = np.arange(len(run["q_pred_mean"]))
            ax.plot(x, run["q_pred_mean"], lw=0.5, alpha=0.7,
                   label=f"seed {seeds[seed_idx]}")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Mean Q-value")
    ax.set_title("Q-function Prediction")
    ax.grid(True, alpha=0.2)

    # 6. Policy loss and value loss
    ax = axes[1, 2]
    for seed_idx, run in enumerate(all_runs):
        if "policy_loss" in run and "value_loss" in run:
            x = np.arange(len(run["policy_loss"]))
            ax.plot(x, run["policy_loss"], lw=0.5, alpha=0.5, color="blue")
            ax.plot(x, run["value_loss"], lw=0.5, alpha=0.5, color="orange")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, label="Policy Loss"),
        Line2D([0], [0], color="orange", lw=1, label="Value Loss"),
    ]
    ax.legend(handles=legend_elements, fontsize=6)
    ax.set_xlabel("Updates")
    ax.set_ylabel("Loss")
    ax.set_title("Policy & Value Loss")
    ax.grid(True, alpha=0.2)

    fig.suptitle("SVPG Diagnostic Analysis: Shapley Value Dynamics",
                fontweight="bold")
    fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "shapley_diagnostics.pdf"))
        fig.savefig(os.path.join(FIG_DIR, "shapley_diagnostics.png"))
        print(f"Saved shapley_diagnostics to {FIG_DIR}")
    plt.close(fig)


def plot_ablation_K(save: bool = True):
    """Plot SVPG performance vs number of Monte Carlo permutations K."""
    K_values = [1, 5, 10, 20, 50]
    env = "key_lock"
    n_agents = 5
    seeds = [42, 123, 456]

    K_peaks = []
    K_stds = []

    for K in K_values:
        seed_bests = []
        for seed in seeds:
            # K=20 uses main experiment (no _K suffix)
            if K == 20:
                log_dir = os.path.join(LOG_DIR, f"svpg_{env}_n{n_agents}_s{seed}")
            else:
                log_dir = os.path.join(LOG_DIR, f"svpg_{env}_n{n_agents}_s{seed}_K{K}")

            metrics_path = os.path.join(log_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    m = json.load(f)
                if "eval_return_mean" in m and len(m["eval_return_mean"]) > 0:
                    seed_bests.append(np.max(m["eval_return_mean"]))

        if seed_bests:
            K_peaks.append(np.mean(seed_bests))
            K_stds.append(np.std(seed_bests))
        else:
            K_peaks.append(0)
            K_stds.append(0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(K_values, K_peaks, yerr=K_stds, marker="o", capsize=4,
               color=COLORS["SVPG"], lw=1.5, markersize=6)
    ax.set_xlabel("K (Monte Carlo Permutations)")
    ax.set_ylabel("Peak Return")
    ax.set_title("Effect of K on SVPG Performance")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "ablation_K.pdf"))
        fig.savefig(os.path.join(FIG_DIR, "ablation_K.png"))
        print(f"Saved ablation_K to {FIG_DIR}")
    plt.close(fig)


def plot_final_results_table(env: str = "key_lock", n_agents: int = 5,
                              seeds: list = None, save: bool = True):
    """Generate a summary results table figure."""
    if seeds is None:
        seeds = [42, 123, 456]

    algorithms = ["SVPG", "MAPPO", "CMAPG", "COMA"]
    headers = ["Algorithm", "Peak Return", "Final Return", "Convergence\n(Steps)", "CA Accuracy"]

    data = []
    for algo_upper in algorithms:
        algo_lower = algo_upper.lower()
        runs = load_seed_metrics(algo_lower, env, n_agents, seeds)
        if not runs:
            continue

        peak_vals = []
        final_vals = []
        for run in runs:
            if "eval_return_mean" in run and len(run["eval_return_mean"]) > 0:
                vals = np.array(run["eval_return_mean"])
                peak_vals.append(np.max(vals))
                final_vals.append(vals[-1])

        # Convergence: step at which 95% of peak is reached
        conv_steps = []
        for run in runs:
            if "eval_return_mean" in run and "eval_step" in run:
                vals = np.array(run["eval_return_mean"])
                steps = np.array(run["eval_step"])
                if len(vals) > 0 and np.max(vals) > 0:
                    threshold = 0.95 * np.max(vals)
                    idx = np.argmax(vals >= threshold)
                    conv_steps.append(steps[idx] if idx < len(steps) else steps[-1])

        ca_acc = []
        for run in runs:
            if "eval_ca_accuracy" in run and len(run["eval_ca_accuracy"]) > 0:
                ca_acc.append(np.max(run["eval_ca_accuracy"]))

        data.append([
            algo_upper,
            f"{np.mean(peak_vals):.2f} ± {np.std(peak_vals):.2f}" if peak_vals else "N/A",
            f"{np.mean(final_vals):.2f} ± {np.std(final_vals):.2f}" if final_vals else "N/A",
            f"{int(np.mean(conv_steps))}" if conv_steps else "N/A",
            f"{np.mean(ca_acc):.3f}" if ca_acc else "N/A",
        ])

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=headers, cellLoc="center",
                    loc="center", colWidths=[0.12, 0.22, 0.22, 0.18, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color SVPG row
    for j in range(len(headers)):
        table[(1, j)].set_facecolor("#e3f2fd")

    ax.set_title(f"Key-Lock Results (N={n_agents}, {len(seeds)} seeds)",
                fontweight="bold", fontsize=11)

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "results_table.pdf"))
        fig.savefig(os.path.join(FIG_DIR, "results_table.png"))
        print(f"Saved results_table to {FIG_DIR}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SVPG Paper Plotting")
    parser.add_argument("--all", action="store_true", default=False,
                       help="Generate all figures")
    parser.add_argument("--training", action="store_true", help="Training curves")
    parser.add_argument("--shapley", action="store_true", help="Shapley diagnostics")
    parser.add_argument("--ablation", action="store_true", help="K ablation")
    parser.add_argument("--table", action="store_true", help="Results table")
    parser.add_argument("--env", type=str, default="key_lock")
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    run_all = args.all or not any([args.training, args.shapley, args.ablation, args.table])

    seeds = list(args.seeds)

    if run_all or args.training:
        print("Plotting training curves...")
        plot_training_curves(args.env, args.n_agents, seeds)
        plot_final_results_table(args.env, args.n_agents, seeds)

    if run_all or args.shapley:
        print("Plotting Shapley diagnostics...")
        plot_shapley_diagnostics(args.env, args.n_agents, seeds)

    if run_all or args.ablation:
        print("Plotting K ablation...")
        plot_ablation_K()

    print("All plots generated.")


if __name__ == "__main__":
    main()
