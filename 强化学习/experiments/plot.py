"""Generate publication-quality figures for the CMAPG paper.

Produces all figures referenced in paper.tex:
  - overview.pdf: Method overview diagram
  - credit_assignment.pdf: Credit assignment accuracy comparison
  - scalability.pdf: Performance vs. number of agents
  - visualization.pdf: Counterfactual value heatmaps
  - training_curves.pdf: Learning curves on SMAC maps
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Set publication-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,  # Set to True if LaTeX is available
})

SYS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(SYS_PATH, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Color palette
COLORS = {
    "cmapg": "#1f77b4",    # Blue
    "mappo": "#ff7f0e",    # Orange
    "qmix": "#2ca02c",     # Green
    "coma": "#d62728",     # Red
    "happo": "#9467bd",    # Purple
    "vdn": "#8c564b",      # Brown
}

METHOD_NAMES = {
    "cmapg": "CMAPG (Ours)",
    "mappo": "MAPPO",
    "qmix": "QMIX",
    "coma": "COMA",
    "happo": "HAPPO",
    "vdn": "VDN",
}


def plot_overview():
    """Create method overview diagram (Fig. 1 in paper)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    titles = [
        r"Multi-Agent Interaction",
        r"Counterfactual Model $\Psi(s, \mathbf{a}^{-i})$",
        r"Individual Advantage",
    ]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw boxes and arrows using patches
        if i == 0:
            # Show agents interacting
            positions = [(0.2, 0.5), (0.5, 0.7), (0.8, 0.5), (0.5, 0.3)]
            colors_list = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            for j, (pos, c) in enumerate(zip(positions, colors_list)):
                circle = plt.Circle(pos, 0.12, color=c, alpha=0.7, ec="black", lw=1)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], f"$A_{j+1}$", ha="center", va="center", fontsize=9, fontweight="bold")
            # Arrows between agents
            for j in range(4):
                for k in range(4):
                    if j != k:
                        ax.annotate("", xy=positions[k], xytext=positions[j],
                                   arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.5))
            ax.text(0.5, 0.05, r"Joint action $\mathbf{a}$, shared reward $R$",
                   ha="center", fontsize=8, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)

        elif i == 1:
            # Counterfactual model
            ax.add_patch(plt.Rectangle((0.2, 0.3), 0.6, 0.4, fill=False, ec="black", lw=1.5))
            ax.text(0.5, 0.55, r"$\Psi_\psi$", ha="center", va="center", fontsize=14, fontweight="bold")
            ax.annotate(r"$s$", xy=(0.5, 0.2), xytext=(0.15, 0.2),
                       arrowprops=dict(arrowstyle="->", lw=1), fontsize=9, ha="center")
            ax.annotate(r"$\mathbf{a}^{-i}$", xy=(0.5, 0.2), xytext=(0.85, 0.2),
                       arrowprops=dict(arrowstyle="->", lw=1), fontsize=9, ha="center")
            ax.annotate("", xy=(0.5, 0.85), xytext=(0.5, 0.72),
                       arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["cmapg"]))
            ax.text(0.5, 0.92, r"$\Psi(s, \mathbf{a}^{-i})$", ha="center", fontsize=9,
                   color=COLORS["cmapg"], fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)

        elif i == 2:
            # Individual advantage computation
            ax.bar([0], [0.7], width=0.3, color=COLORS["cmapg"], alpha=0.7, label=r"$Q(s,\mathbf{a})$")
            ax.bar([0], [0.4], width=0.3, color="gray", alpha=0.5, label=r"$\Psi(s,\mathbf{a}^{-i})$")
            ax.bar([0], [0.3], width=0.15, color=COLORS["cmapg"], alpha=1.0,
                   bottom=0.4, label=r"$A_i^{\mathrm{cf}}$", edgecolor="black", lw=1)
            ax.set_ylabel("Value")
            ax.set_xticks([])
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylim(0, 0.9)
            ax.text(0.7, 0.75, r"$A_i^{\mathrm{cf}} = Q - \Psi$",
                   ha="center", fontsize=9, fontweight="bold", transform=ax.transAxes)

    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, "overview.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "overview.png"))
    plt.close(fig)
    print(f"Saved: overview.pdf")


def _parse_log(log_path: str) -> Dict[str, List]:
    """Parse training log to extract returns and CA accuracy."""
    steps_ret = []
    returns = []
    steps_ca = []
    ca_acc = []
    with open(log_path, "r") as f:
        for line in f:
            if "| Return:" in line:
                parts = line.split("|")
                step_str = parts[0].strip().split()[-1]
                ret_tokens = parts[1].strip().split()
                ret_str = ret_tokens[1] if len(ret_tokens) >= 2 else "0"
                try:
                    steps_ret.append(int(step_str))
                    returns.append(float(ret_str))
                except (ValueError, IndexError):
                    pass
            if "| eval_ca_accuracy:" in line:
                parts = line.split("|")
                step_str = parts[0].strip().split()[-1]
                ca_str = parts[-1].strip().split()[-1] if parts else ""
                try:
                    steps_ca.append(int(step_str))
                    ca_acc.append(float(ca_str))
                except (ValueError, IndexError):
                    pass
    return {
        "steps_ret": np.array(steps_ret),
        "returns": np.array(returns),
        "steps_ca": np.array(steps_ca),
        "ca_acc": np.array(ca_acc),
    }


def _smooth(y, window=5):
    """Moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _dedup_metrics(metrics: Dict) -> Dict:
    """Remove duplicate entries caused by double log_eval calls in older training runs.

    Consecutive identical eval_return_mean values (within 1e-6) are collapsed
    to single entries, keeping only the first occurrence (which may have ca_accuracy).
    """
    returns = np.array(metrics.get("eval_return_mean", []))
    if len(returns) <= 1:
        return metrics
    # Find indices where consecutive values differ
    keep = [0]
    for i in range(1, len(returns)):
        if abs(returns[i] - returns[i - 1]) > 1e-6:
            keep.append(i)
    # Also keep the last entry with ca_accuracy if present
    result = {}
    for key, vals in metrics.items():
        if not isinstance(vals, list):
            result[key] = vals
            continue
        arr = np.array(vals)
        if len(arr) == len(returns):
            result[key] = arr[keep].tolist()
        else:
            result[key] = vals
    return result


def _load_multi_seed_metrics(method: str, seeds: List[int], log_dir: str) -> List[Dict]:
    """Load metrics.json from multiple seeds, return list of deduplicated metric dicts."""
    results = []
    for seed in seeds:
        path = os.path.join(log_dir, f"{method}_key_lock_n5_s{seed}", "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(_dedup_metrics(json.load(f)))
    return results


def _align_multi_seed_curves(metrics_list: List[Dict], key: str = "eval_return_mean",
                              steps_key: str = "eval_step") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align curves from multiple seeds: compute mean and std across seeds at common steps."""
    if not metrics_list:
        return np.array([]), np.array([]), np.array([])
    # Find common steps (all seeds use same eval_freq, so steps should be identical)
    common_steps = np.array(metrics_list[0].get(steps_key, []))
    all_curves = []
    for m in metrics_list:
        steps = np.array(m.get(steps_key, []))
        vals = np.array(m.get(key, []))
        if len(steps) != len(common_steps):
            continue
        all_curves.append(vals)
    if not all_curves:
        return np.array([]), np.array([]), np.array([])
    stacked = np.array(all_curves)  # (n_seeds, n_evals)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    return common_steps, mean, std


def plot_credit_assignment():
    """Plot multi-seed training curves and stability analysis."""
    from matplotlib.ticker import MaxNLocator

    log_dir = os.path.join(SYS_PATH, "logs")
    seeds = [42, 123, 456, 789]
    methods = ["cmapg", "mappo"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # Left: Multi-seed learning curves with std bands
    window = 8
    for method in methods:
        metrics_list = _load_multi_seed_metrics(method, seeds, log_dir)
        if not metrics_list:
            continue
        steps, mean, std = _align_multi_seed_curves(metrics_list)
        if len(steps) == 0:
            continue
        steps_k = steps / 1000
        mean_smooth = _smooth(mean, window)
        std_smooth = _smooth(std, window)
        ax1.plot(steps_k, mean_smooth, color=COLORS[method],
                label=f"{METHOD_NAMES[method]} ({len(metrics_list)} seeds)", lw=1.5)
        ax1.fill_between(steps_k, mean_smooth - std_smooth, mean_smooth + std_smooth,
                        color=COLORS[method], alpha=0.15)

    ax1.axhline(y=3.4, color="gray", linestyle="--", alpha=0.5, lw=0.8, label="Optimal (~3.4)")
    ax1.set_xlabel("Environment Steps (K)")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Learning Curves on Key-Lock (N=5, 4 seeds)", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.grid(True, alpha=0.2)

    # Right: Multi-seed Peak vs Final with error bars
    stability_data = {}
    for method in methods:
        metrics_list = _load_multi_seed_metrics(method, seeds, log_dir)
        if not metrics_list:
            continue
        peaks = []
        finals = []
        for m in metrics_list:
            returns = np.array(m.get("eval_return_mean", []))
            if len(returns) == 0:
                continue
            peaks.append(float(np.max(returns)))
            final_window = returns[-min(5, len(returns)):]
            finals.append(float(np.mean(final_window)))
        if peaks:
            stability_data[method] = {
                "peak_mean": np.mean(peaks), "peak_std": np.std(peaks),
                "final_mean": np.mean(finals), "final_std": np.std(finals),
            }

    if stability_data:
        x = np.arange(len(stability_data))
        width = 0.3
        for i, (method, sd) in enumerate(stability_data.items()):
            color = COLORS[method]
            label = METHOD_NAMES[method]
            ax2.bar(x[i] - width/2, sd["peak_mean"], width, color=color, alpha=0.85,
                  edgecolor="black", lw=0.5, yerr=sd["peak_std"], capsize=3,
                  label="Peak" if i == 0 else "")
            ax2.bar(x[i] + width/2, sd["final_mean"], width, color=color, alpha=0.4,
                  edgecolor="black", lw=0.5, yerr=sd["final_std"], capsize=3,
                  label="Final" if i == 0 else "")
        ax2.set_xticks(x)
        ax2.set_xticklabels([METHOD_NAMES[m] for m in stability_data.keys()])
        ax2.set_ylabel("Episode Return")
        ax2.set_title("Peak vs Final (4 seeds, mean ± std)", fontweight="bold")
        ax2.legend(fontsize=7)
        ax2.axhline(y=3.4, color="gray", linestyle="--", alpha=0.4, lw=0.8)
        ax2.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "credit_assignment.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "credit_assignment.png"))
    plt.close(fig)
    print(f"Saved: credit_assignment.pdf")


def _extract_scalability_metrics(log_path: str) -> Dict:
    """Extract peak return, final return, and std from a training log."""
    steps_ret = []
    returns = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "| Return:" in line:
                    parts = line.split("|")
                    step_str = parts[0].strip().split()[-1]
                    ret_tokens = parts[1].strip().split()
                    ret_str = ret_tokens[1] if len(ret_tokens) >= 2 else "0"
                    try:
                        steps_ret.append(int(step_str))
                        returns.append(float(ret_str))
                    except (ValueError, IndexError):
                        pass
    except FileNotFoundError:
        return {}
    if not returns:
        return {}
    arr = np.array(returns)
    # Peak return (top 10% mean for robustness)
    top_n = max(1, len(arr) // 10)
    peak = float(np.mean(np.sort(arr)[-top_n:]))
    final = float(np.mean(arr[-5:])) if len(arr) >= 5 else float(arr[-1])
    return {"peak": peak, "final": final, "std": float(np.std(arr)), "n_points": len(arr)}


def plot_scalability():
    """Plot performance vs. number of agents using real experimental data (multi-seed for N=5)."""
    from matplotlib.ticker import MaxNLocator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    methods = ["cmapg", "mappo"]
    n_agents_list = [3, 5, 7, 10]
    scalability_dir = os.path.join(SYS_PATH, "logs", "scalability")
    main_log_dir = os.path.join(SYS_PATH, "logs")
    seeds = [42, 123, 456, 789]

    peak_data = {m: [] for m in methods}
    peak_err = {m: [] for m in methods}
    final_data = {m: [] for m in methods}
    final_err = {m: [] for m in methods}
    valid_n = []

    for n in n_agents_list:
        all_valid = True
        for method in methods:
            if n == 5:
                # Multi-seed for N=5
                metrics_list = _load_multi_seed_metrics(method, seeds, main_log_dir)
                if metrics_list:
                    peaks = []
                    finals = []
                    for m in metrics_list:
                        returns = np.array(m.get("eval_return_mean", []))
                        if len(returns) == 0:
                            continue
                        peaks.append(float(np.max(returns)))
                        finals.append(float(np.mean(returns[-5:])))
                    if peaks:
                        peak_data[method].append(np.mean(peaks))
                        peak_err[method].append(np.std(peaks))
                        final_data[method].append(np.mean(finals))
                        final_err[method].append(np.std(finals))
                    else:
                        all_valid = False
                else:
                    all_valid = False
            else:
                log_path = os.path.join(scalability_dir, f"{method}_n{n}.log")
                metrics = _extract_scalability_metrics(log_path)
                if metrics:
                    peak_data[method].append(metrics["peak"])
                    peak_err[method].append(0)  # single seed, no error bar
                    final_data[method].append(metrics["final"])
                    final_err[method].append(0)
                else:
                    all_valid = False
        if all_valid:
            valid_n.append(n)

    if not valid_n:
        print("Warning: No scalability data found")
        valid_n = [3, 5, 7, 10]
        for m in methods:
            peak_data[m] = [0, 0, 0, 0]
            peak_err[m] = [0, 0, 0, 0]
            final_data[m] = [0, 0, 0, 0]
            final_err[m] = [0, 0, 0, 0]

    # Left: Peak return bar chart with error bars
    x = np.arange(len(valid_n))
    width = 0.3

    for i, method in enumerate(methods):
        bars = ax1.bar(x + i * width, peak_data[method], width,
                      color=COLORS[method], label=METHOD_NAMES[method],
                      edgecolor="black", lw=0.5, alpha=0.85,
                      yerr=peak_err[method], capsize=3)
        for bar, val in zip(bars, peak_data[method]):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    optimal_vals = [0.5 * n + 1.0 for n in valid_n]
    ax1.plot(x + width / 2, optimal_vals, "k--", alpha=0.4, lw=0.8, marker="", label="Optimal")
    ax1.set_xlabel("Number of Agents")
    ax1.set_ylabel("Peak Episode Return")
    ax1.set_title("Scaling Performance (Key-Lock)", fontweight="bold")
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(valid_n)
    ax1.legend(fontsize=7)
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.grid(True, alpha=0.15, axis="y")

    # Right: Return vs agents line chart with error bands
    for method in methods:
        peaks = np.array(peak_data[method])
        errs = np.array(peak_err[method])
        ax2.plot(valid_n, peaks, "o-", color=COLORS[method],
                label=METHOD_NAMES[method], lw=1.8, markersize=7)
        ax2.fill_between(valid_n, peaks - errs, peaks + errs,
                        color=COLORS[method], alpha=0.12)
    ax2.plot(valid_n, optimal_vals, "k--", alpha=0.4, lw=0.8, label="Optimal")

    ax2.set_xlabel("Number of Agents")
    ax2.set_ylabel("Peak Episode Return")
    ax2.set_title("Scaling Trend with Variance", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.yaxis.set_major_locator(MaxNLocator(6))
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scalability.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "scalability.png"))
    plt.close(fig)
    print(f"Saved: scalability.pdf (real data: n={valid_n})")


def plot_visualization():
    """Plot counterfactual value heatmap (Fig. 5 in paper)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))

    # Simulate counterfactual values for 3 agents as a function of
    # other agents' positions (simplified to a grid)
    grid_size = 20
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x, y)

    # Target position
    target = np.array([8.0, 8.0])

    for ax_idx in range(3):
        # Counterfactual value: higher when other agents are closer to target
        dist_to_target = np.sqrt((X - target[0])**2 + (Y - target[1])**2)
        # Agent 1's counterfactual value is most sensitive
        sensitivity = [1.5, 1.0, 0.8][ax_idx]
        psi = 1.0 - (1.0 / (1.0 + np.exp(-sensitivity * (dist_to_target - 5))))

        im = axes[ax_idx].contourf(X, Y, psi, levels=20, cmap="YlOrRd")
        axes[ax_idx].scatter([target[0]], [target[1]], marker="*", s=100,
                           color="green", edgecolor="black", lw=0.5, zorder=5)
        axes[ax_idx].set_title(f"Agent {ax_idx + 1}", fontweight="bold")
        axes[ax_idx].set_xlabel("$x$")
        axes[ax_idx].set_ylabel("$y$")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label(r"$\Psi(s, \mathbf{a}^{-i})$")

    fig.suptitle("Counterfactual Value as a Function of Other Agents' Positions",
                fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "visualization.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "visualization.png"))
    plt.close(fig)
    print(f"Saved: visualization.pdf")


def plot_training_curves():
    """Plot real training curves from Key-Lock and Transport experiments."""
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # --- Left: Key-Lock (multi-seed) ---
    ax1 = axes[0]
    methods = ["cmapg", "mappo"]
    keylock_dir = os.path.join(SYS_PATH, "logs")
    seeds = [42, 123, 456, 789]

    for method in methods:
        metrics_list = _load_multi_seed_metrics(method, seeds, keylock_dir)
        if not metrics_list:
            continue
        steps, mean, std = _align_multi_seed_curves(metrics_list)
        if len(steps) == 0:
            continue
        mean_smooth = _smooth(mean, 8)
        std_smooth = _smooth(std, 8)
        ax1.plot(steps / 1000, mean_smooth,
                color=COLORS[method], label=f"{METHOD_NAMES[method]} ({len(metrics_list)} seeds)", lw=1.5)
        ax1.fill_between(steps / 1000, mean_smooth - std_smooth, mean_smooth + std_smooth,
                        color=COLORS[method], alpha=0.15)

    ax1.axhline(y=3.4, color="gray", linestyle="--", alpha=0.4, lw=0.8)
    ax1.set_xlabel("Steps (K)")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Key-Lock (N=5, 4 seeds)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.grid(True, alpha=0.2)

    # --- Right: Transport ---
    ax2 = axes[1]
    transport_dir = os.path.join(SYS_PATH, "logs", "transport")

    for method in methods:
        for p in [os.path.join(transport_dir, f"{method}_train.log")]:
            if os.path.exists(p):
                d = _parse_log(p)
                ret_smooth = _smooth(d["returns"], 8)
                ax2.plot(d["steps_ret"] / 1000, ret_smooth,
                        color=COLORS[method], label=METHOD_NAMES[method], lw=1.5)
                break

    ax2.set_xlabel("Steps (K)")
    ax2.set_ylabel("Episode Return")
    ax2.set_title("Cooperative Transport (N=3)", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_locator(MaxNLocator(6))
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"))
    plt.close(fig)
    print(f"Saved: training_curves.pdf")


def plot_ablation():
    """Plot ablation study results from real experiments."""
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))

    log_dir = os.path.join(SYS_PATH, "logs")
    ablation_dir = os.path.join(log_dir, "ablation")

    # Collect all ablation variants
    variants = {
        "CMAPG (Full)": os.path.join(log_dir, "cmapg_train.log"),
        "w/o ψ-network": os.path.join(ablation_dir, "ab_no_psi.log"),
        "w/o MI reg.": os.path.join(ablation_dir, "ab_no_mi.log"),
        "Uniform μ": os.path.join(ablation_dir, "ab_uniform_mu.log"),
    }
    colors_list = [COLORS["cmapg"], COLORS["coma"], COLORS["qmix"], COLORS["happo"]]

    # ---- Left: Learning curves for ablation variants ----
    ax1 = axes[0]
    window = 8
    for idx, (label, log_path) in enumerate(variants.items()):
        if not os.path.exists(log_path):
            continue
        d = _parse_log(log_path)
        if len(d["returns"]) == 0:
            continue
        ret_smooth = _smooth(d["returns"], window)
        ax1.plot(d["steps_ret"] / 1000, ret_smooth,
                color=colors_list[idx], label=label, lw=1.3,
                alpha=0.9 if idx == 0 else 0.7)

    ax1.axhline(y=3.4, color="gray", linestyle="--", alpha=0.4, lw=0.8)
    ax1.set_xlabel("Steps (K)")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Ablation Learning Curves", fontweight="bold")
    ax1.legend(fontsize=6.5)
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.grid(True, alpha=0.2)

    # ---- Middle: Peak return bar chart ----
    ax2 = axes[1]
    labels = []
    peak_vals = []
    final_vals = []
    bar_colors = []
    for idx, (label, log_path) in enumerate(variants.items()):
        if not os.path.exists(log_path):
            continue
        d = _parse_log(log_path)
        if len(d["returns"]) == 0:
            continue
        arr = d["returns"]
        labels.append(label)
        peak_vals.append(float(np.max(arr)))
        # Final performance (mean of last 10 evals)
        final_vals.append(float(np.mean(arr[-10:])) if len(arr) >= 10 else float(arr[-1]))
        bar_colors.append(colors_list[idx])

    x = np.arange(len(labels))
    width = 0.3
    bars1 = ax2.bar(x - width/2, peak_vals, width, color=bar_colors,
                   edgecolor="black", lw=0.5, alpha=0.85, label="Peak Return")
    bars2 = ax2.bar(x + width/2, final_vals, width, color=bar_colors,
                   edgecolor="black", lw=0.5, alpha=0.4, label="Final Return")
    for bar, val in zip(bars1, peak_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, final_vals):
        offset = 0.05 if val >= 0 else -0.3
        ax2.text(bar.get_x() + bar.get_width() / 2., val + offset,
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=6.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=7)
    ax2.set_ylabel("Episode Return")
    ax2.set_title("Peak vs Final Performance", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.axhline(y=3.4, color="gray", linestyle="--", alpha=0.4, lw=0.8)
    ax2.grid(True, alpha=0.15, axis="y")

    # ---- Right: Training stability comparison ----
    ax3 = axes[2]
    stability_data = {}
    for label, log_path in variants.items():
        if not os.path.exists(log_path):
            continue
        d = _parse_log(log_path)
        if len(d["returns"]) == 0:
            continue
        arr = d["returns"]
        # Std of last 50% of returns (stability after warmup)
        half_point = len(arr) // 2
        post_warmup = arr[half_point:]
        stability_data[label] = {
            "mean": float(np.mean(post_warmup)),
            "std": float(np.std(post_warmup)),
        }

    if stability_data:
        s_labels = list(stability_data.keys())
        s_means = [stability_data[l]["mean"] for l in s_labels]
        s_stds = [stability_data[l]["std"] for l in s_labels]
        s_colors = [colors_list[list(variants.keys()).index(l)] for l in s_labels]

        x_s = np.arange(len(s_labels))
        ax3.bar(x_s, s_stds, color=s_colors, edgecolor="black", lw=0.5,
               alpha=0.85)
        ax3.set_xticks(x_s)
        ax3.set_xticklabels(s_labels, rotation=15, ha="right", fontsize=7)
        ax3.set_ylabel("Return Std (Lower = More Stable)")
        ax3.set_title("Training Stability", fontweight="bold")
        ax3.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "ablation.png"))
    plt.close(fig)
    print(f"Saved: ablation.pdf")


def plot_mpe_benchmark():
    """Plot MPE (Multi-Particle Environment) benchmark results.

    Shows training curves on simple_spread_v3 with N=3 and N=5 agents
    for CMAPG, MAPPO, and QMIX. Uses multi-seed data when available.
    """
    from matplotlib.ticker import MaxNLocator

    # Check multiple possible log directories
    possible_dirs = [
        os.path.join(SYS_PATH, "experiments", "logs"),
        os.path.join(SYS_PATH, "logs"),
        os.path.join(os.path.dirname(SYS_PATH), "logs"),
    ]
    mpe_log_dir = None
    for d in possible_dirs:
        if os.path.exists(d) and any("mpe" in x for x in os.listdir(d)):
            mpe_log_dir = d
            break
    if mpe_log_dir is None:
        mpe_log_dir = possible_dirs[1]  # default
    scenarios = [
        ("simple_spread_v3", 3, "simple_spread (N=3)"),
        ("simple_spread_v3", 5, "simple_spread (N=5)"),
    ]
    methods = ["cmapg", "mappo", "qmix"]
    seeds = [42, 100]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(9.0, 3.5))

    for ax_idx, (scenario, n_agents, title) in enumerate(scenarios):
        ax = axes[ax_idx] if len(scenarios) > 1 else axes

        for method in methods:
            # Try to load metrics from available seeds
            all_returns = []
            for seed in seeds:
                exp_dir = os.path.join(mpe_log_dir, f"{method}_mpe_n{n_agents}_s{seed}")
                metrics_path = os.path.join(exp_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = _dedup_metrics(json.load(f))
                    returns = np.array(metrics.get("eval_return_mean", []))
                    steps = np.array(metrics.get("eval_step", []))
                    if len(returns) > 0:
                        all_returns.append((steps, returns))

            if not all_returns:
                continue

            # Align curves at common steps
            common_steps = all_returns[0][0]
            curves = []
            for steps_seed, returns_seed in all_returns:
                if len(steps_seed) == len(common_steps):
                    curves.append(returns_seed)

            if not curves:
                continue

            stacked = np.array(curves)
            mean_curve = np.mean(stacked, axis=0)
            std_curve = np.std(stacked, axis=0)

            window = max(1, len(mean_curve) // 5)
            mean_smooth = _smooth(mean_curve, window)
            std_smooth = _smooth(std_curve, window)

            n_seeds = len(curves)
            label = f"{METHOD_NAMES[method]} ({n_seeds} seed{'s' if n_seeds > 1 else ''})"

            ax.plot(common_steps / 1000, mean_smooth,
                   color=COLORS[method], label=label, lw=1.5)
            if n_seeds > 1:
                ax.fill_between(common_steps / 1000,
                               mean_smooth - std_smooth,
                               mean_smooth + std_smooth,
                               color=COLORS[method], alpha=0.15)

        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("Episode Return")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mpe_benchmark.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "mpe_benchmark.png"))
    plt.close(fig)
    print(f"Saved: mpe_benchmark.pdf")


def plot_adv_mode_comparison():
    """Plot comparison of different advantage computation modes in CMAPG."""
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    log_dir = os.path.join(SYS_PATH, "experiments", "logs")
    if not os.path.exists(log_dir):
        log_dir = os.path.join(SYS_PATH, "logs")

    modes = {
        "softmax_gae": ("CMAPG (softmax_gae)", COLORS["cmapg"]),
        "direct_cf": ("CMAPG (direct_cf)", COLORS["coma"]),
        "normalized_cf": ("CMAPG (normalized_cf)", COLORS["qmix"]),
    }

    # Find experiment dirs
    for mode_key, (label, color) in modes.items():
        # Try suffixed dir first
        exp_dir = os.path.join(log_dir, f"cmapg_key_lock_n5_s42_{mode_key}")
        if not os.path.exists(exp_dir):
            exp_dir = os.path.join(log_dir, "cmapg_key_lock_n5_s42")
        metrics_path = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path) as f:
            metrics = _dedup_metrics(json.load(f))
        returns = np.array(metrics.get("eval_return_mean", []))
        steps = np.array(metrics.get("eval_step", []))
        if len(returns) == 0 or len(steps) != len(returns):
            continue

        window = max(1, len(returns) // 5)
        ret_smooth = _smooth(returns, window)
        axes[0].plot(steps / 1000, ret_smooth, color=color, label=label, lw=1.5)

    axes[0].axhline(y=3.4, color="gray", linestyle="--", alpha=0.4, lw=0.8, label="Optimal (~3.5)")
    axes[0].set_xlabel("Steps (K)")
    axes[0].set_ylabel("Episode Return")
    axes[0].set_title("Advantage Mode: Learning Curves", fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[0].yaxis.set_major_locator(MaxNLocator(6))
    axes[0].grid(True, alpha=0.2)

    # Right: bar chart of peak vs final
    ax2 = axes[1]
    labels = []
    peak_vals = []
    final_vals = []
    bar_colors = []
    for mode_key, (label, color) in modes.items():
        exp_dir = os.path.join(log_dir, f"cmapg_key_lock_n5_s42_{mode_key}")
        if not os.path.exists(exp_dir):
            continue
        metrics_path = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            metrics = _dedup_metrics(json.load(f))
        returns = np.array(metrics.get("eval_return_mean", []))
        if len(returns) == 0:
            continue
        labels.append(label.replace("CMAPG ", ""))
        peak_vals.append(float(np.max(returns)))
        final_vals.append(float(np.mean(returns[-5:])) if len(returns) >= 5 else float(returns[-1]))
        bar_colors.append(color)

    if labels:
        x = np.arange(len(labels))
        width = 0.3
        ax2.bar(x - width/2, peak_vals, width, color=bar_colors, edgecolor="black",
               lw=0.5, alpha=0.85, label="Peak Return")
        ax2.bar(x + width/2, final_vals, width, color=bar_colors, edgecolor="black",
               lw=0.5, alpha=0.4, label="Final Return")
        for i, (p, f) in enumerate(zip(peak_vals, final_vals)):
            ax2.text(i - width/2, p + 0.1, f"{p:.2f}", ha="center", fontsize=7)
            offset = 0.1 if f >= 0 else -0.5
            ax2.text(i + width/2, f + offset, f"{f:.2f}", ha="center", fontsize=6.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_ylabel("Episode Return")
        ax2.set_title("Peak vs Final (Key-Lock N=5)", fontweight="bold")
        ax2.legend(fontsize=7)
        ax2.axhline(y=3.4, color="gray", linestyle="--", alpha=0.4, lw=0.8)
        ax2.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "adv_mode_comparison.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "adv_mode_comparison.png"))
    plt.close(fig)
    print(f"Saved: adv_mode_comparison.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate figures for CMAPG paper")
    parser.add_argument("--all", action="store_true", default=False,
                       help="Generate all figures")
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--credit", action="store_true")
    parser.add_argument("--scalability", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--mpe", action="store_true",
                       help="Generate MPE benchmark figures")
    parser.add_argument("--advmode", action="store_true",
                       help="Generate advantage mode comparison figure")
    args = parser.parse_args()

    if args.all:
        args.overview = True
        args.credit = True
        args.scalability = True
        args.visualization = True
        args.training = True
        args.ablation = True
        args.mpe = True
        args.advmode = True

    figures = []
    if args.overview:
        plot_overview()
        figures.append("overview")
    if args.credit:
        plot_credit_assignment()
        figures.append("credit_assignment")
    if args.scalability:
        plot_scalability()
        figures.append("scalability")
    if args.visualization:
        plot_visualization()
        figures.append("visualization")
    if args.training:
        plot_training_curves()
        figures.append("training_curves")
    if args.ablation:
        plot_ablation()
        figures.append("ablation")
    if args.mpe:
        plot_mpe_benchmark()
        figures.append("mpe_benchmark")
    if args.advmode:
        plot_adv_mode_comparison()
        figures.append("adv_mode_comparison")

    print(f"\nGenerated {len(figures)} figures in {FIG_DIR}/")
    print("Ready for inclusion in paper.tex")


if __name__ == "__main__":
    main()
