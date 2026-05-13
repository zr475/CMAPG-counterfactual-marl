"""Diagnostic analysis: Why doesn't CMAPG outperform MAPPO?

Analyzes training logs to answer:
1. Does Ψ learn to copy Q? (high q_psi_corr → MI regularization fails)
2. Do cf_advantages provide meaningful per-agent differentiation?
3. Does the softmax mixing destroy counterfactual sign information?
4. How does the effective advantage compare between CMAPG variants?

Usage:
    python analyze_cmapg_diagnostics.py --log_dir ./logs
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "softmax_gae": "#1f77b4",
    "direct_cf": "#d62728",
    "normalized_cf": "#2ca02c",
    "mappo": "#ff7f0e",
}


def load_metrics(exp_dir: str) -> dict:
    """Load metrics.json from an experiment directory, return deduplicated metrics."""
    path = os.path.join(exp_dir, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        metrics = json.load(f)
    return _dedup_metrics(metrics)


def _dedup_metrics(metrics: dict) -> dict:
    """Remove duplicate consecutive eval entries."""
    returns = np.array(metrics.get("eval_return_mean", []))
    if len(returns) <= 1:
        return metrics
    keep = [0]
    for i in range(1, len(returns)):
        if abs(returns[i] - returns[i - 1]) > 1e-6:
            keep.append(i)
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


def analyze_diagnostics(log_dir: str, fig_dir: str):
    """Main analysis: load all experiments and produce diagnostic figures."""

    # Find experiment directories
    variants = {
        "CMAPG (softmax_gae)": os.path.join(log_dir, "cmapg_key_lock_n5_s42"),
        "CMAPG (direct_cf)": os.path.join(log_dir, "cmapg_key_lock_n5_s42"),
        "CMAPG (normalized_cf)": os.path.join(log_dir, "cmapg_key_lock_n5_s42"),
        "MAPPO": os.path.join(log_dir, "mappo_key_lock_n5_s42"),
    }

    # Find the actual adv_mode experiments
    all_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    print("Available experiment directories:")
    for d in all_dirs:
        metrics_path = os.path.join(log_dir, d, "metrics.json")
        if os.path.exists(metrics_path):
            m = load_metrics(os.path.join(log_dir, d))
            n_metrics = len(m)
            print(f"  {d}: {n_metrics} metrics entries")

    # Find experiments with diagnostic data (q_psi_corr, sign_agreement, cf_adv_mean)
    diag_experiments = {}
    for d in all_dirs:
        exp_path = os.path.join(log_dir, d)
        metrics = load_metrics(exp_path)
        if "q_psi_corr" in metrics:
            diag_experiments[d] = metrics
            print(f"\nDiagnostic data found: {d}")
            for key in ["q_psi_corr", "sign_agreement", "cf_adv_mean", "cf_adv_std"]:
                if key in metrics:
                    vals = metrics[key]
                    if isinstance(vals, list) and len(vals) > 0:
                        print(f"  {key}: {len(vals)} entries, mean={np.mean(vals):.4f}, last={vals[-1]:.4f}")

    if not diag_experiments:
        print("\nNo diagnostic data found yet. Experiments may still be running.")
        print("Re-run after experiments complete for full analysis.")
        return

    # Generate analysis figure
    n_exps = len(diag_experiments)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for exp_name, metrics in diag_experiments.items():
        steps = np.array(metrics.get("step", []))
        if len(steps) == 0:
            continue

        # Panel 1: Q-Ψ correlation over time (should be LOW if MI works)
        ax = axes[0, 0]
        q_psi = np.array(metrics.get("q_psi_corr", []))
        if len(q_psi) > 0 and len(q_psi) == len(steps):
            ax.plot(steps / 1000, q_psi, label=exp_name, lw=1.5)
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("Corr(Q, mean(Ψ))")
        ax.set_title("Q-Ψ Correlation\n(lower = better MI regularization)")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Danger zone")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

        # Panel 2: Sign agreement between cf_adv and GAE
        ax = axes[0, 1]
        sign_agree = np.array(metrics.get("sign_agreement", []))
        if len(sign_agree) > 0 and len(sign_agree) == len(steps):
            ax.plot(steps / 1000, sign_agree, label=exp_name, lw=1.5)
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("Sign Agreement")
        ax.set_title("Sign Agreement: cf_adv vs GAE\n(0.5 = random, 1.0 = perfect)")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

        # Panel 3: cf_adv mean magnitude
        ax = axes[0, 2]
        cf_mean = np.array(metrics.get("cf_adv_mean", []))
        cf_std = np.array(metrics.get("cf_adv_std", []))
        if len(cf_mean) > 0 and len(cf_mean) == len(steps):
            ax.plot(steps / 1000, cf_mean, label=f"{exp_name} mean", lw=1.5)
            ax.fill_between(steps / 1000, cf_mean - cf_std, cf_mean + cf_std, alpha=0.15)
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("cf_adv = Q - Ψ")
        ax.set_title("Counterfactual Advantage Distribution")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

        # Panel 4: Softmax entropy (if available; only for softmax_gae)
        ax = axes[1, 0]
        sm_entropy = np.array(metrics.get("softmax_entropy", []))
        if len(sm_entropy) > 0 and len(sm_entropy) == len(steps):
            n_agents = 5  # assume 5 agents
            max_entropy = np.log(n_agents)
            ax.plot(steps / 1000, sm_entropy / max_entropy, label=exp_name, lw=1.5)
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("Normalized Entropy")
        ax.set_title("Softmax Weight Entropy\n(1.0 = uniform weights)")
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

        # Panel 5: Training return
        ax = axes[1, 1]
        returns = np.array(metrics.get("eval_return_mean", []))
        eval_steps = np.array(metrics.get("eval_step", []))
        if len(returns) > 0 and len(eval_steps) == len(returns):
            ax.plot(eval_steps / 1000, returns, label=exp_name, lw=1.5)
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("Episode Return")
        ax.set_title("Training Performance")
        ax.axhline(y=3.4, color="gray", linestyle="--", alpha=0.4)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

        # Panel 6: Policy loss
        ax = axes[1, 2]
        policy_loss = np.array(metrics.get("policy_loss", []))
        if len(policy_loss) > 0:
            p_steps = np.array(metrics.get("step", []))[:len(policy_loss)]
            if len(p_steps) == len(policy_loss):
                ax.plot(p_steps / 1000, policy_loss, label=exp_name, lw=1.0, alpha=0.7)
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss Over Time")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    fig.suptitle("CMAPG Diagnostic Analysis: Why No Difference from MAPPO?",
                fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "diagnostics.pdf"))
    fig.savefig(os.path.join(fig_dir, "diagnostics.png"))
    plt.close(fig)
    print(f"\nDiagnostic figure saved to {fig_dir}/diagnostics.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    log_dir = args.log_dir
    # Try multiple paths
    possible_dirs = [
        log_dir,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "logs"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"),
    ]
    actual_log_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            actual_log_dir = d
            break

    if actual_log_dir is None:
        print(f"Log directory not found. Tried: {possible_dirs}")
        return

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    print(f"Analyzing logs in: {actual_log_dir}")
    analyze_diagnostics(actual_log_dir, fig_dir)


if __name__ == "__main__":
    main()
