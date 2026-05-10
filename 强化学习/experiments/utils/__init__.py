"""Logging and evaluation utilities."""

import os
import json
import time
from typing import Dict, List, Any
import numpy as np
import torch


class Logger:
    """Simple logger for tracking training metrics and saving results."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics: Dict[str, List[float]] = {}
        self.start_time = time.time()

    def log(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def log_eval(self, step: int, avg_return: float, std_return: float, extra: Dict = None):
        if "eval_step" not in self.metrics:
            self.metrics["eval_step"] = []
            self.metrics["eval_return_mean"] = []
            self.metrics["eval_return_std"] = []
        self.metrics["eval_step"].append(step)
        self.metrics["eval_return_mean"].append(avg_return)
        self.metrics["eval_return_std"].append(std_return)
        if extra:
            for k, v in extra.items():
                key = f"eval_{k}"
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(v)

    def save(self):
        filepath = os.path.join(self.log_dir, "metrics.json")
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time


def evaluate(
    algorithm,
    env_ctor,
    n_agents: int,
    n_episodes: int = 10,
    use_states: bool = True,
    render: bool = False,
) -> Dict[str, float]:
    """Evaluate an algorithm over multiple episodes.

    Returns mean and std of episode returns.
    """
    returns = []
    for ep in range(n_episodes):
        env = env_ctor()
        obs, _ = env.reset()
        obs_list = [obs[f"agent_{i}"] for i in range(n_agents)]

        ep_return = 0.0
        done = False

        while not done:
            if hasattr(algorithm, "get_actions"):
                actions, _ = algorithm.get_actions(obs_list, deterministic=True)
            else:
                actions = algorithm.select_actions(obs_list, explore=False).tolist()

            action_dict = {f"agent_{i}": actions[i] for i in range(n_agents)}
            next_obs, reward, terminated, truncated, _ = env.step(action_dict)
            done = terminated or truncated
            ep_return += reward
            obs_list = [next_obs[f"agent_{i}"] for i in range(n_agents)]

            if render:
                env.render()

        returns.append(ep_return.to(algorithm.device) if isinstance(ep_return, torch.Tensor) else ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def compute_credit_assignment_accuracy(
    algorithm,
    env,
    n_agents: int,
    n_steps: int = 50,
) -> float:
    """Evaluate credit assignment accuracy.

    Computes Pearson correlation between algorithm's advantage estimates
    and environment's ground-truth individual contributions.
    """
    obs, _ = env.reset()
    obs_list = [obs[f"agent_{i}"] for i in range(n_agents)]

    all_estimated = []
    all_ground_truth = []

    for _ in range(n_steps):
        # Get actions and log probs
        if hasattr(algorithm, "get_actions"):
            actions, log_probs = algorithm.get_actions(obs_list, deterministic=False)
        else:
            actions = algorithm.select_actions(obs_list, explore=False).tolist()
            log_probs = [0.0 for _ in range(n_agents)]

        action_dict = {f"agent_{i}": actions[i] for i in range(n_agents)}
        next_obs, reward, terminated, truncated, info = env.step(action_dict)

        # Get ground-truth credit from env
        if hasattr(env, "get_ground_truth_credit"):
            gt_credit = env.get_ground_truth_credit()
            all_ground_truth.append(gt_credit)

        # Use log probs as a proxy for advantage estimates
        # (actual advantages would come from the algorithm's internal computation)
        est = np.array([lp.item() if hasattr(lp, 'item') else float(lp) for lp in log_probs])
        all_estimated.append(est)

        obs_list = [next_obs[f"agent_{i}"] for i in range(n_agents)]

        if terminated or truncated:
            obs, _ = env.reset()
            obs_list = [obs[f"agent_{i}"] for i in range(n_agents)]

    if len(all_ground_truth) == 0:
        return 0.0

    # Compute Pearson correlation
    est_arr = np.array(all_estimated)
    gt_arr = np.array(all_ground_truth)

    correlations = []
    for i in range(n_agents):
        if est_arr.shape[1] > i and gt_arr.shape[1] > i:
            est_i = est_arr[:, i]
            gt_i = gt_arr[:, i]
            if np.std(est_i) > 1e-8 and np.std(gt_i) > 1e-8:
                corr = np.corrcoef(est_i, gt_i)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 0.0
