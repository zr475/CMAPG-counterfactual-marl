"""Main training script for CMAPG and baseline algorithms.

Usage:
    python train.py --algo cmapg --env key_lock --n_agents 5
    python train.py --algo mappo --env key_lock --n_agents 5
    python train.py --algo qmix --env key_lock --n_agents 5
    python train.py --algo cmapg --env transport --n_agents 3
"""

import argparse
import os
import sys
import signal
import time
import json
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.envs import SequentialKeyLockEnv, CooperativeTransportEnv
from experiments.algorithms import CMAPG, MAPPO, QMIX
from experiments.utils import Logger, evaluate, compute_credit_assignment_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="MARL Training with CMAPG")

    # Experiment
    parser.add_argument("--algo", type=str, default="cmapg",
                        choices=["cmapg", "mappo", "qmix"])
    parser.add_argument("--env", type=str, default="key_lock",
                        choices=["key_lock", "transport"])
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name suffix")

    # Training
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--log_freq", type=int, default=1_000)
    parser.add_argument("--save_freq", type=int, default=50_000)

    # PPO-specific
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # CMAPG-specific
    parser.add_argument("--lambda_q", type=float, default=1.0)
    parser.add_argument("--lambda_psi", type=float, default=0.5)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument("--alpha_default", type=float, default=0.5)

    # QMIX-specific
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_anneal", type=int, default=50000)
    parser.add_argument("--buffer_capacity", type=int, default=5000)

    # Misc
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")

    return parser.parse_args()


def make_env(env_name: str, n_agents: int, seed: int):
    if env_name == "key_lock":
        return SequentialKeyLockEnv(n_agents=n_agents, max_steps=200, seed=seed, press_radius=1.5)
    elif env_name == "transport":
        return CooperativeTransportEnv(n_agents=n_agents, max_steps=200, seed=seed)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def create_algorithm(algo_name: str, env, args):
    """Create algorithm instance based on name."""
    # Determine observation and action dimensions from env
    sample_obs = env.observation_space[f"agent_0"]
    if hasattr(sample_obs, "shape"):
        obs_dim = sample_obs.shape[0]
    else:
        obs_dim = sample_obs.n

    obs_dims = [obs_dim] * args.n_agents
    state_dim = obs_dim * args.n_agents  # Global state = concatenated observations

    if isinstance(env.action_space[f"agent_0"], type(env.action_space[f"agent_0"])):
        if hasattr(env.action_space[f"agent_0"], "n"):
            # Discrete actions
            action_dims = [env.action_space[f"agent_0"].n] * args.n_agents
            discrete = True
        else:
            action_dims = [env.action_space[f"agent_0"].shape[0]] * args.n_agents
            discrete = False
    else:
        action_dims = [4] * args.n_agents  # Default for Key-Lock
        discrete = True

    kwargs = dict(
        n_agents=args.n_agents,
        obs_dims=obs_dims,
        state_dim=state_dim,
        action_dims=action_dims,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
    )

    if algo_name == "cmapg":
        kwargs.update(
            discrete_actions=discrete,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            lambda_q=args.lambda_q,
            lambda_psi=args.lambda_psi,
            lambda_mi=args.lambda_mi,
            alpha_default=args.alpha_default,
        )
        algo = CMAPG(**kwargs)
    elif algo_name == "mappo":
        kwargs.update(
            discrete_actions=discrete,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
        )
        algo = MAPPO(**kwargs)
    elif algo_name == "qmix":
        n_actions = action_dims[0]
        kwargs.pop("action_dims", None)  # QMIX uses n_actions, not action_dims
        kwargs.update(
            n_actions=n_actions,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_anneal=args.epsilon_anneal,
        )
        algo = QMIX(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return algo


def train_policy_gradient(algo, env_ctor, args, logger: Logger):
    """Training loop for on-policy algorithms (CMAPG, MAPPO)."""
    from experiments.utils.buffer import TrajectoryBuffer

    env = env_ctor()
    obs, _ = env.reset()
    obs_list = [obs[f"agent_{i}"] for i in range(args.n_agents)]

    buffer = TrajectoryBuffer()
    episode_return = 0.0
    episode_num = 0
    total_steps = 0
    best_return = -float("inf")

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = os.path.join(logger.log_dir, args.resume) if not os.path.isabs(args.resume) else args.resume
        print(f"Resuming from: {resume_path}")
        algo.load(resume_path)
        state_path = resume_path.replace(".pt", "_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            total_steps = state.get("total_steps", 0)
            episode_num = state.get("episode_num", 0)
            best_return = state.get("best_return", -float("inf"))
            logger.metrics = state.get("metrics", {})
            # For QMIX: restore epsilon
            if hasattr(algo, "steps"):
                algo.steps = total_steps
            print(f"Resumed at step {total_steps}, episode {episode_num}")

    while total_steps < args.total_steps:
        # Collect actions
        actions, log_probs = algo.get_actions(obs_list, deterministic=False)
        action_dict = {f"agent_{i}": actions[i] for i in range(args.n_agents)}
        log_prob_dict = {f"agent_{i}": log_probs[i].item() for i in range(args.n_agents)}

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated
        episode_return += reward

        # Global state (approximate as concatenated observations)
        state = np.concatenate(obs_list)

        # Store transition
        buffer.add(obs, action_dict, reward, done, state, log_prob_dict)

        total_steps += 1

        # Update when buffer is full
        if len(buffer) >= args.episode_length:
            batch = buffer.get_batch(args.n_agents, args.device)
            metrics = algo.update(
                batch["obs"], batch["actions"], batch["rewards"],
                batch["dones"], batch["states"], batch["old_log_probs"],
            )
            logger.log(metrics, total_steps)
            buffer.clear()

        if done:
            obs, _ = env.reset()
            episode_num += 1
            logger.log({"episode_return": episode_return, "episode": float(episode_num)}, total_steps)
            episode_return = 0.0
        else:
            obs = next_obs

        obs_list = [obs[f"agent_{i}"] for i in range(args.n_agents)]

        # Evaluation
        if total_steps % args.eval_freq == 0:
            eval_results = evaluate(algo, env_ctor, args.n_agents, n_episodes=10)
            logger.log_eval(total_steps, eval_results["mean_return"], eval_results["std_return"])

            # Credit assignment accuracy
            ca_env = env_ctor()
            ca_acc = compute_credit_assignment_accuracy(algo, ca_env, args.n_agents)
            logger.log_eval(total_steps, eval_results["mean_return"], eval_results["std_return"],
                           {"ca_accuracy": ca_acc})
            ca_env.close()
            print(f"Step {total_steps:7d} | Return: {eval_results['mean_return']:8.3f} ± {eval_results['std_return']:6.3f}")

        # Save
        if total_steps % args.save_freq == 0:
            save_path = os.path.join(logger.log_dir, f"model_{total_steps}.pt")
            algo.save(save_path)
            # Save training state for resume
            state = {
                "total_steps": total_steps,
                "episode_num": episode_num,
                "episode_return": episode_return,
                "best_return": best_return,
                "metrics": logger.metrics,
            }
            state_path = save_path.replace(".pt", "_state.json")
            with open(state_path, "w") as f:
                json.dump(state, f)

        # Log
        if total_steps % args.log_freq == 0 and len(list(logger.metrics.keys())) > 0:
            recent_key = list(logger.metrics.keys())[-1]
            print(f"Step {total_steps:7d} | {recent_key}: {logger.metrics[recent_key][-1]:.4f}")

    env.close()
    logger.save()


def train_qmix(algo: QMIX, env_ctor, args, logger: Logger):
    """Training loop for off-policy QMIX."""
    from experiments.utils.buffer import ReplayBuffer

    buffer = ReplayBuffer(args.buffer_capacity)
    env = env_ctor()
    obs, _ = env.reset()
    obs_list = [obs[f"agent_{i}"] for i in range(args.n_agents)]

    episode_return = 0.0
    episode_num = 0
    total_steps = 0
    best_return = -float("inf")

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = os.path.join(logger.log_dir, args.resume) if not os.path.isabs(args.resume) else args.resume
        print(f"Resuming from: {resume_path}")
        algo.load(resume_path)
        state_path = resume_path.replace(".pt", "_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            total_steps = state.get("total_steps", 0)
            episode_num = state.get("episode_num", 0)
            best_return = state.get("best_return", -float("inf"))
            logger.metrics = state.get("metrics", {})
            if hasattr(algo, "steps"):
                algo.steps = total_steps
            print(f"Resumed at step {total_steps}, episode {episode_num}")

    while total_steps < args.total_steps:
        # Select actions (epsilon-greedy)
        actions = algo.select_actions(obs_list, explore=True)
        action_dict = {f"agent_{i}": int(actions[i]) for i in range(args.n_agents)}

        next_obs, reward, terminated, truncated, _ = env.step(action_dict)
        done = terminated or truncated
        episode_return += reward

        state = np.concatenate(obs_list)
        next_state = np.concatenate([next_obs[f"agent_{i}"] for i in range(args.n_agents)])

        # Store
        transition = {
            **{f"obs_{i}": obs[f"agent_{i}"] for i in range(args.n_agents)},
            **{f"next_obs_{i}": next_obs[f"agent_{i}"] for i in range(args.n_agents)},
            "actions": np.array(actions),
            "rewards": np.array([reward]),
            "dones": np.array([float(done)]),
            "states": state,
            "next_states": next_state,
        }
        buffer.push(transition)
        total_steps += 1

        # Update
        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            metrics = algo.update(batch)
            logger.log(metrics, total_steps)

        if done:
            obs, _ = env.reset()
            episode_num += 1
            logger.log({"episode_return": episode_return, "episode": float(episode_num)}, total_steps)
            episode_return = 0.0
        else:
            obs = next_obs

        obs_list = [obs[f"agent_{i}"] for i in range(args.n_agents)]

        if total_steps % args.eval_freq == 0:
            eval_results = evaluate(algo, env_ctor, args.n_agents, n_episodes=10)
            logger.log_eval(total_steps, eval_results["mean_return"], eval_results["std_return"])
            print(f"Step {total_steps:7d} | Return: {eval_results['mean_return']:8.3f} ± {eval_results['std_return']:6.3f}")

        # Save checkpoint
        if total_steps % args.save_freq == 0:
            save_path = os.path.join(logger.log_dir, f"model_{total_steps}.pt")
            algo.save(save_path)
            state = {
                "total_steps": total_steps,
                "episode_num": episode_num,
                "episode_return": episode_return,
                "best_return": best_return,
                "metrics": logger.metrics,
            }
            state_path = save_path.replace(".pt", "_state.json")
            with open(state_path, "w") as f:
                json.dump(state, f)

    env.close()
    logger.save()


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = f"{args.algo}_{args.env}_n{args.n_agents}_s{args.seed}"
    if args.exp_name:
        exp_name += f"_{args.exp_name}"
    log_dir = os.path.join(args.log_dir, exp_name)
    logger = Logger(log_dir)

    print(f"=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Algorithm: {args.algo} | Environment: {args.env} | Agents: {args.n_agents}")
    print(f"Total steps: {args.total_steps} | Device: {args.device}")
    print(f"Log dir: {log_dir}")
    print(f"=" * 60)

    # Create environment factory
    def env_ctor():
        return make_env(args.env, args.n_agents, args.seed)

    env = env_ctor()

    # Create algorithm
    algo = create_algorithm(args.algo, env, args)
    algo.to(args.device)

    print(f"Algorithm created: {type(algo).__name__}")
    if hasattr(algo, "actors"):
        print(f"Parameters: {sum(p.numel() for actor in algo.actors for p in actor.parameters())}")

    env.close()

    # Graceful shutdown: save checkpoint on Ctrl+C
    interrupted = [False]
    def _save_on_exit(signum=None, frame=None):
        if not interrupted[0]:
            interrupted[0] = True
            save_path = os.path.join(log_dir, "model_interrupted.pt")
            print(f"\n\nInterrupted! Saving checkpoint to {save_path} ...")
            try:
                algo.save(save_path)
                state = {"metrics": logger.metrics, "experiment": exp_name}
                with open(save_path.replace(".pt", "_state.json"), "w") as f:
                    json.dump(state, f, default=str)
                logger.save()
                print("Checkpoint saved. Resume with:")
                print(f"  --resume {save_path}")
            except Exception as e:
                print(f"Save failed: {e}")
            sys.exit(0)

    signal.signal(signal.SIGINT, _save_on_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, _save_on_exit)  # kill / system shutdown

    # Train
    try:
        if args.algo in ["cmapg", "mappo"]:
            train_policy_gradient(algo, env_ctor, args, logger)
        else:
            train_qmix(algo, env_ctor, args, logger)
    except KeyboardInterrupt:
        _save_on_exit()

    print(f"Training complete! Results saved to {log_dir}")


if __name__ == "__main__":
    main()
