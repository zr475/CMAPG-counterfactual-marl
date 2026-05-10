"""Training utilities and helpers."""

from typing import Dict, List, Tuple, Any
import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    """Simple replay buffer for off-policy algorithms (QMIX)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Dict[str, np.ndarray]):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = {}
        for key in self.buffer[0].keys():
            batch[key] = torch.FloatTensor(np.stack([self.buffer[i][key] for i in indices]))
        return batch

    def __len__(self):
        return len(self.buffer)


class TrajectoryBuffer:
    """On-policy trajectory buffer for policy gradient methods."""

    def __init__(self):
        self.obs: Dict[str, List] = {}
        self.actions: Dict[str, List] = {}
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.states: List[np.ndarray] = []
        self.log_probs: Dict[str, List] = {}
        self._initialized = False

    def add(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: float,
        dones: bool,
        state: np.ndarray,
        log_probs: Dict[str, float],
    ):
        if not self._initialized:
            for agent_id in obs:
                self.obs[agent_id] = []
                self.actions[agent_id] = []
                self.log_probs[agent_id] = []
            self._initialized = True

        for agent_id, o in obs.items():
            self.obs[agent_id].append(o)
            self.actions[agent_id].append(actions[agent_id])
            self.log_probs[agent_id].append(log_probs[agent_id])
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.states.append(state)

    def get_batch(self, n_agents: int, device: str) -> Dict[str, Any]:
        seq_len = len(self.rewards)

        obs_batch = [
            torch.FloatTensor(np.stack(self.obs[f"agent_{i}"])).unsqueeze(0).to(device)
            for i in range(n_agents)
        ]
        actions_batch = [
            torch.FloatTensor(np.stack(self.actions[f"agent_{i}"])).unsqueeze(0).to(device)
            for i in range(n_agents)
        ]
        rewards_batch = torch.FloatTensor(self.rewards).unsqueeze(0).to(device)
        dones_batch = torch.FloatTensor(self.dones).unsqueeze(0).to(device)
        states_batch = torch.FloatTensor(np.stack(self.states)).unsqueeze(0).to(device)
        old_log_probs_batch = [
            torch.FloatTensor(self.log_probs[f"agent_{i}"]).unsqueeze(0).to(device)
            for i in range(n_agents)
        ]

        return {
            "obs": obs_batch,
            "actions": actions_batch,
            "rewards": rewards_batch,
            "dones": dones_batch,
            "states": states_batch,
            "old_log_probs": old_log_probs_batch,
        }

    def clear(self):
        self.obs = {}
        self.actions = {}
        self.rewards = []
        self.dones = []
        self.states = []
        self.log_probs = {}
        self._initialized = False

    def __len__(self):
        return len(self.rewards)
