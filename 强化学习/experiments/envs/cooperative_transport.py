"""Heterogeneous Cooperative Transport environment.

Agents with different capabilities must cooperatively transport objects to targets.
Each agent has different speed and carrying capacity, creating natural heterogeneity
that requires accurate credit assignment.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CooperativeTransportEnv(gym.Env):
    """Heterogeneous Cooperative Transport.

    Agents must push a box to a target location. Each agent has different
    strength (how much it can push) and speed. The box moves according to
    the net force applied by nearby agents. Reward is based on the box's
    distance to the target.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        n_agents: int = 3,
        world_size: float = 10.0,
        target_pos: Optional[np.ndarray] = None,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.world_size = world_size
        self.max_steps = max_steps

        # Each agent's capability
        self.rng = np.random.RandomState(seed)
        self.agent_strengths = self.rng.uniform(0.5, 2.0, n_agents)
        self.agent_speeds = self.rng.uniform(0.5, 1.5, n_agents)

        # Observation: [agent_x, agent_y, box_x, box_y, target_x, target_y, my_strength, my_speed]
        obs_dim = 6 + 2
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
            for i in range(n_agents)
        })

        # Action: [dx, dy] continuous force vector
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            for i in range(n_agents)
        })

        if target_pos is None:
            target_pos = np.array([world_size * 0.8, world_size * 0.8])
        self.target_pos = target_pos

    def _reset_internal(self):
        # Randomize agent positions
        self.agent_positions = self.rng.uniform(0, self.world_size * 0.3, (self.n_agents, 2))
        self.box_pos = self.rng.uniform(self.world_size * 0.3, self.world_size * 0.5, 2)
        self.steps = 0

        # Track contributions
        self.individual_contributions = np.zeros(self.n_agents)

    def _get_obs(self):
        obs = {}
        for i in range(self.n_agents):
            obs_array = np.concatenate([
                self.agent_positions[i],
                self.box_pos,
                self.target_pos,
                [self.agent_strengths[i], self.agent_speeds[i]],
            ]).astype(np.float32)
            obs[f"agent_{i}"] = obs_array
        return obs

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, bool, Dict]:
        self.steps += 1
        net_force = np.zeros(2)

        for i in range(self.n_agents):
            action = actions[f"agent_{i}"]
            dist = np.linalg.norm(self.agent_positions[i] - self.box_pos)

            # Agent can only push if close enough to the box
            if dist < 1.5:
                force = action * self.agent_strengths[i]
                net_force += force
                self.individual_contributions[i] += np.linalg.norm(force)

            # Move agent
            self.agent_positions[i] += action * self.agent_speeds[i] * 0.1
            self.agent_positions[i] = np.clip(self.agent_positions[i], 0, self.world_size)

        # Move box
        self.box_pos += net_force * 0.05
        self.box_pos = np.clip(self.box_pos, 0, self.world_size)

        # Reward: negative distance to target
        prev_dist = np.linalg.norm(self.box_pos - self.target_pos)
        step_reward = -prev_dist / self.world_size

        # Bonus for close to target
        if prev_dist < 1.0:
            step_reward += 1.0
        if prev_dist < 0.5:
            step_reward += 2.0
        if prev_dist < 0.1:
            step_reward += 5.0

        terminated = prev_dist < 0.1
        truncated = self.steps >= self.max_steps

        return self._get_obs(), float(step_reward), terminated, truncated, {}

    def get_ground_truth_credit(self) -> np.ndarray:
        """Return normalized ground-truth contribution."""
        total = self.individual_contributions.sum()
        if total > 0:
            return self.individual_contributions / total
        return np.ones(self.n_agents) / self.n_agents

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._reset_internal()
        return self._get_obs(), {}
