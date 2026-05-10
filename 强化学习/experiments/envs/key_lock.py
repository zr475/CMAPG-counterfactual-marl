"""Sequential Key-Lock environment for credit assignment evaluation.

Agents must press N buttons in a specific order. Reward is +1 only when all buttons
are pressed in the correct sequence. This creates a sparse, long-horizon credit
assignment problem where each agent's contribution is identifiable.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SequentialKeyLockEnv(gym.Env):
    """Sequential Key-Lock environment.

    N agents, each starts at a random position on a line. There are N buttons
    positioned on the same line. Buttons must be pressed in order (button 0,
    then button 1, ..., then button N-1). Each agent has a "press" action that
    activates the nearest unpressed button if they're within range.

    The ground-truth contribution of agent i is: whether it pressed the correct
    button at the right time, weighted by how many buttons remained.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        n_agents: int = 5,
        line_length: float = 10.0,
        press_radius: float = 1.0,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.line_length = line_length
        self.press_radius = press_radius
        self.max_steps = max_steps

        # Observation: agent position(1) + velocity(1) + button positions(n_agents) + button states(n_agents)
        obs_dim = 2 + 2 * n_agents
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
            for i in range(n_agents)
        })

        # Each agent: move left (-1), move right (+1), stay (0), press button
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(4) for i in range(n_agents)
        })
        self.action_map = np.array([-0.5, 0.5, 0.0, 0.0])  # dx for actions 0,1,2; action 3 is press

        self.rng = np.random.RandomState(seed)
        self._reset_internal()

    def _reset_internal(self):
        # Randomize agent starting positions
        self.agent_positions = self.rng.uniform(0, self.line_length, self.n_agents)
        # Randomize button positions
        self.button_positions = np.sort(self.rng.uniform(0.5, self.line_length - 0.5, self.n_agents))
        self.button_pressed = np.zeros(self.n_agents, dtype=bool)
        self.next_button = 0  # Index of the next button that needs to be pressed
        self.steps = 0

        # Track ground-truth credit
        self.individual_credit = np.zeros(self.n_agents)

    def _get_obs(self):
        obs = {}
        for i in range(self.n_agents):
            obs_array = np.concatenate([
                [self.agent_positions[i]],
                [self._get_velocity()],
                self.button_positions,
                self.button_pressed.astype(np.float32),
            ]).astype(np.float32)
            obs[f"agent_{i}"] = obs_array
        return obs

    def _get_velocity(self):
        """Dummy velocity (can be extended to multi-step movement)."""
        return 0.0

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        self.steps += 1
        step_reward = 0.0
        successes_before = self.next_button
        info = {"individual_rewards": np.zeros(self.n_agents)}

        # First: apply movement actions
        for i in range(self.n_agents):
            action = actions[f"agent_{i}"]
            if action in [0, 1, 2]:  # Movement
                self.agent_positions[i] += self.action_map[action]
                self.agent_positions[i] = np.clip(self.agent_positions[i], 0, self.line_length)

        # Second: apply press actions (order matters for credit assignment)
        for i in range(self.n_agents):
            action = actions[f"agent_{i}"]
            if action == 3 and self.next_button < self.n_agents:  # Press
                target_button = self.next_button
                dist = abs(self.agent_positions[i] - self.button_positions[target_button])
                if dist <= self.press_radius:
                    # Correct button pressed at the right time
                    self.button_pressed[target_button] = True
                    self.individual_credit[i] += 1.0
                    self.next_button += 1
                    step_reward += 0.5  # Reward shaping for each correct press

        # Reward shaping: distance-based guidance toward the next target
        if self.next_button < self.n_agents:
            target_pos = self.button_positions[self.next_button]
            min_dist = min(abs(self.agent_positions[i] - target_pos) for i in range(self.n_agents))
            step_reward -= 0.01 * min_dist  # Encourage getting close to the next button

        # Time penalty to encourage efficiency
        step_reward -= 0.001

        # Check if all buttons pressed in correct order
        done = self.next_button == self.n_agents
        if done:
            step_reward += 1.0  # Terminal completion bonus

        terminated = done
        truncated = self.steps >= self.max_steps

        return self._get_obs(), step_reward, terminated, truncated, info

    def get_ground_truth_credit(self) -> np.ndarray:
        """Return ground-truth credit for evaluation."""
        return self.individual_credit

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._reset_internal()
        return self._get_obs(), {}

    def render(self):
        line = ["-"] * int(self.line_length * 4)
        for i, pos in enumerate(self.button_positions):
            idx = int(pos * 4)
            if self.button_pressed[i]:
                line[idx] = "X"
            else:
                line[idx] = "B"
        for i, pos in enumerate(self.agent_positions):
            idx = min(int(pos * 4), len(line) - 1)
            line[idx] = str(i)
        print(f"Step {self.steps}: {''.join(line)} | Next: {self.next_button}")
