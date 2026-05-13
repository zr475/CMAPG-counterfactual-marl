"""PettingZoo MPE (Multi-Particle Environment) wrapper.

Adapts PettingZoo MPE parallel_env to the gymnasium.Env interface expected by
the training framework. MPE is a standard MARL benchmark used by MADDPG, QMIX,
MAPPO, and many other papers.
"""

from typing import Dict, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MPEEnvWrapper(gym.Env):
    """Gymnasium wrapper for PettingZoo MPE parallel environments.

    Supported scenarios:
        - simple_spread_v3: N agents spread to cover N landmarks (cooperative)
        - simple_push_v3:  2 agents push a ball to a target (mixed, but
                           when run cooperatively both push the ball)

    Args:
        scenario: MPE scenario name (e.g. "simple_spread_v3").
        n_agents: Number of agents (for scenarios that accept N parameter).
        max_cycles: Maximum steps per episode.
        continuous_actions: If True, use continuous action space variant.
        seed: Random seed.
        local_ratio: Fraction of local vs global observations (0=fully local).
    """

    def __init__(
        self,
        scenario: str = "simple_spread_v3",
        n_agents: int = 3,
        max_cycles: int = 100,
        continuous_actions: bool = False,
        seed: Optional[int] = None,
        local_ratio: float = 0.5,
    ):
        super().__init__()
        self._scenario = scenario
        self._n_agents = n_agents
        self._max_cycles = max_cycles
        self._continuous = continuous_actions
        self._seed = seed
        self._local_ratio = local_ratio

        self._env = self._make_env()
        self._raw_agent_ids = self._env.possible_agents
        self.n_agents = len(self._raw_agent_ids)
        # Remap arbitrary agent IDs to standard agent_0, agent_1, ...
        self._agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self._id_map = dict(zip(self._raw_agent_ids, self._agent_ids))
        self._reverse_map = dict(zip(self._agent_ids, self._raw_agent_ids))

        # Build observation/action spaces with standardized agent IDs
        raw_obs_sample = self._env.observation_space(self._raw_agent_ids[0])
        raw_act_sample = self._env.action_space(self._raw_agent_ids[0])

        self.observation_space = spaces.Dict({
            aid: spaces.Box(
                low=raw_obs_sample.low if hasattr(raw_obs_sample, "low") else -np.inf,
                high=raw_obs_sample.high if hasattr(raw_obs_sample, "high") else np.inf,
                shape=raw_obs_sample.shape,
                dtype=np.float32,
            )
            for aid in self._agent_ids
        })

        if isinstance(raw_act_sample, spaces.Discrete):
            self.action_space = spaces.Dict({
                aid: spaces.Discrete(raw_act_sample.n) for aid in self._agent_ids
            })
        else:
            self.action_space = spaces.Dict({
                aid: spaces.Box(
                    low=raw_act_sample.low,
                    high=raw_act_sample.high,
                    shape=raw_act_sample.shape,
                    dtype=np.float32,
                )
                for aid in self._agent_ids
            })

        self.metadata = {"render_modes": ["human"], "render_fps": 30}
        self._render_mode = None
        self._last_obs = None

    def _make_env(self):
        """Create the underlying MPE parallel environment."""
        mod = self._import_scenario()
        kw = dict(max_cycles=self._max_cycles, render_mode=None)
        if self._scenario == "simple_spread_v3":
            kw["N"] = self._n_agents
            kw["local_ratio"] = self._local_ratio
            if self._continuous:
                kw["continuous_actions"] = True
        elif self._scenario == "simple_push_v3":
            if self._continuous:
                kw["continuous_actions"] = True
        return mod.parallel_env(**kw)

    def _import_scenario(self):
        import importlib
        return importlib.import_module(f"pettingzoo.mpe.{self._scenario}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed

        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

        self._env = self._make_env()
        self._raw_agent_ids = self._env.possible_agents
        self._id_map = dict(zip(self._raw_agent_ids, self._agent_ids))
        self._reverse_map = dict(zip(self._agent_ids, self._raw_agent_ids))

        raw_obs, info = self._env.reset(seed=self._seed)
        # Remap obs keys to standard agent_0, agent_1, ...
        self._last_obs = {self._id_map[aid]: obs for aid, obs in raw_obs.items()}
        return self._last_obs.copy(), {}

    def step(self, actions: Dict[str, np.ndarray]):
        # Remap standard keys back to raw agent IDs
        raw_actions = {self._reverse_map[aid]: act for aid, act in actions.items()}
        raw_obs, rewards_dict, terms, truncs, infos = self._env.step(raw_actions)
        # Remap obs back to standard keys
        self._last_obs = {self._id_map[aid]: obs for aid, obs in raw_obs.items()}

        reward = float(np.mean([rewards_dict[aid] for aid in self._raw_agent_ids]))

        terminated = any(terms.values()) if isinstance(terms, dict) else bool(terms)
        truncated = any(truncs.values()) if isinstance(truncs, dict) else bool(truncs)

        return self._last_obs.copy(), reward, terminated, truncated, {}

    def render(self):
        if self._render_mode == "human":
            try:
                self._env.render()
            except Exception:
                pass

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def get_state(self) -> np.ndarray:
        """Return global state as concatenation of all agent observations."""
        if self._last_obs is None:
            return np.zeros(sum(
                self.observation_space[aid].shape[0] for aid in self._agent_ids
            ), dtype=np.float32)
        return np.concatenate([self._last_obs[aid] for aid in self._agent_ids])

    @property
    def max_steps(self):
        return self._max_cycles
