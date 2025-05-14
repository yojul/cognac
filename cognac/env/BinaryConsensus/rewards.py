import numpy as np
from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward


class FactoredRewardModel(BaseReward):
    def __init__(self, max_reward: float = 100.0, min_reward: float = -100.0):
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.target_value = None

    def __call__(
        self,
        action: dict,
        env: ParallelEnv,
        is_done: bool,
        is_truncated: bool,
        as_global: bool = False,
    ) -> dict[float]:
        if is_done or is_truncated:
            malus = self.min_reward if is_truncated else 0.0
            temporal_factor = (env.max_steps - env.timestep) / env.max_steps
            consensus_value = self.max_reward * (
                self.get_consensus_value(env) / env.n_agents
            )
            reward = {
                agent: temporal_factor * (consensus_value + malus) / env.n_agents
                for agent in range(env.n_agents)
            }

        else:
            maj = env.get_majority_value()
            reward = {
                agent: 1.0 if maj == env.state()[agent] else -1.0
                for agent in range(env.n_agents)
            }
        return (
            reward
            if not as_global
            else {
                agent: self._get_global_reward(reward) for agent in env.possible_agents
            }
        )

    def get_consensus_value(self, env: ParallelEnv) -> int:
        """
        Returns:
            int : Current number of majority vote
        """
        state = env.state()
        return max(np.sum(state), np.sum(np.ones_like(state) - state))


class RewardWInitTarget(BaseReward):
    def __init__(self, max_reward: float = 10.0, min_reward: float = -10.0):
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.target_value = None

    def reset(self, init_state: np.ndarray):
        self.target_value = np.argmax(np.bincount(init_state))

    def __call__(
        self,
        action: dict,
        env: ParallelEnv,
        is_done: bool,
        is_truncated: bool,
        as_global: bool = False,
    ) -> dict[float]:
        if is_done or is_truncated:
            if np.all(env.state == self.target_value):
                # Consensus from initial majority reached
                reward = {agent: self.max_reward for agent in env.possible_agents}
            else:
                # Consensus not reached
                reward = {agent: self.min_reward for agent in env.possible_agents}
        else:
            # Intermediate reward to guide the learning
            reward = {
                agent: 1.0 if self.target_value == env.state[agent] else -1.0
                for agent in range(env.n_agents)
            }

        return reward
