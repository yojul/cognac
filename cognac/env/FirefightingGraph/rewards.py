from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward


class DefaultFFGReward(BaseReward):
    def __init__(self, max_reward: float = 0.0, min_reward: float = 0.0):
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(
        self,
        action: dict,
        env: ParallelEnv,
        is_done: bool,
        is_truncated: bool,
        as_global: bool = False,
    ) -> dict[float]:
        state = env.state
        reward = {
            agent: -state[house] for agent, house in env.last_visited_house.items()
        }
        return reward
