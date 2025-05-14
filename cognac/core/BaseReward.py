from abc import ABC, abstractmethod


class BaseReward(ABC):
    def __init__(self):
        self.min_reward = 0.0
        self.max_reward = 0.0

    @abstractmethod
    def __call__(self, action, env, is_done, is_truncated, as_global=False):
        return {agent: self.min_reward for agent in env.possible_agents}

    def _get_global_reward(self, local_rewards):
        return sum(local_rewards.values())

    def reset(self, init_state):
        return
