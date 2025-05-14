from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward


class SysAdminDefaultReward(BaseReward):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        action: dict,
        env: ParallelEnv,
        is_done: bool,
        is_truncated: bool,
        as_global: bool = False,
    ):
        done_mask = env._done_mask()
        rewards = {agent: int(done_mask[agent]) for agent in env.possible_agents}
        return rewards if not as_global else self._get_global_reward(rewards)
