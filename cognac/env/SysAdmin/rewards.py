# Copyright 2025 Jules Sintes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
