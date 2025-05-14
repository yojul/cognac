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
