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
