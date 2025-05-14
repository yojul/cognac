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

from ...core.BaseReward import BaseReward


class DefaultMCFReward(BaseReward):
    """
    Default reward function for the Multi Commodity Flow environment.
    The reward is the negative of the total cost of the flow.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, action, env, is_done, is_truncated, as_global=False):
        reward = {}
        for agent in env.possible_agents:
            reward[agent] = (
                sum(
                    [
                        1 / (data["weight"] ** 3) * data["flow"]
                        for _, _, data in env.network.out_edges(agent, data=True)
                    ]
                )
                / 1e4
            )
        return reward
