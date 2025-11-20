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

from pettingzoo import ParallelEnv


class BaseReward(ABC):
    """

    .. note::
        This is a core object for Reward instance.
        It is a work in progress to build a common basis for all
        networked based environments.
    """

    def __init__(self):
        self.min_reward = 0.0
        self.max_reward = 0.0

    @abstractmethod
    def __call__(
        self,
        action: dict,
        env: ParallelEnv,
        is_done: bool,
        is_truncated: bool,
        as_global: bool = False,
    ) -> dict[float]:
        """Abstract method for calling the reward and compute individual rewards for all
        agents in the system.

        :param action: _description_
        :type action: dict
        :param env: _description_
        :type env: ParallelEnv
        :param is_done: _description_
        :type is_done: bool
        :param is_truncated: _description_
        :type is_truncated: bool
        :param as_global: _description_, defaults to False
        :type as_global: bool, optional
        :return: _description_
        :rtype: dict[float]
        """

        return {agent: self.min_reward for agent in env.possible_agents}

    def _get_global_reward(self, local_rewards: dict[float]) -> float:
        """This method generate a shared reward value as the sum of individual local
        rewards. This method can be overwrite when creating a reward class for a
        specific env to get a particular behavior. E.g. average per agent reward or more
        complex aggregation for shared reward.

        :param local_rewards: The dict of individual local rewards per-agent.
        :type local_rewards: dict[float]
        :return: An aggregate shared reward as the sum of local rewards.
        :rtype: float
        """
        return sum(local_rewards.values())

    def reset(self, init_state):
        return
