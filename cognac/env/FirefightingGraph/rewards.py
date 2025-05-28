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
    """Default reward function for the Fire Fighting Graph environment.

    This reward penalizes agents based on the fire level at the house they last visited.
    The reward is negative proportional to the fire intensity, encouraging agents to
    extinguish fires efficiently.

    Parameters
    ----------
    max_reward : float, optional
        Maximum reward value, by default 0.0. (Currently unused in calculation but
        reserved for potential future use.)
    min_reward : float, optional
        Minimum reward value, by default 0.0. (Currently unused in calculation but
        reserved for potential future use.)

    Attributes
    ----------
    max_reward : float
        The maximum reward possible.
    min_reward : float
        The minimum reward possible.
    """

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
        """Compute the reward for all agents based on the current environment state.

        The reward for each agent is the negative fire level of
        the house they last visited.

        Parameters
        ----------
        action : dict
            Dictionary mapping each agent to its last taken action.
        env : ParallelEnv
            The environment instance providing current state and metadata.
        is_done : bool
            Flag indicating whether the episode has terminated.
        is_truncated : bool
            Flag indicating whether the episode has been truncated.
        as_global : bool, optional
            If True, returns a single global reward instead of per-agent rewards.
            Default is False.

        Returns
        -------
        dict of float
            Dictionary mapping agent identifiers to their respective rewards.
        """
        state = env.state
        reward = {
            agent: -state[house] for agent, house in env.last_visited_house.items()
        }
        return reward
