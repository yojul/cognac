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

import numpy as np
from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward


class FactoredRewardModel(BaseReward):
    """Reward model that encourages agents to reach consensus with the majority.

    This reward model assigns rewards based on local agreement with the current
    majority state. Upon episode termination, a factored reward is computed
    based on the proportion of agents in consensus and the remaining time steps.

    This model supports both local (per-agent) and global reward configurations.

    Parameters
    ----------
    max_reward : float, optional
        Maximum reward achievable at full consensus. Default is 100.0.
    min_reward : float, optional
        Penalty applied when the episode is truncated before consensus.
        Default is -100.0.
    """

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
    ) -> dict[int, float]:
        """Compute the reward for all agents at the current step.

        Parameters
        ----------
        action : dict
            Dictionary of agent actions, not used directly in this model.
        env : ParallelEnv
            The current environment instance.
        is_done : bool
            Whether the episode has terminated.
        is_truncated : bool
            Whether the episode has been truncated due to max steps.
        as_global : bool, optional
            If True, returns a global reward replicated for each agent.

        Returns
        -------
        dict of float
            Mapping from agent ID to reward.
        """

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
        """Compute how many agents currently agree on the majority value.

        Parameters
        ----------
        env : ParallelEnv
            The current environment instance.

        Returns
        -------
        int
            Number of agents voting for the majority value.
        """
        state = env.state()
        return max(np.sum(state), np.sum(np.ones_like(state) - state))


class RewardWInitTarget(BaseReward):
    """Reward model encouraging agents to converge on the initial majority state.

    This reward model stores the initial majority value after reset, then provides:
    - A large positive reward if the final state reaches full consensus
    on the initial value.
    - A large penalty if consensus is not reached or if the consensus
    is on the wrong value.
    - Stepwise feedback (+1 or -1) during the episode based on
    agreement with the target.

    Parameters
    ----------
    max_reward : float, optional
        Reward for reaching full consensus on the initial majority value.
        Default is 10.0.
    min_reward : float, optional
        Penalty for failing to reach the correct consensus.
        Default is -10.0.
    """

    def __init__(self, max_reward: float = 10.0, min_reward: float = -10.0):
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.target_value = None

    def reset(self, init_state: np.ndarray):
        """Set the target consensus value from the initial state.

        Parameters
        ----------
        init_state : np.ndarray
            Initial binary vector of agent states.
        """
        self.target_value = np.argmax(np.bincount(init_state))

    def __call__(
        self,
        action: dict,
        env: ParallelEnv,
        is_done: bool,
        is_truncated: bool,
        as_global: bool = False,
    ) -> dict[int, float]:
        """Compute the reward for all agents based on agreement with the initial
        majority.

        Parameters
        ----------
        action : dict
            Dictionary of agent actions, not used directly.
        env : ParallelEnv
            The environment instance.
        is_done : bool
            Whether the episode has terminated.
        is_truncated : bool
            Whether the episode has been truncated.
        as_global : bool, optional
            Unused. This reward is always per-agent.

        Returns
        -------
        dict of float
            Mapping from agent ID to reward.
        """

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
