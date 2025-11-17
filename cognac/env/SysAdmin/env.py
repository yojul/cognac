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

import functools

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward
from ...utils.graph_utils import to_stochastic_matrix
from .rewards import SysAdminDefaultReward


class SysAdminNetworkEnvironment(ParallelEnv):
    """Multi-agent environment simulating a network of machines managed by agents, based
    on the "SysAdmin" problem setting.

    Each agent controls one machine, which can be in states representing its
    operational condition (good, faulty, dead) and job state (idle, loaded, successful).
    Machines can influence each other’s failure rates based on a given network adjacency
    matrix.

    The environment follows a discrete timestep progression with reboot actions,
    task completions, faults propagation, and rewards computed per step.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Square matrix of shape (n_agents, n_agents) representing network connections.
        Entry (i, j) indicates influence of agent j on agent i.
        Must have zeros on diagonal initially (no self influence).
    max_steps : int, default=100
        Maximum number of timesteps before the environment is terminated.
    show_neighborhood_state : bool, default=True
        If True, agents observe not only their own state but also their
        neighbors' states.
    reward_class : BaseReward subclass, default=SysAdminDefaultReward
        Class to compute the rewards. Should be derived from BaseReward.
    is_global_reward : bool, default=False
        If True, a single global reward is returned to all agents.
    base_arrival_rate : float, default=0.5
        Probability of new job arriving for an available machine at each step.
    base_fail_rate : float, default=0.1
        Base failure rate for machines without external influence.
    dead_rate_multiplier : float, default=0.2
        Multiplier for probability of machine becoming dead influenced
        by faulty neighbors.
    base_success_rate : float, default=0.3
        Probability that a working loaded machine completes its task successfully.
    faulty_success_rate : float, default=0.1
        Probability that a faulty loaded machine completes its task successfully.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        Original adjacency matrix representing the network structure.
    adjacency_matrix_prob : np.ndarray
        Processed stochastic matrix of influence probabilities with
        self-failures included.
    n_agents : int
        Number of agents (machines) in the environment.
    possible_agents : list of int
        List of all agent IDs.
    state : np.ndarray
        Current state array of shape (n_agents, 2) with machine status
        and job state.
    timestep : int
        Current timestep counter.
    max_steps : int
        Maximum allowed timesteps before termination.
    reward : BaseReward
        Instance of the reward class used to calculate rewards.
    is_global_reward : bool
        Flag indicating if rewards are global or individual.
    neighboring_masks : np.ndarray (bool)
        Boolean mask matrix indicating which agents observe each other.

    Methods
    -------
    reset(seed=None, options=None)
        Reset environment state and return initial observations.
    step(actions)
        Perform one timestep given agents' actions; update state and return results.
    get_obs()
        Get the current observations dictionary for all agents.
    render()
        Print the current state of the environment (to be implemented).
    observation_space(agent)
        Return the observation space object for the specified agent.
    action_space(agent)
        Return the action space object for the specified agent.

    .. warning::
        Methods prefixed with an underscore (`_`) are for internal use only and
        should not be called directly by users.
    """

    metadata = {
        "name": "sysadmin_environment_v0",
    }

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        max_steps: int = 100,
        show_neighborhood_state: bool = False,
        reward_class: BaseReward = SysAdminDefaultReward,
        is_global_reward: bool = False,
        base_arrival_rate: float = 0.2,
        base_fail_rate: float = 0.15,
        dead_rate_multiplier: float = 0.05,
        base_success_rate: float = 0.3,
        faulty_success_rate: float = 0.2,
    ):
        # Env properties
        self.adjacency_matrix = adjacency_matrix.copy()
        self.n_agents = adjacency_matrix.shape[0]
        self.possible_agents = list(range(self.n_agents))
        self.agents = None
        self._state = None
        self.timestep = None
        self.max_steps = max_steps
        self.reward = reward_class()
        self.is_global_reward = is_global_reward

        # Base parameters
        self.base_arrival_rate = base_arrival_rate
        self.base_fail_rate = base_fail_rate
        self.base_success_rate = base_success_rate
        self.faulty_success_rate = faulty_success_rate
        self.dead_rate_multiplier = dead_rate_multiplier
        self.base_dead_rate = self.dead_rate_multiplier * self.base_fail_rate

        # Probability of influencing action in [0,1]
        self.adjacency_matrix_prob = self.adjacency_matrix.copy()
        self._check_adjacency_matrix()
        # Add self influence w/ base fail rate paramater
        np.fill_diagonal(self.adjacency_matrix_prob, self.base_fail_rate)
        self._scale_adjacency_matrix()

        self.neighboring_masks = np.identity(self.n_agents, dtype=bool)
        self.adjacency_matrix_prob = (
            to_stochastic_matrix(self.adjacency_matrix_prob) - 1e-4
        )
        if (
            show_neighborhood_state
        ):  # Fill the neighboring mask to enable visibility of neighbors' states
            self.filled_influence = adjacency_matrix.copy()
            self.neighboring_masks = self.neighboring_masks + (
                self.filled_influence != 0
            )

    def _check_adjacency_matrix(self):
        """
        .. warning::
            Internal use.

        Validates the adjacency matrix to ensure:
        - Diagonal elements are zero (no self influence initially).
        - All entries are probabilities in [0, 1].

        Raises
        ------
        AssertionError
            If any of the validation checks fail.
        """
        # Check that the diagonal is zero
        assert all(
            [self.adjacency_matrix_prob[i, i] == 0 for i in range(self.n_agents)]
        )
        # Check that influence probability are well defined
        assert np.all(self.adjacency_matrix_prob <= 1) and np.all(
            self.adjacency_matrix_prob >= 0
        )

    def _scale_adjacency_matrix(self):
        """
        .. warning::
            Internal use.

        Scales the adjacency_matrix_prob so that the maximum row or
        column sum is at most 1.
        This ensures the resulting matrix can be interpreted as a
        stochastic influence
        matrix.
        """
        row_sums = self.adjacency_matrix_prob.sum(axis=1)
        max_row_sum = np.max(row_sums)

        if max_row_sum > 1.0:
            self.adjacency_matrix_prob = (
                self.adjacency_matrix_prob / max_row_sum
            )  # scale everything so max row sum is 1.0
        col_sums = self.adjacency_matrix_prob.sum(axis=0)
        max_col_sum = np.max(col_sums)
        if max_col_sum > 1.0:
            self.adjacency_matrix_prob = (
                self.adjacency_matrix_prob / max_col_sum
            )  # scale everything so max row sum is 1.0

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the random number generator for reproducibility.
        options : dict or None, optional
            Additional options for environment reset (currently unused).

        Returns
        -------
        observations : dict
            Dictionary mapping agent IDs to their initial observations.
        infos : dict
            Dictionary mapping agent IDs to info dictionaries
            (empty in this implementation).
        """
        self.agents = list(range(self.n_agents))
        self.timestep = 0
        self._state = np.zeros((self.n_agents, 2))

        initial_jobs = np.random.binomial(1, self.base_arrival_rate, size=self.n_agents)

        self._state[:, 1] = initial_jobs

        # Initialize observations for every agent using the initial state vector
        observations = self.get_obs()

        # Get dummy info. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Advance the environment by one timestep given agents' actions.

        Parameters
        ----------
        actions : dict
            Dictionary mapping agent IDs to their actions. Each action is an integer:
            0 for "do nothing", 1 for "reboot".

        Returns
        -------
        observations : dict
            Dictionary mapping agent IDs to their new observations.
        rewards : dict
            Dictionary mapping agent IDs to their rewards for this step.
        terminations : dict
            Dictionary mapping agent IDs to termination flags (bool).
        truncations : dict
            Dictionary mapping agent IDs to truncation flags (bool).
        infos : dict
            Dictionary mapping agent IDs to info dictionaries (empty in this
            implementation).

        Notes
        -----
        - Rebooted machines are reset to working and idle state.
        - Machines working on tasks may complete successfully based on probabilities.
        - Faulty and dead states evolve probabilistically influenced by network
            neighbors.
        - Rewards are computed via the configured reward class.
        - Environment terminates after `max_steps`.
        """

        # STEP 0 : Launch reboot of machines as requested
        act_vect = np.array([act for act in actions.values()]).reshape((self.n_agents,))
        self._state[act_vect == 1, 0] = 0  # Set machines to reboot to working state
        self._state[act_vect == 1, 1] = 0

        # STEP 1 : Solve current task
        if np.any(self._working_loaded_mask()):
            self._state[self._working_loaded_mask(), 1] += np.random.binomial(
                1, p=self.base_success_rate, size=self._working_loaded_mask().sum()
            )
        if np.any(self._faulty_loaded_mask()):
            self._state[self._faulty_loaded_mask(), 1] += np.random.binomial(
                1, p=self.faulty_success_rate, size=self._faulty_loaded_mask().sum()
            )

        # STEP 2 : Compute rewards & reset machines where tasks are done
        # Terminations
        terminations = {agent: False for agent in range(self.n_agents)}
        is_done = False
        if self.timestep >= self.max_steps - 1:
            is_done = True
            terminations = {
                agent: True for agent in self.possible_agents
            }  # Consensus reached
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in range(self.n_agents)}
        is_truncated = False
        if self.timestep >= self.max_steps - 1:
            # rewards = self.get_final_reward(gain=10)
            is_truncated = True
            truncations = {a: True for a in self.possible_agents}
            self.agents = []

        # Compute rewards
        rewards = self.reward(
            actions, self, is_done, is_truncated, as_global=self.is_global_reward
        )
        # Reset states
        self._state[self._done_mask(), 1] = 0

        # STEP 3 : Draw new jobs for available machines
        if np.any(self._available_mask()):
            self._state[self._available_mask(), 1] += np.random.binomial(
                1,
                self.base_arrival_rate,
                size=self._state[self._available_mask(), 1].shape,
            )

        working_mask = self._working_mask()
        faulty_mask = self._faulty_working_mask()

        # STEP 4 : Faulty/Dead states propagation
        # --- Working → Faulty transition ---
        if np.any(working_mask):
            if np.any(faulty_mask):
                # Probability each working node stays healthy
                healthy_prob = np.prod(
                    1 - self.adjacency_matrix_prob[working_mask][:, faulty_mask],
                    axis=1,
                )
                p_infected_from_neighbors = 1 - healthy_prob
            else:
                p_infected_from_neighbors = np.zeros(working_mask.sum())

            # Combine base failure rate and network infection independently
            p_fault = 1 - (1 - self.base_fail_rate) * (1 - p_infected_from_neighbors)
            p_fault = np.clip(p_fault, 0.0, 1.0)

            # Apply Bernoulli process
            faulty_processes = np.random.binomial(1, p_fault, size=p_fault.shape[0])
            self._state[working_mask, 0] += faulty_processes

        # --- Faulty → Dead transition ---
        if np.any(faulty_mask):
            # Probability faulty node stays alive despite other faulty/dead neighbors
            healthy_prob_for_dead = np.prod(
                1 - self.adjacency_matrix_prob[faulty_mask][:, faulty_mask],
                axis=1,
            )
            p_dead_from_neighbors = 1 - healthy_prob_for_dead

            # Combine base death rate and propagation independently
            # (dead_rate_multiplier scales the strength of propagation)
            p_dead = 1 - (1 - self.base_dead_rate) * (
                1 - self.dead_rate_multiplier * p_dead_from_neighbors
            )
            p_dead = np.clip(p_dead, 0.0, 1.0)

            # Apply Bernoulli process
            dead_processes = np.random.binomial(1, p_dead, size=p_dead.shape[0])
            self._state[faulty_mask, 0] += dead_processes

        # --- Clip states to allowed range {0=working, 1=faulty, 2=dead} ---
        self._state[:, 0] = np.clip(self._state[:, 0], 0, 2)

        self.timestep += 1
        observations = self.get_obs()

        infos = {agent: {} for agent in range(self.n_agents)}

        return observations, rewards, terminations, truncations, infos

    def _working_mask(self):
        """
        .. warning::
            Internal use.

        Boolean mask for machines currently in working state.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates the machine is working (state code 0).
        """

        return self._state[:, 0] == 0

    def _faulty_working_mask(self):
        """
        .. warning::
            Internal use.

        Boolean mask for machines currently in faulty or dead state.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates machine is faulty or dead
            (state code != 0).
        """
        return self._state[:, 0] != 0

    def _working_loaded_mask(self):
        """
        .. warning::
            Internal use.

        Boolean mask for machines that are working and currently loaded with a job.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates working and loaded machines.
        """
        return (self._state[:, 0] == 0) & (self._state[:, 1] == 1)

    def _faulty_loaded_mask(self):
        """
        .. warning::
            Internal use.

        Boolean mask for machines that are faulty and currently loaded with a job.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates faulty and loaded machines.
        """
        return (self._state[:, 0] == 1) & (self._state[:, 1] == 1)

    def _done_mask(self):
        """
        .. warning::
            Internal use.

        Boolean mask for machines that have completed their jobs.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates machines done with their tasks.
        """
        return (self._state[:, 0] != 2) & (self._state[:, 1] == 2)

    def _available_mask(self):
        """
        .. warning::
            Internal use.

        Boolean mask for machines that are available to receive new jobs
        (working and currently idle).

        Returns
        -------
        np.ndarray
            Boolean array where True indicates available machines.
        """
        return (self._state[:, 1] == 0) & (self._state[:, 0] == 0)

    def get_obs(self) -> dict:
        """Retrieve the current observation for each agent.

        Observations consist of each agent's own machine state vector.

        Returns
        -------
        dict
            Mapping from agent IDs to their observations
            (np.ndarray of shape (2,)).
        """
        observations = {
            agent: self._state[self.neighboring_masks[agent]]
            for agent in range(self.n_agents)
        }
        return observations

    def render(self):
        """Render the current environment state.

        Currently prints the raw state array.

        Notes
        -----
        This method is a placeholder and should be implemented to provide
        a graphical or structured visualization of the environment.
        """
        print(self._state)

    def state(self):
        return self._state

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> MultiDiscrete:
        """Return the observation space for the given agent.

        The observation space is a MultiDiscrete space describing machine status
        and job state with discrete values:

        - Machine status: 0=good, 1=faulty, 2=dead
        - Job state: 0=idle, 1=loaded, 2=successful

        Parameters
        ----------
        agent : int
            Agent identifier.

        Returns
        -------
        gymnasium.spaces.MultiDiscrete
            Observation space for the agent.
        """
        return MultiDiscrete([3, 3])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Discrete:
        """Return the action space for the given agent.

        Actions are discrete:

        - 0: do nothing
        - 1: reboot machine

        Parameters
        ----------
        agent : int
            Agent identifier.

        Returns
        -------
        gymnasium.spaces.Discrete
            Action space for the agent.
        """
        return Discrete(2, start=0)
