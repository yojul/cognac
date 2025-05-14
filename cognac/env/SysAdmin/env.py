import functools

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward
from ...utils.graph_utils import to_stochastic_matrix
from .rewards import SysAdminDefaultReward


class SysAdminNetworkEnvironment(ParallelEnv):
    metadata = {
        "name": "sysadmin_environment_v0",
    }

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        max_steps: int = 100,
        show_neighborhood_state: bool = True,
        reward_class: BaseReward = SysAdminDefaultReward,
        is_global_reward: bool = False,
        base_arrival_rate: float = 0.5,
        base_fail_rate: float = 0.1,
        dead_rate_multiplier: float = 0.2,
        base_success_rate: float = 0.3,
        faulty_success_rate: float = 0.1,
    ):
        # Env properties
        self.adjacency_matrix = adjacency_matrix
        self.n_agents = adjacency_matrix.shape[0]
        self.possible_agents = list(range(self.n_agents))
        self.state = None
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

        # Probability of influencing action in [0,1]
        self.adjacency_matrix_prob = self.adjacency_matrix
        self._check_adjacency_matrix()

        # Add self influence w/ base fail rate paramater
        np.fill_diagonal(self.adjacency_matrix_prob, self.base_fail_rate)
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
        # Check that the diagonal is zero
        assert all(
            [self.adjacency_matrix_prob[i, i] == 0 for i in range(self.n_agents)]
        )
        # Check that influence probability are well defined
        assert np.all(self.adjacency_matrix_prob <= 1) and np.all(
            self.adjacency_matrix_prob >= 0
        )

    def reset(self, seed=None, options=None):
        self.agents = list(range(self.n_agents))
        self.timestep = 0
        self.state = np.zeros((self.n_agents, 2))

        initial_jobs = np.random.binomial(1, self.base_arrival_rate, size=self.n_agents)

        self.state[:, 1] = initial_jobs

        # Initialize observations for every agent using the initial state vector
        observations = self.get_obs()

        # Get dummy info. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):

        # STEP 0 : Launch reboot of machines as requested
        act_vect = np.array([act for act in actions.values()]).reshape((self.n_agents,))
        self.state[act_vect == 1, 0] = 0  # Set machines to reboot to working state
        self.state[act_vect == 1, 1] = 0

        # STEP 1 : Solve current task
        if np.any(self._working_loaded_mask()):
            self.state[self._working_loaded_mask(), 1] += np.random.binomial(
                1, p=self.base_success_rate, size=self._working_loaded_mask().sum()
            )
        if np.any(self._faulty_loaded_mask()):
            self.state[self._faulty_loaded_mask(), 1] += np.random.binomial(
                1, p=self.faulty_success_rate, size=self._faulty_loaded_mask().sum()
            )

        # STEP 2 : Compute rewards & reset machines where tasks are done
        # Terminations
        terminations = {agent: False for agent in range(self.n_agents)}
        is_done = False
        if self.timestep > self.max_steps:
            is_done = True
            terminations = {
                agent: True for agent in self.possible_agents
            }  # Consensus reached
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in range(self.n_agents)}
        is_truncated = False
        if self.timestep > self.max_steps:
            # rewards = self.get_final_reward(gain=10)
            is_truncated = True
            truncations = {a: True for a in self.possible_agents}
            self.agents = []

        # Compute rewards
        rewards = self.reward(
            actions, self, is_done, is_truncated, as_global=self.is_global_reward
        )
        # Reset states
        self.state[self._done_mask(), 1] = 0

        # STEP 3 : Draw new jobs for available machines
        if np.any(self._available_mask()):
            self.state[self._available_mask(), 1] += np.random.binomial(
                1,
                self.base_arrival_rate,
                size=self.state[self._available_mask(), 1].shape,
            )

        # STEP 4 : Randomly make machines faulty or dead with networked influence
        faulty_processes = np.random.binomial(
            1,
            np.sum(self.adjacency_matrix_prob[self._working_mask()], axis=1),
            size=self.adjacency_matrix_prob[self._working_mask()].shape[0],
        )
        self.state[self._working_mask(), 0] = faulty_processes

        dead_processes = np.random.binomial(
            1,
            self.dead_rate_multiplier
            * np.sum(self.adjacency_matrix_prob[self._faulty_working_mask()], axis=1),
            size=self.adjacency_matrix_prob[self._faulty_working_mask()].shape[0],
        )
        self.state[self._faulty_working_mask(), 0] = dead_processes

        self.timestep += 1
        observations = self.get_obs()

        infos = {agent: {} for agent in range(self.n_agents)}

        return observations, rewards, terminations, truncations, infos

    def _working_mask(self):
        return self.state[:, 0] == 0

    def _faulty_working_mask(self):
        return self.state[:, 0] != 0

    def _working_loaded_mask(self):
        return (self.state[:, 0] == 0) & (self.state[:, 1] == 1)

    def _faulty_loaded_mask(self):
        return (self.state[:, 0] == 1) & (self.state[:, 1] == 1)

    def _done_mask(self):
        return (self.state[:, 0] != 2) & (self.state[:, 1] == 2)

    def _available_mask(self):
        return (self.state[:, 1] == 0) & (self.state[:, 0] == 0)

    def get_obs(self) -> dict:
        """Get the observation dict for the multi-agent environment from
        the current state of the system.

        Returns:
            dict: Dictionnary of observations for each agent.
        """
        observations = {agent: self.state[agent] for agent in range(self.n_agents)}
        return observations

    def render(self):  # TODO
        print(self.state)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> MultiDiscrete:
        return MultiDiscrete(
            [3, 3]
        )  # ["good","faulty", "dead"] ["idle", "loaded", "successful"]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Discrete:
        return Discrete(2, start=0)  # ["do nothing", "reboot"]
