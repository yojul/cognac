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
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv

from .rewards import DefaultFFGReward

"""
Multi-agent fire fighting graph environments for PettingZoo.

These environments simulate scenarios in which agents coordinate
to extinguish fires in a linear (row) or grid layout of houses.
They are useful for studying coordination and distributed control
in multi-agent reinforcement learning.
"""


class RowFireFightingGraphEnvironment(ParallelEnv):
    """A row-based multi-agent fire fighting environment.

    Each agent is responsible for extinguishing fires in adjacent houses.
    Fire levels increase probabilistically based on neighboring fires
    and the presence of firefighters.

    Parameters
    ----------
    n_agents : int
        Number of agents.
    max_steps : int
        Maximum number of environment steps.
    max_fire_level : int
        Maximum fire level a house can reach.
    reward_class : type, optional
        Reward function class to compute rewards.
    is_global_reward : bool, optional
        Whether to use a shared global reward or individual rewards.

    Attributes
    ----------
    n_agents : int
        Number of agents.
    n_houses : int
        Number of houses (n_agents + 1).
    max_steps : int
        Maximum number of environment steps.
    max_fire_level : int
        Maximum fire level a house can reach.
    reward : DefaultFFGReward
        Reward function instance.
    is_global_reward : bool
        Whether to use a shared global reward.
    """

    metadata = {"name": "row_firefighting_environment_v0"}

    def __init__(
        self,
        n: int = 10,
        max_steps: int = 100,
        max_fire_level: int = 100,
        reward_class: type = DefaultFFGReward,
        is_global_reward: bool = False,
    ):

        self.n_agents = n
        self.n_houses = n + 1
        self.possible_agents = list(range(self.n_agents))
        self.state = np.empty(self.n_houses, dtype=int)
        self.timestep = None
        self.max_steps = max_steps
        self.max_fire_level = max_fire_level
        self.reward = reward_class()
        self.is_global_reward = is_global_reward
        self.rng = None

        # Fire level increase probabilities (for burning houses)
        self.p_fire_increase_no_firefighter = (
            0.4  # Probability fire level increases with no firefighter present
        )
        self.p_fire_increase_with_burning_neighbor = (
            0.7  # Higher probability if a neighbor is also burning
        )

        # Ignition probability (for non-burning houses)
        self.p_ignition_from_burning_neighbor = 0.3
        # Probability a non-burning house catches fire due to a burning neighbor

        # Fire extinguishing probabilities
        self.p_fire_extinguish_two_agents = 1.0  # Two agents fully extinguish the fire
        self.p_fire_extinguish_single_no_neighbors_burning = (
            1.0  # One agent extinguishes with prob 1 if no neighbors are burning
        )
        self.p_fire_extinguish_single_with_neighbors_burning = (
            0.6  # One agent extinguishes with prob 0.6 if neighbors are burning
        )

        # Flame observation probabilities
        self.p_observe_flames_not_burning = 0.2
        self.p_observe_flames_fire_level_1 = 0.5
        self.p_observe_flames_fire_level_greater_than_1 = 0.8

        # State/Action info
        self.last_visited_house = {agent: agent for agent in range(self.n_agents)}

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Resets the environment to an initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Initialization options, including an optional "init_vect"
            to specify initial fire levels.

        Returns
        -------
        tuple
            A dictionary of agent observations and a dictionary
            of additional info for each agent.
        """
        self.agents = self.possible_agents
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

        if options and "init_vect" in options:
            assert len(options["init_vect"]) == self.n_houses
            self.state = np.array(options["init_vect"])
        else:
            self.state = np.random.randint(
                0, self.max_fire_level + 1, size=self.n_houses
            )

        observations = self.get_obs()
        infos = {
            agent: {"fireman_position": self.last_visited_house[agent]}
            for agent in range(self.n_agents)
        }
        self.reward.reset(self.state)
        return observations, infos

    def step(self, actions: dict) -> tuple:
        """Executes a single step of environment dynamics.

        Parameters
        ----------
        actions : dict
            A mapping from agent identifiers to actions
            (0 to stay, 1 to move right).

        Returns
        -------
        tuple
            A 5-tuple containing:
            - observations (dict): Updated observations.
            - rewards (dict): Rewards for each agent.
            - terminations (dict): Episode termination status.
            - truncations (dict): Episode truncation status.
            - infos (dict): Additional per-agent information.
        """

        self.last_visited_house = {
            agent: int(agent + act) for agent, act in actions.items()
        }
        visited_house = Counter(self.last_visited_house.values())
        last_state = self.state.copy()

        for house in range(self.n_houses):
            if house not in visited_house:
                if self._is_burning_neighbors(house, last_state):
                    if self.state[house] == 0:
                        self.state[house] += self.rng.binomial(
                            1, self.p_ignition_from_burning_neighbor
                        )
                    else:
                        self.state[house] += self.rng.binomial(
                            1, self.p_fire_increase_with_burning_neighbor
                        )
                elif self.state[house] > 0:
                    self.state[house] += self.rng.binomial(
                        1, self.p_fire_increase_no_firefighter
                    )
            elif visited_house[house] > 1:
                self.state[house] = 0  # TODO : Add behavior if the proba is < 1.0
            else:
                if (
                    self._is_burning_neighbors(house, last_state)
                    and self.state[house] > 0
                ):
                    self.state[house] -= self.rng.binomial(
                        1, self.p_fire_extinguish_single_with_neighbors_burning
                    )
                elif (
                    not self._is_burning_neighbors(house, last_state)
                    and self.state[house] > 0
                ):
                    self.state[house] -= self.rng.binomial(
                        1, self.p_fire_extinguish_single_no_neighbors_burning
                    )
        self.state = np.clip(self.state, a_min=0, a_max=self.max_fire_level)

        terminations = {agent: False for agent in range(self.n_agents)}
        is_done = np.all(self.state == 0)
        if is_done:
            terminations = {agent: True for agent in self.agents}
            self.agents = []

        truncations = {a: False for a in range(self.n_agents)}
        if self.timestep > self.max_steps:
            truncations = {a: True for a in self.agents}
            self.agents = []

        rewards = self.reward(
            actions,
            self,
            is_done,
            self.timestep > self.max_steps,
            as_global=self.is_global_reward,
        )
        self.timestep += 1
        observations = self.get_obs()
        infos = {
            agent: {"fireman_position": self.last_visited_house[agent]}
            for agent in range(self.n_agents)
        }

        return observations, rewards, terminations, truncations, infos

    def _is_burning_neighbors(self, house: int, state: np.ndarray) -> bool:
        if house == 0:
            return state[house + 1] > 0
        elif house == state.shape[0] - 1:
            return state[house - 1] > 0
        else:
            return state[house - 1] > 0 or state[house + 1] > 0

    def get_obs(self) -> dict:
        """Computes the observable flame status for each agent's current location.

        Returns
        -------
        dict
            Observations per agent as binary flame detection (0 or 1).
        """
        obs = {}
        for agent in self.possible_agents:
            visited_house = self.last_visited_house[agent]
            if self.state[visited_house] == 0:
                obs[agent] = self.rng.binomial(
                    1, self.p_observe_flames_not_burning, size=(1,)
                )
            elif self.state[visited_house] == 1:
                obs[agent] = self.rng.binomial(
                    1, self.p_observe_flames_fire_level_1, size=(1,)
                )
            elif self.state[visited_house] > 1:
                obs[agent] = self.rng.binomial(
                    1, self.p_observe_flames_fire_level_greater_than_1, size=(1,)
                )

        return obs

    def render(self, save_frame: bool = False, fig=None, ax=None) -> None:
        """Renders the current fire level and agent positions on the graph.

        Parameters
        ----------
        save_frame : bool, optional
            If True, saves the rendered frame.
        fig : Figure, optional
            Matplotlib figure object.
        ax : Axes, optional
            Matplotlib axes object.
        """
        if fig is None:
            plt.figure(figsize=(6, 6))

        # Update node states
        for agent, state_value in enumerate(self.state):
            self.graph_rendering.nodes[agent]["state"] = state_value

        # Node colors based on state
        node_colors = [self.state[i] for i in self.graph_rendering.nodes]
        edge_colors = [
            self.influence_activation[i, j] for i, j in self.graph_rendering.edges()
        ]

        # Draw network
        nx.draw(
            self.graph_rendering,
            self.pos_layout,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            cmap=plt.cm.coolwarm,
            edge_cmap=plt.cm.Paired,
            node_size=500,
            ax=ax,
        )

        plt.title("Multi-Agent Network State")

        if save_frame:
            plt.savefig(f"frame_{len(self.frames)}.png")  # Save frame
            self.frames.append(f"frame_{len(self.frames)}.png")  # Store filename
        if ax is None:
            plt.show()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: int) -> Discrete:
        """Returns the observation space of a given agent.

        Parameters
        ----------
        agent : int
            Index of the agent.

        Returns
        -------
        Discrete
            Observation space with binary outcome (0 or 1).
        """
        return Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> Discrete:
        """Returns the action space of a given agent.

        Parameters
        ----------
        agent : int
            Index of the agent.

        Returns
        -------
        Discrete
            Action space with 2 actions (0 or 1).
        """
        return Discrete(2)


class GridFireFightingGraphEnvironment(ParallelEnv):
    """A grid-based multi-agent fire fighting environment.

    Each agent controls a cell in a grid and must cooperate
    with others to suppress fires in a shared neighborhood.
    Fire spread and extinguishing dynamics depend on spatial
    proximity and the actions of neighboring agents.

    Parameters
    ----------
    n_width : int
        Grid width.
    n_height : int
        Grid height.
    max_steps : int
        Maximum number of environment steps.
    max_fire_level : int
        Maximum fire level a house can reach.
    reward_class : type, optional
        Reward function class to compute rewards.
    is_global_reward : bool, optional
        Whether to use a shared global reward or individual rewards.

    Attributes
    ----------
    n_width : int
        Grid width.
    n_height : int
        Grid height.
    n_agents : int
        Total number of agents.
    n_houses : int
        Total number of houses.
    max_steps : int
        Maximum number of environment steps.
    max_fire_level : int
        Maximum fire level a house can reach.
    reward : DefaultFFGReward
        Reward function instance.
    is_global_reward : bool
        Whether to use a shared global reward.
    """

    metadata = {"name": "grid_firefighting_environment_v0"}

    def __init__(
        self,
        n_width: int = 5,
        n_height: int = 5,
        max_steps: int = 100,
        max_fire_level: int = 100,
        reward_class: type = DefaultFFGReward,
        is_global_reward: bool = False,
    ):

        self.n_width, self.n_heigth = n_width, n_height
        self.n_agents = n_height * n_width
        self.n_houses = (n_height + 1) * (n_width + 1)
        self.possible_agents = list(product(range(self.n_heigth), range(self.n_width)))
        self.state = np.empty((n_height + 1, n_width + 1), dtype=int)
        self.timestep = None
        self.max_steps = max_steps
        self.max_fire_level = max_fire_level
        self.reward = reward_class()
        self.is_global_reward = is_global_reward
        self.rng = None

        # Fire level increase probabilities (for burning houses)
        self.p_fire_increase_no_firefighter = (
            0.4  # Probability fire level increases with no firefighter present
        )
        self.p_fire_increase_with_burning_neighbor = (
            0.7  # Higher probability if a neighbor is also burning
        )

        # Ignition probability (for non-burning houses)
        self.p_ignition_from_burning_neighbor = 0.3
        # Probability a non-burning house catches fire due to a burning neighbor

        # Fire extinguishing probabilities
        self.p_fire_extinguish_two_agents = 1.0
        # Two agents fully extinguish the fire
        self.p_fire_extinguish_single_no_neighbors_burning = (
            1.0  # One agent extinguishes with prob 1 if no neighbors are burning
        )
        self.p_fire_extinguish_single_with_neighbors_burning = (
            0.6  # One agent extinguishes with prob 0.6 if neighbors are burning
        )

        # Flame observation probabilities
        self.p_observe_flames_not_burning = 0.2
        self.p_observe_flames_fire_level_1 = 0.5
        self.p_observe_flames_fire_level_greater_than_1 = 0.8

        # State/Action info
        self.last_visited_house = {(i, j): (i, j) for i, j in self.possible_agents}
        self._act_id_to_house_pos_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Resets the grid environment to an initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Initialization options including optional
            initial fire states under "init_vect".

        Returns
        -------
        tuple
            Observations and additional information for each agent.
        """
        self.agents = self.possible_agents
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

        if options and "init_vect" in options:
            assert len(options["init_vect"]) == self.state.shape
            self.state = np.array(options["init_vect"])
        else:
            self.state = np.random.randint(
                0, self.max_fire_level, size=self.state.shape
            )

        observations = self.get_obs()
        infos = {
            agent: {"fireman_position": self.last_visited_house[agent]}
            for agent in self.possible_agents
        }
        self.reward.reset(self.state)
        return observations, infos

    def step(self, actions: dict[tuple, int]) -> tuple:
        """Executes a step in the grid-based environment.

        Parameters
        ----------
        actions : dict of tuple to int
            Mapping from (row, col) agent indices to actions.

        Returns
        -------
        tuple
            Updated observations, rewards, terminations,
            truncations, and info dictionaries.
        """

        self.last_visited_house = {
            agent: self._act_id_to_house_pos(agent, act)
            for agent, act in actions.items()
        }
        visited_house = self._get_visit_map(actions)
        last_state = self.state.copy()

        # CASE 1 : Not visited, burning, burning neighbors
        mask_1 = (
            (visited_house == 0)
            & (self._is_burning_neighbors(last_state))
            & (self._is_burning())
        )
        self.state[mask_1] += self.rng.binomial(
            1, self.p_fire_increase_with_burning_neighbor, size=self.state[mask_1].shape
        )
        # CASE 2 : Not visited, not burning, burning neighbors
        mask_2 = (
            (visited_house == 0)
            & (self._is_burning_neighbors(last_state))
            & (~self._is_burning())
        )
        self.state[mask_2] += self.rng.binomial(
            1, self.p_ignition_from_burning_neighbor, size=self.state[mask_2].shape
        )

        # CASE 3 : Not visited, burning, no burning neighbors
        mask_3 = (
            (visited_house == 0)
            & (~self._is_burning_neighbors(last_state))
            & (self._is_burning())
        )
        self.state[mask_3] += self.rng.binomial(
            1, self.p_fire_increase_no_firefighter, size=self.state[mask_3].shape
        )

        # CASE 4 : Visited by 2
        mask_4 = visited_house == 2
        self.state[mask_4] = np.zeros_like(self.state[mask_4])

        # CASE 5 : Visited by 1, burning, burning neighbors
        mask_5 = (
            (visited_house == 1)
            & (self._is_burning())
            & self._is_burning_neighbors(last_state)
        )
        self.state[mask_5] -= self.rng.binomial(
            1,
            self.p_fire_extinguish_single_with_neighbors_burning,
            self.state[mask_5].shape,
        )

        # CASE 6 : Visited by 1, burning, no burning neighbors
        mask_6 = (
            (visited_house == 1)
            & (self._is_burning())
            & (~self._is_burning_neighbors(last_state))
        )
        self.state[mask_6] -= self.rng.binomial(
            1,
            self.p_fire_extinguish_single_no_neighbors_burning,
            self.state[mask_6].shape,
        )
        self.state = np.clip(self.state, a_min=0, a_max=self.max_fire_level)

        terminations = {agent: False for agent in self.possible_agents}
        is_done = np.all(self.state == 0)
        if is_done:
            terminations = {agent: True for agent in self.agents}
            self.agents = []

        truncations = {a: False for a in self.possible_agents}
        if self.timestep > self.max_steps:
            truncations = {a: True for a in self.agents}
            self.agents = []

        rewards = self.reward(
            actions,
            self,
            is_done,
            self.timestep > self.max_steps,
            as_global=self.is_global_reward,
        )
        self.timestep += 1
        observations = self.get_obs()
        infos = {
            agent: {"fireman_position": self.last_visited_house[agent]}
            for agent in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos

    def _is_burning_neighbors(self, state: np.ndarray) -> np.ndarray:
        """Determine which cells in a 2D grid have burning neighbors.

        Parameters
        ----------
        state : np.ndarray
            2D array of fire levels.

        Returns
        -------
        np.ndarray
            Boolean array indicating where at least one neighbor is burning.
        """
        up = np.zeros_like(state, dtype=bool)
        down = np.zeros_like(state, dtype=bool)
        left = np.zeros_like(state, dtype=bool)
        right = np.zeros_like(state, dtype=bool)

        up[1:, :] = state[:-1, :] > 0
        down[:-1, :] = state[1:, :] > 0
        left[:, 1:] = state[:, :-1] > 0
        right[:, :-1] = state[:, 1:] > 0

        return up | down | left | right

    def _is_burning_neighbors_1d(state: np.ndarray) -> np.ndarray:
        """Determine which houses in a 1D row have burning neighbors.

        .. warning:: Internal mask method use for the step dynamics.

        Parameters
        ----------
        state : np.ndarray
            1D array of fire levels.

        Returns
        -------
        np.ndarray
            Boolean array indicating where a neighbor is burning.
        """
        left = np.zeros_like(state, dtype=bool)
        right = np.zeros_like(state, dtype=bool)

        left[1:] = state[:-1] > 0  # left neighbor burning
        right[:-1] = state[1:] > 0  # right neighbor burning

        return left | right  # burning neighbor on either side

    def _is_burning(self) -> np.ndarray:
        """Identify which houses are currently burning.

        .. warning:: Internal mask method use for the step dynamics.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates a burning house.
        """
        return self.state > 0

    def _act_id_to_house_pos(self, agent: tuple, act_id: int) -> tuple:
        """Map an action ID to the house position relative to an agent.

        .. warning:: Internal mask method use for the step dynamics.

        Parameters
        ----------
        agent : tuple
            Agent's grid position.
        act_id : int
            Action identifier.

        Returns
        -------
        tuple
            The corresponding house position in the grid.
        """
        i, j = self._act_id_to_house_pos_dict[int(act_id)]
        return agent[0] + i, agent[1] + j

    def _get_visit_map(self, joint_act: dict) -> np.ndarray:
        visit_map = np.zeros_like(self.state)
        for agent, act in joint_act.items():
            visit_map[self._act_id_to_house_pos(agent, act)] += 1
        return visit_map

    def get_obs(self) -> dict:
        """Compute the number of visits each house receives from all agents.

        Parameters
        ----------
        joint_act : dict
            Mapping from agent positions to their chosen action.

        Returns
        -------
        np.ndarray
            A 2D array indicating the number of visits per house.
        """
        obs = {}
        for agent in self.possible_agents:
            visited_house = self.last_visited_house[agent]
            if self.state[visited_house] == 0:
                obs[agent] = self.rng.binomial(
                    1, self.p_observe_flames_not_burning, size=(1,)
                )
            elif self.state[visited_house] == 1:
                obs[agent] = self.rng.binomial(
                    1, self.p_observe_flames_fire_level_1, size=(1,)
                )
            elif self.state[visited_house] > 1:
                obs[agent] = self.rng.binomial(
                    1, self.p_observe_flames_fire_level_greater_than_1, size=(1,)
                )

        return obs

    def render(self, save_frame: bool = False, fig=None, ax=None) -> None:
        """Render the current state of the environment.

        Parameters
        ----------
        save_frame : bool, optional
            Whether to save the current frame as an image. Default is False.
        fig : matplotlib.figure.Figure, optional
            Figure object to use for rendering. If None, a new figure is created.
        ax : matplotlib.axes.Axes, optional
            Axes object to render on. If None, rendering occurs on the current axes.
        """

        if fig is None:
            plt.figure(figsize=(6, 6))

        # Update node states
        for agent, state_value in enumerate(self.state):
            self.graph_rendering.nodes[agent]["state"] = state_value

        # Node colors based on state
        node_colors = [self.state[i] for i in self.graph_rendering.nodes]
        edge_colors = [
            self.influence_activation[i, j] for i, j in self.graph_rendering.edges()
        ]

        # Draw network
        nx.draw(
            self.graph_rendering,
            self.pos_layout,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            cmap=plt.cm.coolwarm,
            edge_cmap=plt.cm.Paired,
            node_size=500,
            ax=ax,
        )

        plt.title("Multi-Agent Network State")

        if save_frame:
            plt.savefig(f"frame_{len(self.frames)}.png")  # Save frame
            self.frames.append(f"frame_{len(self.frames)}.png")  # Store filename
        if ax is None:
            plt.show()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: tuple) -> Discrete:
        """Define the observation space for a given agent.

        Parameters
        ----------
        agent : tuple
            Agent identifier.

        Returns
        -------
        gymnasium.spaces.Discrete
            Observation space of the agent (binary outcomes).
        """
        return Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: tuple) -> Discrete:
        """Define the action space for a given agent.

        Parameters
        ----------
        agent : tuple
            Agent identifier.

        Returns
        -------
        gymnasium.spaces.Discrete
            Action space of the agent (4 discrete actions).
        """
        return Discrete(4)
