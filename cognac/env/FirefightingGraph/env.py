import functools
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv

from .rewards import DefaultFFGReward


class RowFireFightingGraphEnvironment(ParallelEnv):
    """
    A multi-agent environment where agents try to manage a row of houses in fire.
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
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Options for initialization,
                such as initial state vector. Defaults to None.

        Returns:
            tuple: Observations and information dictionary.
        """
        self.agents = self.possible_agents
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

        if options and "init_vect" in options:
            assert len(options["init_vect"]) == self.n_houses
            self.state = np.array(options["init_vect"])
        else:
            self.state = np.random.randint(0, self.max_fire_level, size=self.n_houses)

        observations = self.get_obs()
        infos = {
            agent: {"fireman_position": self.last_visited_house[agent]}
            for agent in range(self.n_agents)
        }
        self.reward.reset(self.state)
        return observations, infos

    def step(self, actions: dict) -> tuple:
        """
        Execute one step in the environment given agents' actions.

        Args:
            actions (dict): Dictionary mapping agent indices to their actions.

        Returns:
            tuple: Observations, rewards, terminations, truncations, and info.
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
        """
        Retrieve the observation dictionary for all agents.

        Returns:
            dict: Dictionary mapping agent indices to their respective observations.
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
        """
        Render the current state of the environment.

        Args:
            save_frame (bool, optional): Whether to save the current
                frame as an image. Defaults to False.
            fig (matplotlib.figure.Figure, optional): Figure for rendering.
                Defaults to None.
            ax (matplotlib.axes.Axes, optional): Axes for rendering.
                Defaults to None.
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
        """
        Define the observation space for a given agent.

        Args:
            agent (int): Agent index.

        Returns:
            Discrete: Observation space of the agent.
        """
        return Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> Discrete:
        """
        Define the action space for an agent.

        Args:
            agent (int): Agent index.

        Returns:
            Discrete: Action space (binary choice: 0 or 1).
        """
        return Discrete(2)


class GridFireFightingGraphEnvironment(ParallelEnv):
    """
    A multi-agent environment where agents try to manage a row of houses in fire.
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
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Options for initialization,
                such as initial state vector. Defaults to None.

        Returns:
            tuple: Observations and information dictionary.
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
        """
        Execute one step in the environment given agents' actions.

        Args:
            actions (dict): Dictionary mapping agent indices to their actions.

        Returns:
            tuple: Observations, rewards, terminations, truncations, and info.
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
        left = np.zeros_like(state, dtype=bool)
        right = np.zeros_like(state, dtype=bool)

        left[1:] = state[:-1] > 0  # left neighbor burning
        right[:-1] = state[1:] > 0  # right neighbor burning

        return left | right  # burning neighbor on either side

    def _is_burning(self) -> np.ndarray:
        return self.state > 0

    def _act_id_to_house_pos(self, agent: tuple, act_id: int) -> tuple:
        i, j = self._act_id_to_house_pos_dict[int(act_id)]
        return agent[0] + i, agent[1] + j

    def _get_visit_map(self, joint_act: dict) -> np.ndarray:
        visit_map = np.zeros_like(self.state)
        for agent, act in joint_act.items():
            visit_map[self._act_id_to_house_pos(agent, act)] += 1
        return visit_map

    def get_obs(self) -> dict:
        """
        Retrieve the observation dictionary for all agents.

        Returns:
            dict: Dictionary mapping agent indices to their respective observations.
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
        """
        Render the current state of the environment.

        Args:
            save_frame (bool, optional): Whether to save the current frame as an image.
                Defaults to False.
            fig (matplotlib.figure.Figure, optional): Figure for rendering.
                Defaults to None.
            ax (matplotlib.axes.Axes, optional): Axes for rendering.
                Defaults to None.
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
        """
        Define the observation space for a given agent.

        Args:
            agent (int): Agent index.

        Returns:
            Discrete: Observation space of the agent.
        """
        return Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: tuple) -> Discrete:
        """
        Define the action space for an agent.

        Args:
            agent (int): Agent index.

        Returns:
            Discrete: Action space (binary choice: 0 or 1).
        """
        return Discrete(4)
