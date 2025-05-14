import functools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from ...utils.graph_utils import generate_coordination_graph
from .rewards import FactoredRewardModel


class BinaryConsensusNetworkEnvironment(ParallelEnv):
    """
    A multi-agent consensus environment where agents attempt to reach a binary consensus
    based on an influence graph.
    """

    metadata = {"name": "binary_consensus_environment_v0"}

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        max_steps: int = 100,
        show_neighborhood_state: bool = True,
        reward_class: type = FactoredRewardModel,
        is_global_reward: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            adjacency_matrix (np.ndarray): Adjacency matrix.
            max_steps (int, optional): Maximum number of steps before truncation.
                Defaults to 100.
            show_neighborhood_state (bool, optional): If True, agents can observe
                neighbors' states. Defaults to True.
            reward_class (type, optional): Reward model class. Defaults to
                FactoredRewardModel.
            is_global_reward (bool, optional): If True, uses a global reward function.
                Defaults to False.
        """

        self.adjacency_matrix = adjacency_matrix
        self.n_agents = adjacency_matrix.shape[0]
        self.possible_agents = list(range(self.n_agents))
        self._state = None
        self.timestep = None
        self.max_steps = max_steps
        self.reward = reward_class()
        self.is_global_reward = is_global_reward
        self.influence_activation = np.zeros((self.n_agents, self.n_agents), dtype=bool)

        self.influence_sgn = np.sign(self.adjacency_matrix)
        self.adjacency_matrix_prob = np.abs(self.adjacency_matrix)
        self._check_adjacency_matrix()

        self.neighboring_masks = np.identity(self.n_agents, dtype=bool)
        if show_neighborhood_state:
            self.filled_influence = adjacency_matrix.copy()
            self.neighboring_masks = self.neighboring_masks + (
                self.filled_influence != 0
            )

        self.graph_rendering = generate_coordination_graph(self.adjacency_matrix)
        self.pos_layout = nx.circular_layout(self.graph_rendering)

        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }
        self.state_space = MultiDiscrete([2] * self.n_agents)

    def _check_adjacency_matrix(self) -> None:
        """
        Validate the influence graph by ensuring its diagonal is zero
        and that probabilities are in the range [0,1].
        """
        assert all(
            [self.adjacency_matrix_prob[i, i] == 0 for i in range(self.n_agents)]
        )
        assert np.all(self.adjacency_matrix_prob <= 1) and np.all(
            self.adjacency_matrix_prob >= 0
        )

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
        self.agents = list(range(self.n_agents))
        self.timestep = 0

        if options and "init_vect" in options:
            assert len(options["init_vect"]) == self.n_agents
            self._state = np.array(options["init_vect"], dtype=float)
        else:
            self._state = np.random.randint(0, 2, size=self.n_agents).astype(float)
            while np.all(self._state == 0) or np.all(self._state == 1):
                self._state = np.random.randint(0, 2, size=self.n_agents).astype(float)

        observations = self.get_obs()
        infos = {a: {} for a in self.agents}
        self.reward.reset(self._state)
        return observations, infos

    def step(self, actions: dict) -> tuple:
        """
        Execute one step in the environment given agents' actions.

        Args:
            actions (dict): Dictionary mapping agent indices to their actions.

        Returns:
            tuple: Observations, rewards, terminations, truncations, and info.
        """
        self.influence_activation = np.random.binomial(
            1, self.adjacency_matrix_prob, size=(self.n_agents, self.n_agents)
        )

        for agent, act in actions.items():
            influence = self.influence_activation[agent]
            act_modif = -1 if act == 0 else 1
            for i, other_act in actions.items():
                if influence[i] > 0:
                    act_modif = (
                        act_modif + self.influence_sgn[agent][i]
                        if other_act == 1
                        else act_modif - self.influence_sgn[agent][i]
                    )
            act = (
                np.random.choice([0, 1])
                if act_modif == 0
                else (1 if act_modif > 0 else 0)
            )
            self._state[agent] = np.abs(self._state[agent] - act)

        terminations = {agent: False for agent in range(self.n_agents)}
        is_done = np.all(self._state == 0) or np.all(self._state == 1)
        if is_done:
            terminations = {agent: True for agent in self.possible_agents}
            self.agents = []

        truncations = {a: False for a in range(self.n_agents)}
        if self.timestep > self.max_steps:
            truncations = {a: True for a in self.possible_agents}
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
        infos = {agent: {} for agent in range(self.n_agents)}

        return observations, rewards, terminations, truncations, infos

    def get_majority_value(self) -> int:
        """
        Get the current majority value in the system.

        Returns:
            int: 1 if majority is 1, 0 if majority is 0, -1 if tied.
        """
        n_one = np.count_nonzero(self._state)
        return (
            1 if n_one > self.n_agents / 2 else 0 if n_one < self.n_agents / 2 else -1
        )

    def get_obs(self) -> dict:
        """
        Retrieve the observation dictionary for all agents.

        Returns:
            dict: Dictionary mapping agent indices to their respective observations.
        """
        return {
            agent: self._state[self.neighboring_masks[agent]]
            for agent in range(self.n_agents)
        }

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
        for agent, state_value in enumerate(self._state):
            self.graph_rendering.nodes[agent]["state"] = state_value

        # Node colors based on state
        node_colors = [self._state[i] for i in self.graph_rendering.nodes]
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
    def observation_space(self, agent: int) -> MultiDiscrete:
        """
        Define the observation space for a given agent.

        Args:
            agent (int): Agent index.

        Returns:
            MultiDiscrete: Observation space of the agent.
        """
        return MultiDiscrete([2] * np.count_nonzero(self.neighboring_masks[agent]))

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

    def state(self) -> MultiDiscrete:
        return self._state
