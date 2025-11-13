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
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from ...core.BaseReward import BaseReward
from ...utils.graph_utils import generate_coordination_graph
from .rewards import FactoredRewardModel


class BinaryConsensusNetworkEnvironment(ParallelEnv):
    """A multi-agent reinforcement learning environment modeling binary consensus.

    Agents interact on a probabilistic influence graph and attempt to reach a common
    binary state (0 or 1). Each agent's action influences its own state and that of its
    neighbors according to the adjacency matrix.
    """

    metadata = {"name": "binary_consensus_environment_v0"}

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        max_steps: int = 100,
        show_neighborhood_state: bool = True,
        reward_class: type[BaseReward] = FactoredRewardModel,
        is_global_reward: bool = False,
    ):
        """Initialize the consensus environment.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            A square matrix representing the influence between agents.
            Values must be in the range [0, 1] with zero diagonal.
        max_steps : int, optional
            Maximum number of steps before the episode is truncated. Default is 100.
        show_neighborhood_state : bool, optional
            Whether each agent observes its neighborhood. Default is True.
        reward_class : type, optional
            Class used for computing rewards. Must implement the reward model interface.
        is_global_reward : bool, optional
            If True, use a shared global reward. If False, compute rewards per agent.
        """

        self.adjacency_matrix = adjacency_matrix.copy()
        self.n_agents = adjacency_matrix.shape[0]
        self.possible_agents = list(range(self.n_agents))
        self.agents = None
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

        self.individual_state_from_obs = {
            id: sum(row[:id]) for id, row in enumerate(self.neighboring_masks)
        }

    def _check_adjacency_matrix(self) -> None:
        """Ensure adjacency matrix has valid structure.

        Raises
        ------
        AssertionError
            If the diagonal is non-zero or probabilities are not in [0, 1].
        """
        assert all(
            [self.adjacency_matrix_prob[i, i] == 0 for i in range(self.n_agents)]
        )
        assert np.all(self.adjacency_matrix_prob <= 1) and np.all(
            self.adjacency_matrix_prob >= 0
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
        """Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Options for reset (e.g. "init_vect" for setting an initial state).

        Returns
        -------
        observations : dict
            Observations for each agent after reset.
        infos : dict
            Info dictionaries for each agent.
        """

        def sample_binary_vector(n):
            # Step 1: choose number of zeros uniformly
            k = np.random.randint(0, n + 1)

            # Step 2: choose positions for zeros
            vec = np.ones(n, dtype=int)
            zero_positions = np.random.choice(n, k, replace=False)
            vec[zero_positions] = 0

            return vec

        self.agents = list(range(self.n_agents))
        self.timestep = 0

        if options and "init_vect" in options:
            assert len(options["init_vect"]) == self.n_agents
            self._state = np.array(options["init_vect"], dtype=float)
        else:
            self._state = sample_binary_vector(self.n_agents).astype(float)
            # prevent initialization with a consensus
            while np.all(self._state == 0) or np.all(self._state == 1):
                self._state = sample_binary_vector(self.n_agents).astype(float)

        observations = self.get_obs()
        infos = {a: {} for a in self.agents}
        self.reward.reset(self._state.astype(int))
        return observations, infos

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, dict],
    ]:
        """Perform one environment step using the given agent actions.

        Parameters
        ----------
        actions : dict
            Mapping from agent index to their binary action (0 or 1).

        Returns
        -------
        observations : dict
            New observations for each agent.
        rewards : dict
            Rewards assigned to each agent or shared globally.
        terminations : dict
            Flags indicating whether each agent's episode is terminated.
        truncations : dict
            Flags indicating whether each agent's episode is truncated.
        infos : dict
            Additional metadata for each agent.
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
        if self.reward.target is None:
            is_done = np.all(self._state == 0) or np.all(self._state == 1)
        else:
            is_done = np.all(self._state == self.reward.target)
        if is_done:
            terminations = {agent: True for agent in self.possible_agents}
            self.agents = []

        truncations = {a: False for a in range(self.n_agents)}
        is_truncated = self.timestep > self.max_steps - 2
        if is_truncated:
            truncations = {a: True for a in self.possible_agents}
            terminations = {a: True for a in self.possible_agents}
            self.agents = []
        rewards = self.reward(
            actions,
            self,
            is_done,
            is_truncated,
            as_global=self.is_global_reward,
        )
        self.timestep += 1
        observations = self.get_obs()
        infos = {agent: {} for agent in range(self.n_agents)}

        return observations, rewards, terminations, truncations, infos

    def get_majority_value(self) -> int:
        """Compute the current majority binary value.

        Returns
        -------
        int
            1 if majority is 1, 0 if majority is 0, -1 if tied.
        """
        n_one = np.count_nonzero(self._state)
        return (
            1 if n_one > self.n_agents / 2 else 0 if n_one < self.n_agents / 2 else -1
        )

    def get_obs(self) -> Dict[int, np.ndarray]:
        """Get current observations for all agents.

        Returns
        -------
        dict
            Observations keyed by agent index, each containing a binary vector
            of the agent's neighborhood.
        """
        return {
            agent: self._state[self.neighboring_masks[agent]]
            for agent in range(self.n_agents)
        }

    def render(self, save_frame: bool = False, fig=None, ax=None) -> None:
        """Render the current state of the environment.

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
        """Return the observation space for a given agent.

        Parameters
        ----------
        agent : int
            Index of the agent.

        Returns
        -------
        MultiDiscrete
            Observation space describing possible binary observations
            from the agent's neighborhood.
        """
        return MultiDiscrete([2] * np.count_nonzero(self.neighboring_masks[agent]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> Discrete:
        """Return the action space for a given agent.

        Parameters
        ----------
        agent : int
            Index of the agent.

        Returns
        -------
        Discrete
            Action space with two discrete actions: 0 or 1.
        """
        return Discrete(2)

    def state(self) -> np.ndarray:
        """Get the internal environment state.

        Returns
        -------
        np.ndarray
            Array of current binary states for all agents.
        """
        return self._state
