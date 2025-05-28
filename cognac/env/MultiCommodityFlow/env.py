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
import warnings

import networkx as nx
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from ...utils.graph_utils import generate_coordination_graph
from .rewards import DefaultMCFReward


class MultiCommodityFlowEnvironment(ParallelEnv):
    """Multi-Commodity Flow Environment based on a directed graph representing agents as
    nodes controlling flow of multiple commodities through edges.

    This environment models a circulation network where agents redistribute
    commodities along outgoing edges, subject to capacity constraints. It
    supports multi-agent reinforcement learning via the PettingZoo ParallelEnv
    interface.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Square matrix describing the influence graph adjacency between agents.
        Positive/negative entries define direction and influence strength.
    n_commodities : int, optional
        Number of commodity types flowing in the network (default is 5).
    max_capacity : int, optional
        Maximum capacity of each edge or node (default is 100).
    max_steps : int, optional
        Maximum number of steps per episode before termination (default is 20).
    reward_class : type, optional
        Class used to compute the reward at each step (default is DefaultMCFReward).
    is_global_reward : bool, optional
        Whether to use a global reward shared across all agents
        or individual rewards (default False).

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        The input adjacency matrix of the network.
    n_agents : int
        Number of agents/nodes in the network.
    possible_agents : list of int
        List of agent indices representing nodes.
    max_capacity : int
        Maximum capacity of edges/nodes.
    network : networkx.DiGraph
        Directed graph representing the network topology and flows.
    timestep : int
        Current time step in the episode.
    state : object
        Current environment state (custom structure).
    reward : object
        Reward function instance for computing step rewards.
    influence_activation : np.ndarray
        Boolean matrix indicating active influences between agents.
    influence_sgn : np.ndarray
        Matrix indicating sign (+/-) of influences.
    adjacency_matrix_prob : np.ndarray
        Absolute value of adjacency matrix entries, interpreted as probabilities.

    Methods
    -------
    reset(seed=None, options=None)
        Reset the environment to initial state and sample initial flows.
    step(actions)
        Perform one environment step applying agent actions and updating flows.
    get_obs()
        Return observations for all agents.
    observation_space(agent)
        Return the observation space for a given agent.
    action_space(agent)
        Return the action space for a given agent.
    """

    metadata = {"name": "multicommodity_flow_environment_v0"}

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        n_commodities: int = 5,
        max_capacity: int = 100,
        max_steps: int = 20,
        reward_class: type = DefaultMCFReward,
        is_global_reward: bool = False,
    ):

        self.adjacency_matrix = adjacency_matrix
        self.n_agents = self.adjacency_matrix.shape[0]
        self.n_commodities = n_commodities
        self.possible_agents = list(range(self.n_agents))
        self.max_capacity = max_capacity
        self.state = None
        self.timestep = None
        self.max_steps = max_steps
        self.reward = reward_class()
        self.is_global_reward = is_global_reward
        self.influence_activation = np.zeros((self.n_agents, self.n_agents), dtype=bool)

        self.influence_sgn = np.sign(self.adjacency_matrix)
        self.adjacency_matrix_prob = np.abs(self.adjacency_matrix)
        self._check_influence_graph()

        # USELESS ?
        self.neighboring_masks = np.identity(self.n_agents, dtype=bool)
        self.filled_influence = self.adjacency_matrix.copy()
        self.neighboring_masks = self.neighboring_masks + (self.filled_influence != 0)

        self._init_total_circulation = None

        self.network = generate_coordination_graph(self.adjacency_matrix)
        nx.set_edge_attributes(
            self.network,
            self.max_capacity,
            "capacity",
        )
        self._init_type_node()
        self.pos_layout = nx.circular_layout(self.network)

    def _init_type_node(self) -> None:
        """Initialize node types in the network graph based on connectivity.

        Node types:
        - 'source': no predecessors (input node)
        - 'sink': no successors (output node)
        - 'circulation': has both predecessors and successors
        - 'unconnected': isolated node (no predecessors or successors)

        Sets the "type" attribute on each node in the network graph.

        Raises
        ------
        AssertionError
            If the network does not contain at least one source and one sink node,
            or if it contains unsupported types.

        .. warning::
           Internal use only. This method is intended for internal environment setup.
        """

        for n, data in self.network.nodes(data=True):
            if len(self.network.pred[n]) == 0 and len(self.network.succ[n]) == 0:
                data["type"] = "unconnected"
                warnings.warn(
                    f"The node {n} is unconnected, associated agent will"
                    "have no available action nor observation."
                )
            elif len(self.network.pred[n]) == 0:
                data["type"] = "source"
            elif len(self.network.succ[n]) == 0:
                data["type"] = "sink"
            else:
                data["type"] = "circulation"
        available_types = nx.get_node_attributes(self.network, "type").values()
        assert ("source" in available_types and "sink" in available_types) or all(
            n_type in ["circulation", "unconnected"] for n_type in available_types
        ), "The network needs to have at least 1 source and 1 sink,"
        "or alternatively only circulation nodes. Please check the graph."

    def _check_influence_graph(self) -> None:
        """Validate the influence graph properties.

        Checks that the diagonal of the adjacency matrix probability matrix is zero,
        ensuring no self-influence, and verifies that all entries are within [0,1].

        Raises
        ------
        AssertionError
            If the diagonal entries are not zero or if any entry is out of bounds.

        .. warning::
           Internal use only. Used to ensure network consistency.
        """

        assert all(
            [self.adjacency_matrix_prob[i, i] == 0 for i in range(self.n_agents)]
        )
        assert np.all(self.adjacency_matrix_prob <= 1) and np.all(
            self.adjacency_matrix_prob >= 0
        )

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Reset the environment to the initial state.

        Resets all agent states, network flows, and commodities to initial random
        values subject to capacity constraints.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generators to ensure reproducibility.
        options : dict, optional
            Additional options for environment reset.

        Returns
        -------
        tuple
            A tuple containing:
            - observations (dict): Mapping from agent ID to initial observation.
            - infos (dict): Mapping from agent ID to info dict (empty by default).
        """
        self.agents = self.possible_agents.copy()
        self.timestep = 0
        self._init_total_circulation = 0
        assert all(
            [
                data["type"] == "circulation" or "unconnected"
                for _, data in self.network.nodes(data=True)
            ]
        ), "Provided network is not suitable for a circulation-only environment."

        # Initialize flows at 0
        nx.set_edge_attributes(self.network, 0, "flow")

        # Sample initial circulation stocks
        for n, data in self.network.nodes(data=True):
            if data["type"] == "circulation":
                data["commodities"] = np.random.randint(0, self.max_capacity)
                self._init_total_circulation += data["commodities"]
                dsitrib_sample = self.action_space(n).sample()
                flow_distrib = self._split_integer_by_distribution(
                    data["commodities"],
                    [d / sum(dsitrib_sample) for d in dsitrib_sample],
                )
                nx.set_edge_attributes(
                    self.network,
                    {
                        edge: flow_distrib[idx]
                        for idx, edge in enumerate(self.network.out_edges(n))
                    },
                    "flow",
                )
                assert data["commodities"] == sum(flow_distrib)  # Security for dev
            else:
                data["commodities"] = 0

        observations = self.get_obs()
        infos = {a: {} for a in self.agents}
        self.reward.reset(self.state)
        return observations, infos

    def step(self, actions: dict) -> tuple:
        """Execute a step of the environment using the provided agent actions.

        Each agent redistributes its commodity stock along outgoing edges according
        to the action distribution. The environment updates flow values, applies
        reward computation, and checks termination conditions.

        Parameters
        ----------
        actions : dict
            Mapping from agent ID to a list or array of dispatch values for
            outgoing edges.

        Returns
        -------
        tuple
            A 5-tuple containing:
            - observations (dict): Agent observations after step.
            - rewards (dict): Reward values for each agent.
            - terminations (dict): Boolean flags indicating episode
            termination per agent.
            - truncations (dict): Boolean flags indicating episode
            truncation per agent.
            - infos (dict): Additional info dictionaries per agent.
        """
        self.timestep += 1

        # Apply dispatching actions at each nodes
        for agent, act in actions.items():
            if len(act) == 0:
                continue

            distrib = [d / sum(act) for d in act]
            stock_to_dispactch = min(
                self.max_capacity, self.network.nodes[agent]["commodities"]
            )
            self.network.nodes[agent]["commodities"] -= stock_to_dispactch
            dispatch = self._split_integer_by_distribution(
                stock_to_dispactch, distribution=distrib
            )
            nx.set_edge_attributes(
                self.network,
                {
                    edge: dispatch[idx]
                    for idx, edge in enumerate(self.network.out_edges(agent))
                },
                "flow",
            )

        for agent, data in self.network.nodes(data=True):
            data["commodities"] += sum(
                [data["flow"] for _, _, data in self.network.in_edges(agent, data=True)]
            )

        observations = self.get_obs()
        rewards = self.reward(actions, self, False, False)
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if self.timestep >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
            terminations = {agent: True for agent in self.agents}
            self.agents = []
            rewards = self.reward(actions, self, True, True)
            # infos = {agent: {"final_reward": rewards[agent]} for agent in self.agents}

        assert (
            sum([data["commodities"] for _, data in self.network.nodes(data=True)])
            == self._init_total_circulation
        ), "The total circulation is not conserved. Please check the flow computation."

        return observations, rewards, terminations, truncations, infos

    def get_obs(self) -> dict:
        """Get current observations for all agents.

        Observation consists of the agent's current commodity stock and
        the flow values of all incoming edges concatenated into a single numpy array.

        Returns
        -------
        dict
            Mapping from agent ID to numpy array representing the observation.
        """
        observations = {}
        for agent in self.possible_agents:
            incoming_edges = self.network.in_edges(agent, data=True)
            # outgoing_edges = self.network.out_edges(agent, data=True)
            observations[agent] = np.concatenate(
                [
                    np.array([self.network.nodes[agent]["commodities"]]),
                    np.array([data["flow"] for _, _, data in incoming_edges]),
                ]
            )
        return observations

    def _split_integer_by_distribution(
        self, stock: int, distribution: list[float]
    ) -> list:
        """Split an integer stock into parts proportional to a given distribution.

        Ensures that the returned list of integer parts sums exactly to `stock`.
        The splitting is done by flooring the proportional amounts and distributing
        the remainder according to the highest fractional parts.

        Parameters
        ----------
        stock : int
            Total integer value to split.
        distribution : list of float
            List of proportions (not necessarily normalized) that sum to 1.

        Returns
        -------
        list of int
            List of integer parts summing exactly to `stock`.

        .. warning::
           Internal utility method for flow distribution calculation.
        """
        # Step 1: Multiply X by each proportion
        if len(distribution) == 1:
            return [stock]
        raw_parts = [stock * p for p in distribution]
        # Step 2: Floor each part to get the initial integer parts
        int_parts = [int(part) for part in raw_parts]

        # Step 3: Calculate the remainder
        remainder = stock - sum(int_parts)

        # Step 4: Distribute the remainder to the highest fractional parts
        fractional_parts = [part - int(part) for part in raw_parts]
        sorted_indices = sorted(
            range(len(fractional_parts)), key=lambda i: -fractional_parts[i]
        )

        for i in range(remainder):
            int_parts[sorted_indices[i]] += 1
        return int_parts

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: int) -> MultiDiscrete:
        """Split an integer stock into parts proportional to a given distribution.

        Ensures that the returned list of integer parts sums exactly to `stock`.
        The splitting is done by flooring the proportional amounts and distributing
        the remainder according to the highest fractional parts.

        Parameters
        ----------
        stock : int
            Total integer value to split.
        distribution : list of float
            List of proportions (not necessarily normalized) that sum to 1.

        Returns
        -------
        list of int
            List of integer parts summing exactly to `stock`.

        .. warning::
           Internal utility method for flow distribution calculation.
        """
        return MultiDiscrete(
            [self.max_capacity] * (len(self.network.in_edges(agent)) + 1)
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> Discrete:
        """Return the action space specification for a given agent.

        If the agent has no outgoing edges, returns a zero-dimensional Box.
        Otherwise, returns a Box space with shape equal to the number of outgoing edges,
        with each action value in [0.0, 1.0], representing proportions.

        Parameters
        ----------
        agent : int
            Agent index.

        Returns
        -------
        Box
            Gymnasium Box space defining valid actions for the agent.
        """
        out_deg = len(list(self.network.out_edges(agent)))
        if out_deg == 0:
            return Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)
        return Box(low=0.0, high=1.0, shape=(out_deg,), dtype=np.float32)
