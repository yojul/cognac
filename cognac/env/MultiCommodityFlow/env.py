import functools
import warnings

import networkx as nx
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from ...utils.graph_utils import generate_coordination_graph
from .rewards import DefaultMCFReward


class MultiCommodityFlowEnvironment(ParallelEnv):

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
        """
        Apply agent actions to redistribute commodities to successors,
        compute rewards, and update environment state.
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
        """
        Splits integer X into parts according to a given distribution,
        such that the parts are integers and sum exactly to X.

        Args:
            X (int): The total integer to split.
            distribution (list of float): The target proportions (should sum to 1).

        Returns:
            list of int: List of integers summing to X, approximating the distribution.
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
        """
        Define the observation space for a given agent.

        Args:
            agent (int): Agent index.

        Returns:
            MultiDiscrete: Observation space of the agent.
        """
        return MultiDiscrete(
            [self.max_capacity] * (len(self.network.in_edges(agent)) + 1)
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> Discrete:
        """
        Define the action space for an agent.

        Args:
            agent (int): Agent index.

        Returns:
            Discrete: Action space (binary choice: 0 or 1).
        """
        out_deg = len(list(self.network.out_edges(agent)))
        if out_deg == 0:
            return Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)
        return Box(low=0.0, high=1.0, shape=(out_deg,), dtype=np.float32)
