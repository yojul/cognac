import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_influence_graph(matrix):
    G = nx.DiGraph()
    n = matrix.shape[0]

    # Add all nodes to the graph
    G.add_nodes_from(range(n))

    # Add edges with weights
    for i in range(n):
        for j in range(n):
            if matrix[i, j] != 0:  # Only add nonzero influences
                G.add_edge(j, i, weight=matrix[i, j])  # Influence from j to i

    # pos = nx.spring_layout(
    #     G, scale=3, k=3 / np.sqrt(n)
    # )  # Adjust layout for larger graphs
    pos = nx.shell_layout(G, scale=3)
    edges = G.edges(data=True)

    edge_colors = ["red" if data["weight"] < 0 else "blue" for _, _, data in edges]
    edge_labels = {(i, j): f"{data['weight']:.2f}" for i, j, data in edges}

    plt.figure(figsize=(10, 10))  # Increase figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        font_size=10,
        node_color="lightgray",
        edge_color=edge_colors,
        arrows=True,
        connectionstyle="arc3,rad=0.2",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.show()


def generate_adjacency_matrix(n, gain=0.3, p=0.2, symmetry=False):
    matrix = (
        gain
        * np.random.rand(n, n)
        * np.random.choice([0, 1], p=[1 - p, p], size=(n, n))
    )
    np.fill_diagonal(matrix, 0)  # No self-influence
    if symmetry:
        return (matrix + matrix.T) / 2  # Symmetric influence
    else:
        return matrix


def generate_band_adjacency_matrix(
    n: int, neighborhood_size: int, p_min: float = 0.2, p_max: float = 1.0
):
    assert neighborhood_size > 0, "Neighborhood size should be positive."
    assert (
        neighborhood_size < n // 2
    ), "The size of the neighborhood can be at most n/2. Try a lower value"
    matrix = (
        np.tri(n, n, neighborhood_size)
        - np.tri(10, 10, -neighborhood_size - 1)
        - np.identity(10)
    )
    matrix[np.nonzero(matrix)] = 0.25 + 0.75 * np.random.random(
        size=len(np.nonzero(matrix)[0])
    )
    return matrix


def generate_deterministic_adjacency_matrix(n, p=0.2, symmetry=False):
    matrix = (np.random.rand(n, n) > (1 - p)).astype(float)
    np.fill_diagonal(matrix, 0)
    if symmetry:
        return (matrix + matrix.T) / 2  # Symmetric influence
    else:
        return matrix


def generate_dag_influence_matrix(n, p=1):
    """
    Generates a random Directed Acyclic Graph (DAG) with n nodes.

    Parameters:
    - n (int): Number of nodes in the graph.
    - p (float): Probability of an edge existing between two nodes.

    Returns:
    - np.ndarray: An n x n adjacency matrix representing the DAG.
    """
    matrix = (np.random.rand(n, n) < p).astype(float)
    np.fill_diagonal(matrix, 0)  # No self-loops

    # Ensure acyclicity: Keep only upper-triangular part
    matrix = np.triu(matrix, k=1)

    return matrix


def generate_coordination_graph(matrix):
    G = nx.DiGraph()
    n = matrix.shape[0]

    # Add all nodes to the graph
    G.add_nodes_from(range(n))

    # Add edges with weights
    for i in range(n):
        for j in range(n):
            if matrix[i, j] != 0:  # Only add nonzero influences
                G.add_edge(j, i, weight=matrix[i, j])  # Influence from j to i

    return G


def to_stochastic_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms a given matrix into a row-stochastic matrix.

    Each row is normalized so that it sums to 1. If a row sums to 0,
    it is replaced with a uniform distribution over its columns.

    Parameters:
    - prob_matrix (np.ndarray): A 2D numpy array representing a probability matrix.

    Returns:
    - np.ndarray: A row-stochastic version of the input matrix.
    """
    prob_matrix = np.array(prob_matrix, dtype=float)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero: if a row sum is 0, set it to 1 temporarily
    row_sums[row_sums == 0] = 1.0

    stochastic_matrix = prob_matrix / row_sums

    return stochastic_matrix
