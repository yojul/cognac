import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_influence_graph(matrix: np.ndarray):
    """Plot a directed graph representing influences between nodes.

    Each node represents an entity, and directed edges (with weights)
    indicate influence from one node to another.

    Parameters
    ----------
    matrix : np.ndarray
        A square 2D numpy array where entry (i, j) represents
        the influence of node j on node i.

    Returns
    -------
    None
        Displays the plotted influence graph using matplotlib.
    """

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


def generate_adjacency_matrix(
    n: int, gain: float = 0.3, p=0.2, symmetry: bool = False
) -> np.ndarray:
    """Generate a random weighted adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    gain : float, optional
        Maximum value of weights (default is 0.3).
    p : float, optional
        Probability of an edge between two nodes (default is 0.2).
    symmetry : bool, optional
        If True, the matrix will be symmetric (undirected graph).

    Returns
    -------
    np.ndarray
        A square matrix of shape (n, n) representing weighted edges.
    """
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
) -> np.ndarray:
    """Generate a banded adjacency matrix where each node connects to neighbors within a
    band.

    Parameters
    ----------
    n : int
        Number of nodes.
    neighborhood_size : int
        Range of neighbors each node connects to on either side.
    p_min : float, optional
        Minimum weight of an edge (default is 0.2).
    p_max : float, optional
        Maximum weight of an edge (default is 1.0).

    Returns
    -------
    np.ndarray
        A banded weighted adjacency matrix of shape (n, n).
    """
    assert neighborhood_size > 0, "Neighborhood size should be positive."
    assert (
        neighborhood_size < n // 2
    ), "The size of the neighborhood can be at most n/2. Try a lower value"
    matrix = (
        np.tri(n, n, neighborhood_size)
        - np.tri(n, n, -neighborhood_size - 1)
        - np.identity(n)
    )
    matrix[np.nonzero(matrix)] = 0.25 + 0.75 * np.random.random(
        size=len(np.nonzero(matrix)[0])
    )
    return matrix


def generate_deterministic_adjacency_matrix(
    n: int, p: float = 0.2, symmetry: bool = False
) -> np.ndarray:
    """Generate a deterministic binary adjacency matrix with fixed edge probability.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float, optional
        Probability of an edge existing (default is 0.2).
    symmetry : bool, optional
        Whether the matrix should be symmetric.

    Returns
    -------
    np.ndarray
        A binary adjacency matrix of shape (n, n).
    """
    matrix = (np.random.rand(n, n) > (1 - p)).astype(float)
    np.fill_diagonal(matrix, 0)
    if symmetry:
        return (matrix + matrix.T) / 2  # Symmetric influence
    else:
        return matrix


def generate_dag_influence_matrix(n: int, p: float = 1.0) -> np.ndarray:
    """Generate a random Directed Acyclic Graph (DAG) influence matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float, optional
        Probability of an edge existing between two nodes (default is 1).

    Returns
    -------
    np.ndarray
        An upper-triangular binary matrix representing a DAG.
    """
    matrix = (np.random.rand(n, n) < p).astype(float)
    np.fill_diagonal(matrix, 0)  # No self-loops

    # Ensure acyclicity: Keep only upper-triangular part
    matrix = np.triu(matrix, k=1)

    return matrix


def generate_coordination_graph(matrix: np.ndarray) -> nx.DiGraph:
    """Generate a directed NetworkX graph from an influence matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D square numpy array where entry (i, j) indicates
        the influence of node j on node i.

    Returns
    -------
    networkx.DiGraph
        A directed graph representation of the influence matrix.
    """
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
    """Convert a matrix into a row-stochastic matrix.

    Each row is normalized so it sums to 1. If a row has all zeros,
    it is replaced with a uniform distribution over columns.

    Parameters
    ----------
    prob_matrix : np.ndarray
        A 2D numpy array representing transition probabilities or weights.

    Returns
    -------
    np.ndarray
        A row-normalized stochastic matrix.
    """
    prob_matrix = np.array(prob_matrix, dtype=float)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero: if a row sum is 0, set it to 1 temporarily
    row_sums[row_sums == 0] = 1.0

    stochastic_matrix = prob_matrix / row_sums

    return stochastic_matrix
