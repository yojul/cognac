Getting Started
================

COGNAC provides a PettingZoo implementation of various MARL problems with inherent graph-structure. The package can be used directly
*out-of-the-box* with large flexibility as implemented problems have minimal and self-contained dynamics with customizable hyperparameters.

Installation
-------------

Requirements
~~~~~~~~~~~~

This software uses Python 3. It is developed with version 3.12 but should also work with previous version (>= 3.6).
It is available on pypi (python package index) and, hence, can be installed easily. 

We recommend creating a dedicated virtual env to install the package and perform experiments (e.g. conda distribution or venv).
The python requirements are minimal and the package rely on standard package for RL environment (*Gymnasium, Petting-Zoo*) and widely used package such as *numpy* or *matplotlib* for visualization.
As the package is dedicated to problems with graph/network structure we leverage *networkx* package capabilities.

Installation
~~~~~~~~~~~~
**Simple installation directly from Pypi - (Recommended)** 

.. code-block:: bash

    pip install cognac

**Installation for advanced modification of dynamics or development purpose**

.. code-block:: bash

    git clone https://github.com/yojul/cognac.git
    cd cognac
    pip install -e .


.. note:: The code has been developed on MacOS and should be fully compatible with Linux. While, we expect it to work properly on Windows, there might be some bugs. Feel free to report any bug on GitHub.


Basic Usage
------------

Running a simple instance of an environment is straightforward and very similar to any standard PettingZoo environment and Gym-like APIs.
Environments directly inherit from PettingZoo Parallel Environment, for further details, please check `PettingZoo documentation <https://pettingzoo.farama.org/index.html>`_.
Therefore, it is easy to adapt any pre-existing RL and MARL code to one of the COGNAC's environment.
In addition, there are some utility functions to generate graph-structures (mainly through adjacency matrix) but also vizualization tools.

.. code-block:: python

    from cognac.env import BinaryConsensusNetworkEnvironment
    from cognac.utils.graph_utils import generate_adjacency_matrix, plot_influence_graph

    # Utility for generating random adjacency matrix
    adjacency_matrix = generate_adjacency_matrix(10)

    # Ploting influence graph (see figure below)
    plot_influence_graph(adjacency_matrix)

    # Instantiating environment with adjacency matrix and default parameters
    env = BinaryConsensusNetworkEnvironment(adjacency_matrix=adjacency_matrix)

    # Standard PettingZoo usage. See PettingZoo documentation for more details.
    obs, infos = env.reset()
    obs, rewards, dones, truncs, infos = env.step(
        {agent: env.action_space(agent).sample() for agent in env.possible_agents}
    )

.. figure:: assets/network_example.png
  :align: center
  :width: 400
  :alt: Example of randomly generated network from adjacency matrix
  
  Example of randomly generated network from adjacency matrix.