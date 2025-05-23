.. COGNAC documentation master file, created by
   sphinx-quickstart on Thu Apr  3 16:00:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🥃 COGNAC documentation
====================

COGNAC is a Python-based benchmark suite offering flexible, graph-structured, 
cooperative multi-agent environments for MARL research. The package offers standardized minimal 
implementations of several well-known theoretical graph-based MARL problems taken from the literature, 
adapted for empirical benchmarking with modern RL tooling.

What's COGNAC ?
=============

The package implements four different environments as self-contained Petting Zoo environments.
Each environment is highly customizable in terms of size, interactions structure and dynamics parameters.
In addition, it comes with several useful utility tools to generate adjacency matrix and graph-structure to instantiate
environments as well as some rendering and visualization tools.

These environments are inspired by classical pre-existing problems such as the SysAdmin network :cite:`guestrin2001max`
or Firefighting Graph :cite:`oliehoek2016concise`.
These problems have been widely studied and used as benchmark problems to test distributed multi-agent methods. 
However, to the best of our knowledge, there are no standard implementations available, which makes algorithm comparison 
more difficult in the long run. The chosen problems implemented in COGNAC are fully described in the following subsections. 
Some of them can be instantiated with any graph structure defined by the user, and we provide a collection of standard 
graph structures for benchmarking purposes with various sizes and properties: Directed Acyclic Graph, Tree, Undirected, Dense or Sparse graph, etc.

List of Environments
====================

.. list-table:: 
   :header-rows: 1
   :widths: 15 7 7 10 10

   * - Environment
     - Modular Size
     - Graph Agnostic
     - Joint State Space
     - Joint Act. Space
   * - Firefighting Graph(1D)
     - ✔️
     - ❌
     - :math:`\theta^N`
     - :math:`2^N`
   * - Firefighting Graph(2D)
     - ✔️
     - ❌
     - :math:`\theta^{N \times M}`
     - :math:`4^N`
   * - Binary Consensus
     - ✔️
     - ✔️
     - :math:`2^N`
     - :math:`2^N`
   * - SysAdmin
     - ✔️
     - ✔️
     - :math:`9^N`
     - :math:`2^N`
   * - Multi-commodity Flow
     - ✔️
     - ❌
     - :math:`\rho_{\text{max}}^{k \times E}`
     - :math:`\rho_{\text{max}}^{k \times E}`

.. toctree::
   :maxdepth: 2
   :caption: Overview

   quickstart
   marl
   

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation

   source/cognac.core
   source/cognac.env
   source/cognac.utils

.. bibliography:: refs.bib
   :style: plain