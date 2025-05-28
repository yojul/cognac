.. COGNAC documentation master file, created by
   sphinx-quickstart on Thu Apr  3 16:00:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🥃 Welcome to COGNAC's Documentation
==================================

**COGNAC** (COoperative Graph-based Networked Agent Challenges) is a benchmark suite for evaluating and developing decentralized multi-agent reinforcement learning (MARL) algorithms on cooperative tasks with **graph-structured environments**.

Real-world systems such as power grids, traffic networks, and computer systems often exhibit complex interdependencies that can naturally be modeled as graphs. Yet, controlling such systems remains challenging due to their scale, partial observability, and combinatorial complexity. Standard single-agent RL struggles to scale in these settings.

COGNAC bridges the gap between theoretical models of network control and empirical reinforcement learning by providing:

- A flexible and modular suite of environments with **network topology**
- Support for **fully cooperative MARL tasks** on arbitrary graph structures
- Scalable problems designed to highlight the limitations of centralized control
- Tools for testing **decentralized, distributed, and frugal AI methods**

Motivation
----------

Despite recent advances in MARL, there is a lack of standardized, open-source benchmarks specifically focused on graph-structured control problems. Many existing environments either rely on centralized settings or do not fully exploit the graph structure of the domain.

COGNAC was built to fill this gap by offering:
- Minimal yet challenging environments tailored for graph-based cooperation
- Compatibility with modern RL libraries and tooling
- A platform to test scalability, generalization, and communication protocols

Key Features
------------

The package implements four different environments as self-contained Petting Zoo environments.
Each environment is highly customizable in terms of size, interactions structure and dynamics parameters.
In addition, it comes with several useful utility tools to generate adjacency matrix and graph-structure to instantiate
environments as well as some rendering and visualization tools.

- **Graph-native API**: define any graph topology for your cooperative problem
- **Simple, extensible environments**: start small, scale big
- **Baseline integrations**: compatible with standard MARL algorithms
- **Realistic use-cases**: inspired by domains like traffic, power systems, and logistics

Contributions
-------------

- A Python-based library offering **graph-structured multi-agent environments**
- The **first standardized open-source implementations** of theoretical graph-based MARL problems
- A collection of **benchmark results** using independent and centralized learning algorithms

Quick Links
-----------

- 📦 GitHub repository: `COGNAC <https://github.com/yojul/cognac>`_
- 📊 Benchmark examples: `cognac-benchmark-example <https://github.com/yojul/cognac-benchmark-example>`_


COGNAC is a Python-based benchmark suite offering flexible, graph-structured, 
cooperative multi-agent environments for MARL research. The package offers standardized minimal 
implementations of several well-known theoretical graph-based MARL problems taken from the literature such as the SysAdmin network :cite:authorpar:`guestrin2001max`
or Firefighting Graph :cite:authorpar:`oliehoek2016concise`, 
adapted for empirical benchmarking with modern RL tooling. 

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
   :maxdepth: 1
   :caption: Technical Documentation

   source/cognac.core
   source/cognac.env
   source/cognac.utils

References
----------

.. bibliography:: refs.bib
   :style: plain