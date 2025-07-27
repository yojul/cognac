💻 SysAdmin Network
===========================

.. contents:: Table of Contents
    :depth: 3

The Multi-agent SysAdmin problem is a widely used benchmark in the study of decision-making over networked systems. Originally introduced by Guestrin et al. in 2002 as a single-agent factored MDP benchmark :cite:authorpar:`guestrin2001max`, it was later extended into a multi-agent formulation to evaluate coordinated reinforcement learning algorithms :cite:authorpar:`guestrin2002coordinated`. Over the years, it has remained a standard reference in the field, with recent works reaffirming its relevance :cite:authorpar:`bargiacchi2021cooperative`, :cite:authorpar:`bianchi2024developing`.

This environment provides a modern and open-source implementation of the multi-agent version of the SysAdmin problem, specifically designed for multi-agent reinforcement learning (MARL). It maintains the original intent of testing structure-aware planning and coordination under uncertainty, while ensuring compatibility with modern MARL libraries.

Environment Description
-----------------------

The environment models a network of computers performing tasks. Each computer (agent) can be in one of several health states and may also be processing a task. Over time, machines may become *faulty*, slowing down task completion, or *dead*, making the task impossible to complete. Faults can spread probabilistically to neighboring computers. At each timestep, agents may choose to *reboot* their machine, which resets it to a working state (*good*) with high probability, but also discards progress on any active task.

This decentralized setting models a **Partially Observable Markov Decision Process** (PoMDP), where coordination between agents is critical for maintaining system-wide performance and limiting fault propagation.

.. figure:: ../assets/sysadmin-illustration.png
  :align: center
  :width: 100%
  :alt: Illustration for one state update in the SysAdmin Network problem.
  
  Illustration for one state update in the SysAdmin Network problem.

Graph Topology
~~~~~~~~~~~~~~

As in the Binary Consensus environment, the underlying graph defines the network topology. It determines which agents (computers) are neighbors and hence how faults can spread between them.

State Space
-----------

Each agent's state is defined by two categorical variables:

- **Health status:** one of *good*, *faulty*, or *dead*
- **Task status:** one of *idle*, *loaded*, or *successful*

Formally, the global joint state at time :math:`t` is an element of:

.. math::

    S(t) \in \{\text{good}, \text{faulty}, \text{dead}\}^N \times \{\text{idle}, \text{loaded}, \text{successful}\}^N

This results in a total state space size of :math:`9^N`.

Action Space
------------

At each timestep, every agent selects one of two discrete actions:

- **Do nothing** (continue current operation)
- **Reboot** the machine (resets health to *good* with high probability but loses current task progress)

Objective
---------

The overall goal is to **maximize the number of successfully completed tasks** over time. This objective can be framed as either a finite-horizon or infinite-horizon cumulative reward problem. Performance is closely tied to how effectively agents collaborate and leverage the graph structure to mitigate cascading faults.

By modeling local and global trade-offs in a structured environment, the Multi-agent SysAdmin problem serves as an effective testbed for evaluating decentralized and semi-centralized MARL strategies under uncertainty and partial observability.



Environment
------------------------------

.. automodule:: cognac.env.SysAdmin.env
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:

Rewards
----------------------------------

.. automodule:: cognac.env.SysAdmin.rewards
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members: