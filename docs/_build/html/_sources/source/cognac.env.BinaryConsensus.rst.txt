🤝 Binary Consensus
==================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

The Binary Consensus problem is a simple yet non-trivial decentralized multi-agent environment inspired by the classic voter model introduced by Holley and Liggett in 1975 :cite:authorpar:`holley1975ergodic`. In the original model, agents (or "particles") interact over a graph and influence each other's decisions. While this model has been extensively studied in statistical physics, its adaptation to the Markov Decision Process (MDP) framework—particularly in decentralized, fully cooperative settings—is relatively rare.

This environment adapts the voter model into a discrete-time decentralized partially observable Markov decision process (Dec-POMDP). Each agent observes its own binary vote as well as those of its neighbors and can choose to retain or flip its vote at each time step. Agents’ states are also stochastically influenced by their neighbors’ actions. Despite its simplicity, the problem presents interesting challenges for coordination and decision-making.

.. figure:: ../assets/binary-consensus-illustration.png
  :align: center
  :width: 100%
  :alt: Illustration for one state update in the binary consensus problem.
  
  Illustration for one state update in the Binary Consensus problem.

Problem Formulation
-------------------

**Agents and Votes:**  
There are :math:`N` agents. Each agent maintains a binary state :math:`s_i(t) \in \{0, 1\}`, representing its vote at time :math:`t`. The joint state at time :math:`t` is:

.. math::

    S(t) = \{s_i(t)\}_{i=1}^N

**Observations:**  
Each agent observes its own vote and the votes of its direct neighbors as defined by a fixed graph structure.

**Action Space:**  
At every time step, each agent selects an action :math:`a_i(t) \in \{0, 1\}`:

- :math:`a_i(t) = 0`: keep the current vote  
- :math:`a_i(t) = 1`: switch to the opposite vote

**Objective:**  
The goal is to reach a consensus corresponding to the initial majority vote within a fixed time horizon :math:`T`. The initial majority :math:`m_0` is defined as:

.. math::

    m_0 = \arg\max_{v \in \{0,1\}} \sum_{i=1}^{N} \mathbf{1}[s_i(0) = v]

A consensus is said to be reached at time :math:`t` if all agents share the same vote:

.. math::

    s_i(t) = m_0 \quad \forall i \in \{1, \dots, N\}

An episode terminates either when consensus is reached (even if incorrect) or when the time horizon :math:`T` is exceeded.

Use Cases and Complexity
------------------------

While this theoretical model is not tied to any specific real-world application, its simplicity and flexibility make it a strong benchmark for evaluating decentralized, centralized, and hybrid learning strategies across varying graph sizes and topologies.

The environment has both a state space and an action space of size :math:`2^N`, which quickly becomes intractable as the number of agents increases. This property makes it a useful stress test for centralized methods and a valuable tool for studying scalability in multi-agent reinforcement learning.

A simple, intuitive heuristic policy is available as a baseline: at each time step, each agent adopts the majority vote of its local neighborhood. This provides a competitive reference for evaluating learned strategies.

Environment
-------------------------------------

.. automodule:: cognac.env.BinaryConsensus.env
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:

Rewards
-----------------------------------------

The default rewards here is the `FactoredRewardModel`. 
This reward gives a penalty to each agent at each step for disagreeing with the current majority (which does not necessarily match the objective consensus). 
At the terminal state, it gives a large reward for reaching the consensus, weighted by the time it took to reach it (the faster, the better).
If the consensus is not reached and the game reaches the maximum horizon, it gives a large negative reward weighted by the distance to the consensus.

More formally, the reward model works like this:

**During an episode**:  
Each agent gets a local reward at each step:

.. math::

   r_i(t) =
   \begin{cases}
       0 & \text{if agent } i \text{ agrees with majority} \\
       -1 & \text{otherwise}
   \end{cases}

**At episode end**:

- Let :math:`\tau` be the temporal weight factor in the reward, it 

.. math::

   \tau = \frac{t_{\max}-t_{\rm final}}{t_{\max}}, \quad

:math:`t_{\max}` is the maximum length of an episode and :math:`t_{\rm final}` is the actual terminal timestep. Thus, this temporal factor goes linearly from 1 to 0 in an episode.

- Let :math:`\xi` be a penalty term that is added whenever the consensus is not reach at the end of an episode.

.. math::
    \xi =
        \begin{cases}
            -100 & \tau = 0 \\
            0 & \text{otherwise}
        \end{cases}

Then the final reward is computed using the ratio to the consensus :math:`x_{\rm final}/N`, :math:`x` being the number of agents agreeing with the objective value.

.. math::
   r_i(\text{end}) = \frac{\tau \,\left(100\,x_{\rm final}/N + \xi \right)}{N}




.. automodule:: cognac.env.BinaryConsensus.rewards
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:

