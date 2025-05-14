from ...core.BaseReward import BaseReward


class DefaultMCFReward(BaseReward):
    """
    Default reward function for the Multi Commodity Flow environment.
    The reward is the negative of the total cost of the flow.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, action, env, is_done, is_truncated, as_global=False):
        reward = {}
        for agent in env.possible_agents:
            reward[agent] = (
                sum(
                    [
                        1 / (data["weight"] ** 3) * data["flow"]
                        for _, _, data in env.network.out_edges(agent, data=True)
                    ]
                )
                / 1e4
            )
        return reward
