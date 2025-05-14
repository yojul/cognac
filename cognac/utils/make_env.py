from ..env import (
    BinaryConsensusNetworkEnvironment,
    GridFireFightingGraphEnvironment,
    MultiCommodityFlowEnvironment,
    RowFireFightingGraphEnvironment,
    SysAdminNetworkEnvironment,
)

ENV_NAME_TO_CLASS = {
    "binary_consensus": BinaryConsensusNetworkEnvironment,
    "grid_firefighting_graph": GridFireFightingGraphEnvironment,
    "row_firefighting_graph": RowFireFightingGraphEnvironment,
    "multi_commodity_flow": MultiCommodityFlowEnvironment,
    "sysadmin_network": SysAdminNetworkEnvironment,
}


def make_env(env_name, **config):
    env_list = ENV_NAME_TO_CLASS.keys()
    assert (
        env_name in env_list
    ), f"{env_name} is not an env from COGNAC. Available env are {', '.join(env_list)}."

    return ENV_NAME_TO_CLASS[env_name](**config)
