from agents.DGN import DGN
from agents.DQN import DQN
from agents.RandomAgent import RandomAgent


# TODO this has been replaced by instantiate
def make(agent_type, config):
    """
    Create an agent of the type specified in input.
    :param agent_type: the type of agent to create, e.g. DGN, DQN, Random, ...
    :param config: the hydra configuration.
    :return: the created agent.
    """
    if agent_type == "DGN":
        return DGN(config["env"]["name"], config["images"]["shape"], config["env"]["n_actions"])
    if agent_type == "DQN":
        return DQN(config["env"]["name"], config["images"]["shape"], config["env"]["n_actions"])
    if agent_type == "Random":
        return RandomAgent(config["env"]["n_actions"])
    raise Exception("Error: Agent type unsupported.")
