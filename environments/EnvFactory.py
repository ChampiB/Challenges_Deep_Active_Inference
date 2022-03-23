import gym
from environments.dSpritesEnv import dSpritesEnv
from environments.RandomdSpritesEnv import RandomdSpritesEnv
from environments.MnistEnv import MnistEnv
from environments.MazeEnv import MazeEnv


def make(config):
    """
    Create the environment according to the configuration.
    :param config: the hydra configuration.
    :return: the created environment.
    """

    # The list of custom environments.
    environments = {
        "RandomdSprites": RandomdSpritesEnv,
        "dSprites": dSpritesEnv,
        "MNIST": MnistEnv,
        "Maze": MazeEnv
    }

    # Instantiate the environment requested by the user.
    env_name = config["env"]["name"]
    if env_name in environments.keys():
        return environments[env_name](config)
    else:
        return gym.make(env_name)
