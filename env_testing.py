from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import numpy as np
import random
import torch
from agents.save.Checkpoint import Checkpoint


@hydra.main(config_path="config", config_name="testing")
def test(config):
    # Set the seed requested by the user.
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    Logger.get(name="Training").info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    env = EnvFactory.make(config)
    with open_dict(config):
        config.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, config["images"]["shape"])

    # Create the agent and train it.
    archive = Checkpoint(config["agent"]["tensorboard_dir"], config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])
    agent.test(env, config)


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    test()
