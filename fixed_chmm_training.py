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


def get_override(config):
    return {
        "agent_module": config["agent"]["module"],
        "agent_class": config["agent"]["class"],
        "efe_lr": config["agent"]["efe_lr"],
        "discount_factor": config["agent"]["discount_factor"],
        "g_value": config["agent"]["g_value"],
        "n_steps_between_synchro": config["agent"]["n_steps_between_synchro"],
        "critic": instantiate(config["agent"]["critic"]),
        "action_selection": instantiate(config["agent"]["action_selection"])
    }


@hydra.main(config_path="config", config_name="training")
def train(config):
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
    archive = Checkpoint(config["agent"]["tensorboard_dir"], config["agent"]["loading_checkpoint_file"])
    if not archive.exists():
        print("Error: the trained HMM archive could not be located.")
        exit(1)
    agent = archive.load_model(override=get_override(config))
    agent.train(env, config)


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    train()
