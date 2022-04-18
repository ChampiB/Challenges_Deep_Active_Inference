import agents.layers.ConvTranspose2d
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


@hydra.main(config_path="config", config_name="training")
def train(config):
    # TODO conv = agents.layers.ConvTranspose2d.ConvTranspose2d(1, 1, kernel_size=(2, 2), padding='same')
    # TODO x = torch.tensor([[[
    # TODO     [55, 52],
    # TODO     [57, 50]
    # TODO ]]], dtype=torch.float32)
    # TODO w = torch.tensor([[[
    # TODO     [1, 2],
    # TODO     [2, 1]
    # TODO ]]], dtype=torch.float32)
    # TODO for param in conv.parameters():
    # TODO     if len(param.shape) == 4:
    # TODO         with torch.no_grad():
    # TODO             param[0, 0, 0, 0] = w[0, 0, 0, 0].item()
    # TODO             param[0, 0, 0, 1] = w[0, 0, 0, 1].item()
    # TODO             param[0, 0, 1, 0] = w[0, 0, 1, 0].item()
    # TODO             param[0, 0, 1, 1] = w[0, 0, 1, 1].item()
    # TODO for param in conv.parameters():
    # TODO     print(param)
    # TODO print(conv(x))

    # TODO t = torch.arange(0, 100)
    # TODO t = torch.reshape(t, (10, 10))
    # TODO print(t)
    # TODO print(t[1:3, 2:])

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
    archive = Checkpoint(config, config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])
    agent.train(env, config)


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    train()
