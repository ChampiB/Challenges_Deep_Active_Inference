import numpy as np

from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import agents.math_fc.functions as math_fc
from hydra.utils import instantiate
from agents.save.Checkpoint import Checkpoint
from environments.dSpritesEnv import dSpritesEnv


@hydra.main(config_path="config", config_name="training")
def display_images(config):
    # Create the logger and keep track of the configuration.
    Logger.get().info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    Logger.get().info("Load the environment...\n")
    initial_state = np.array([0, 2, 5, 15, 15, 5], dtype=np.float64)
    env = dSpritesEnv(config, initial_state)
    with open_dict(config):
        config.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, config["images"]["shape"])

    # Load the agent from the checkpoint.
    Logger.get().info("Load the agent...\n")
    archive = Checkpoint(config, config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])

    # Collect the initial image from the environment and infer the associated state.
    obs = env.reset()
    obs = torch.unsqueeze(obs, dim=0)
    mean, log_var = agent.encoder(obs)
    next_state = math_fc.reparameterize(mean, log_var)

    # Collect images from the environment.
    Logger.get().info("Gather images from the environment...\n")
    images = []
    actions = [torch.tensor([2])] * 10
    for action in actions:

        # Generate an images using the VAE.
        image = agent.decoder(next_state)
        image = torch.nn.Sigmoid()(image)
        images.append(obs)
        images.append(image)

        # Take an action in the environment.
        obs, _, _, _ = env.step(action)
        obs = torch.unsqueeze(obs, dim=0)

        # Make the agent imagine what would append if it was taking the action.
        mean, log_var = agent.transition(next_state, action)
        next_state = math_fc.reparameterize(mean, log_var)

    # Display the images in tensorboard.
    Logger.get().info("Display images...\n")
    images = torch.cat(images, dim=0)
    agent.writer.add_images("An example of (true and generated) trajectories", images)
    Logger.get().info("End.\n")


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    display_images()
