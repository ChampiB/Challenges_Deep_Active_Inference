from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate


@hydra.main(config_path="config", config_name="training")
def display_images(config):
    # Create the logger and keep track of the configuration.
    Logger.get().info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    Logger.get().info("Load the environment...\n")
    env = EnvFactory.make(config)
    with open_dict(config):
        config.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, config["images"]["shape"])

    # Load the agent from the checkpoint.
    Logger.get().info("Load the agent...\n")
    agent = instantiate(config["agent"])
    agent.load(config["checkpoint"]["directory"])

    # Collect images from the environment.
    Logger.get().info("Gather images from the environment...\n")
    images = []
    obs = env.reset()
    for i in range(0, 10):
        # Take an action in the environment.
        action = agent.step(obs, config)
        obs, _, _, _ = env.step(action)

        # Generate an images using the VAE.
        image = torch.unsqueeze(obs, dim=0)
        mean, log_var = agent.encoder(image)
        next_state = agent.reparameterize(mean, log_var)
        image = agent.decoder(next_state)
        images.append(torch.unsqueeze(obs, dim=0))
        images.append(image)

    # Display the images in tensorboard.
    images = torch.cat(images, dim=0)
    agent.writer.add_images("An example of generated images", images)


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    display_images()
