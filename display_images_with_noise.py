from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import torch
from torch import tensor
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
from torch.distributions.multivariate_normal import MultivariateNormal


# TODO add this to the GUI
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
    agent.load(config["checkpoint"]["file"])

    # Collect images from the environment.
    Logger.get().info("Gather images from the environment...\n")
    images = []
    obs = env.reset()
    for i in range(0, 10):
        # Take an action in the environment.
        action = agent.step(obs, config)
        obs, _, _, _ = env.step(action)

        # Create noisy input image.
        image = torch.unsqueeze(obs, dim=0)
        noise = MultivariateNormal(tensor([0.0]), tensor([[0.01]])).sample(image.shape)
        noise = torch.squeeze(noise, dim=4)
        epsilon = 0.0001
        image = torch.clip(image + noise, min=epsilon, max=1-epsilon)

        # Generate an images using the VAE.
        mean, log_var = agent.encoder(image)
        next_state = agent.reparameterize(mean, log_var)
        image_rec = agent.decoder(next_state)
        image_rec = torch.nn.Sigmoid()(image_rec)
        images.append(image)
        images.append(image_rec)

    # Display the images in tensorboard.
    Logger.get().info("Display images...\n")
    images = torch.cat(images, dim=0)
    agent.writer.add_images("An example of generated images from noisy input", images)
    Logger.get().info("End.\n")


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    display_images()
