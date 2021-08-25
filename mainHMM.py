from algorithms.VAE import VAE
from algorithms.HMM import HMM
from algorithms.data.ExperienceBuffer import ExperienceBuffer, Experience
from torchvision.utils import make_grid
from wrappers.DefaultWrappers import DefaultWrappers
from debug.ImagesToGIF import ImagesToGIF
import gym


def train_hmm(env_name="MsPacman-v0", nb_steps=10000, debug=False):
    # Hyper-parameters
    n_states = 10
    batch_size = 32
    buffer_start_size = 32
    image_shape = (3, 84, 84)  # (number of images, image width, image height)

    # Create a environment
    env = DefaultWrappers.apply(gym.make(env_name), image_shape)
    obs = env.reset()

    # Render the environment if needed
    if debug:
        env.render()

    # Create the VAE
    vae = VAE(n_states, image_shape)
    vae.train()

    # Create the HMM
    hmm = HMM(vae, n_states, env.action_space.n)

    # Create the replay buffer
    buffer = ExperienceBuffer()

    # Train the VAE
    grid_images = []
    for step in range(nb_steps):

        # Select a random action.
        action = env.action_space.sample()

        # Execute the action in the environment.
        old_obs = obs
        obs, reward, done, _ = env.step(action)

        # Add the experience to the replay buffer.
        buffer.append(Experience(old_obs, action, reward, done, obs))

        # Perform one iteration of training
        if len(buffer) > buffer_start_size:
            loss_vae, loss_transition, images = hmm.training_step(buffer, batch_size)

            if debug and step % 100 == 0:
                # Print the step number
                print(f"========== Step {step} ==========")

                # Save reconstructed images in GIF file
                grid_images.append(make_grid(images.detach().cpu()))
                ImagesToGIF.convert(grid_images)

                # Print the Variational Free Energy
                print("Variational Free Energy: ", loss_vae.item())

                # Print the Variational Free Energy
                print("Transition model error: ", loss_transition.item())
                print()

        # Render the environment if needed
        if debug:
            env.render()

        # Reset the HMM and environment when a trial ends
        if done:
            obs = env.reset()

    # Close the environment
    env.close()


if __name__ == '__main__':
    # Train the HMM
    train_hmm("MsPacman-v0", debug=True)
