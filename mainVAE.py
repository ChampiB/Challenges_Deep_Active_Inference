from algorithms.VAE import VAE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import zeros
from torchvision.utils import make_grid
from networks.decoders.MnistDecoder import MnistDecoder
from networks.encoders.MnistEncoder import MnistEncoder
from debug.ImagesToGIF import ImagesToGIF


def train_vae(dataset_loader):
    # Hyper-parameters
    n_states = 10
    n_epochs = 300

    # Prior over hidden states
    mean_p = zeros(n_states, requires_grad=False)
    log_variance_p = zeros(n_states, requires_grad=False)

    # Create the VAE
    vae = VAE(n_states=n_states)
    vae.encoder = MnistEncoder(n_states)
    vae.decoder = MnistDecoder(n_states)
    vae.train()

    # Train the VAE
    grid_images = []
    for epoch in range(n_epochs):
        for step, (x, _) in enumerate(dataset_loader):

            # Perform one iteration of training
            loss, image, _, _ = vae.training_step(x, mean_p, log_variance_p, 0.001)

            if step % 100 == 0:
                # Print the Variational Free Energy
                print("Variational Free Energy: ", loss.item())

            if step == 0:
                # Save reconstructed images
                print("Epoch number: ", epoch)
                image_grid = make_grid(image.detach().cpu())
                grid_images.append(image_grid)

                # Save reconstructed images in GIF file
                ImagesToGIF.convert(grid_images)


if __name__ == '__main__':
    # Loader MNIST dataset
    ds = MNIST("data/mnist", download=True, transform=transforms.ToTensor())
    ds_loader = DataLoader(dataset=ds, batch_size=32, shuffle=True)

    # Train the VAE
    train_vae(ds_loader)
