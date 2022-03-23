from __future__ import print_function
from singletons.Logger import Logger
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
import random
import torch
from singletons.Device import Device

import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file, transform=None):
        """
        Args:
            file (string): File containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filepath = file
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample, []


def load_dsprites(config, val_split=0.9):
    # img_size = 64
    path = config["env"]["images_archive"]
    dataset = DisentangledSpritesDataset(path, transform=transforms.ToTensor())

    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.seed(config["seed"])
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=val_sampler)

    return train_loader, val_loader


@hydra.main(config_path="config", config_name="training")
def train(config):
    # Set the seed requested by the user.
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    Logger.get(name="Training").info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Load the dsprites dataset.
    train_loader, _ = load_dsprites(config, val_split=0.1)

    # Create the agent.
    agent = instantiate(config["agent"])
    agent.load(config["checkpoint"]["directory"])

    # Train the agent.
    for epoch in range(1, 100):

        for _, (data, _) in enumerate(train_loader):
            # Sent the data to device.
            data = data.to(Device.get())

            # Compute the variational free energy.
            vfe_loss = agent.compute_vfe(config, data)

            # Perform one step of gradient descent.
            agent.optimizer.zero_grad()
            vfe_loss.backward()
            agent.optimizer.step()

            # Save the agent (if needed).
            if agent.steps_done % config["checkpoint"]["frequency"] == 0:
                agent.save(config["checkpoint"]["directory"])

            # Increase number of steps done.
            agent.steps_done += 1

    Logger.get().info("End.")


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    train()
