import tkinter as tk
from tkinter import messagebox
import torch


#
# Class representing the main navigation bar.
#
class NavBar(tk.Menu):

    def __init__(self, gui):
        """
        Construct the main navigation bar.
        :param gui: the graphical user interface.
        """
        # Call super class contructor.
        super().__init__(gui.window)

        # Store the graphical user interface.
        self.gui = gui

        # Add the load tab to the navigation bar.
        self.add_command(label="Load", command=self.load_cmd)

        # Add the model tab to the navigation bar, if needed.
        self.modelbar = tk.Menu(self, tearoff=0)
        if self.gui.model is None:
            self.add_command(label="Model", command=self.no_model_cmd)
        else:
            self.add_model_cascade()

        # Add the dataset tab to the navigation bar.
        if self.gui.dataset is None:
            self.add_command(label="Dataset", command=self.no_dataset_cmd)
        else:
            self.add_command(label="Dataset", command=self.dataset_cmd)

        # Add the sample tab to the navigation bar.
        if len(self.gui.samples) == 0:
            self.add_command(label="Sample", command=self.no_sample_cmd)
        else:
            self.add_command(label="Sample", command=self.sample_cmd)

        # Add the visualisation tab to the navigation bar.
        if self.gui.model is None:
            self.add_command(label="Visualisation", command=self.no_model_cmd)
        else:
            self.add_command(label="Visualisation", command=self.visualisation_cmd)

        # Add the disentanglement tab to the navigation bar.
        if self.gui.dataset is None:
            self.add_command(label="Disentanglement", command=self.no_dataset_cmd)
        elif self.gui.model is None:
            self.add_command(label="Disentanglement", command=self.no_model_cmd)
        else:
            self.add_command(label="Disentanglement", command=self.disentanglement_cmd)

    def add_model_cascade(self):
        """
        Add the model cascade to the nagivation bar.
        :return: nothing.
        """
        if self.model_has_attr(['encoder', 'decoder']):
            self.modelbar.add_command(label="Encoder/Decoder", command=self.encoder_decoder_cmd)
        if self.model_has_attr(['encoder', 'decoder', 'transition']):
            self.modelbar.add_command(label="Transition", command=self.transition_cmd)
        if self.model_has_attr(['encoder', 'critic']):
            self.modelbar.add_command(label="Critic", command=self.critic_cmd)
        if not self.model_has_attr(['encoder']) and self.model_has_attr(['critic']):
            self.modelbar.add_command(label="Critic", command=self.critic_without_encoder_cmd)
        if self.model_has_attr(['encoder', 'decoder', 'transition', 'discriminator']):
            self.modelbar.add_command(label="Discriminator", command=self.discriminator_cmd)
        self.add_cascade(label="Model", menu=self.modelbar)

    def model_has_attr(self, attrs):
        """
        Check if the gui's model has the requested attributes.
        :param attrs: the attributes to check.
        :return: True if the gui's model has the requested attributes, False otherwise.
        """

        # Check that the model has been loaded.
        if self.gui.model is None:
            return False

        # Check whether the model has the requested attributes.
        for attr in attrs:
            if attr == 'discriminator':
                if not self.model_has_discriminator():
                    return False
            elif not hasattr(self.gui.model, attr):
                return False
        return True

    def model_has_discriminator(self):
        """
        Check if the gui's model has a discriminator.
        :return: True if the gui's model has a discriminator, False otherwise.
        """
        if hasattr(self.gui.model, 'discriminator'):
            return True
        if not hasattr(self.gui.model, 'encoder'):
            return False
        img = torch.zeros(self.gui.config["images"]["shape"])
        img = torch.unsqueeze(img, dim=0)
        return len(self.gui.model.encoder(img)) >= 3

    def load_cmd(self):
        """
        Display page used to load the model and dataset.
        :return: nothing.
        """
        self.gui.show_frame("LoadFrame")

    def disentanglement_cmd(self):
        """
        Display page used to disentangle the latent space.
        :return: nothing.
        """
        self.gui.show_frame("DisentanglementFrame")

    @staticmethod
    def no_model_cmd():
        """
        Display a popup explaining to the user to load a model.
        :return: nothing.
        """
        error_msg = "You must provide a model before to click on this tab."
        messagebox.showerror("Error", error_msg)

    @staticmethod
    def no_dataset_cmd():
        """
        Display a popup explaining to the user to load a dataset.
        :return: nothing.
        """
        error_msg = "You must provide a dataset before to click on this tab."
        messagebox.showerror("Error", error_msg)

    @staticmethod
    def no_sample_cmd():
        """
        Display a popup explaining to the user to add images to the sample.
        :return: nothing.
        """
        error_msg = "You must add images to the sample before to click on this tab."
        messagebox.showerror("Error", error_msg)

    def discriminator_cmd(self):
        """
        Display the page used visualise the discriminator output.
        :return: nothing.
        """
        self.gui.show_frame("DiscriminatorFrame")

    def encoder_decoder_cmd(self):
        """
        Display the page used visualise the encoder and decoder outputs.
        :return: nothing.
        """
        self.gui.show_frame("EncoderDecoderFrame")

    def transition_cmd(self):
        """
        Display the page used visualise the transition outputs.
        :return: nothing.
        """
        self.gui.show_frame("TransitionFrame")

    def critic_cmd(self):
        """
        Display the page used visualise the critic outputs.
        :return: nothing.
        """
        self.gui.show_frame("CriticFrame")

    def critic_without_encoder_cmd(self):
        """
        Display the page used visualise the critic outputs when no encoder is available.
        :return: nothing.
        """
        self.gui.show_frame("CriticWithoutEncoderFrame")

    def dataset_cmd(self):
        """
        Display the page used to add images of the dataset to the sample.
        :return: nothing.
        """
        self.gui.show_frame("DatasetFrame")

    def sample_cmd(self):
        """
        Display the page used to visualise the current sample.
        :return: nothing.
        """
        self.gui.show_frame("SampleFrame")

    def visualisation_cmd(self):
        """
        Display the page used to visualise the latent space of the model.
        :return: nothing.
        """
        self.gui.show_frame("VisualisationFrame")
