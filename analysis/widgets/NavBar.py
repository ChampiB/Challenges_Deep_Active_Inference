import tkinter as tk


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

        # Add the model tab to the navigation bar.
        self.modelbar = tk.Menu(self, tearoff=0)
        self.modelbar.add_command(label="Encoder/Decoder", command=self.encoder_decoder_cmd)
        self.modelbar.add_command(label="Transition", command=self.transition_cmd)
        self.modelbar.add_command(label="Critic", command=self.critic_cmd)
        self.add_cascade(label="Model", menu=self.modelbar)

        # Add the dataset tab to the navigation bar.
        self.add_command(label="Dataset", command=self.dataset_cmd)

        # Add the sample tab to the navigation bar.
        self.add_command(label="Sample", command=self.sample_cmd)

        # Add the visualisation tab to the navigation bar.
        self.add_command(label="Visualisation", command=self.visualisation_cmd)

    def load_cmd(self):
        """
        Display page used to load the model and dataset.
        :return: nothing.
        """
        self.gui.show_frame("LoadFrame")

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
