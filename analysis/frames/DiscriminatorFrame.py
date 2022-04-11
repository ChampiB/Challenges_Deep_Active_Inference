import tkinter as tk
import torch
from singletons.Device import Device
from analysis.widgets.Gallery import Gallery


#
# Class representing the page displaying the output of the discriminator.
#
class DiscriminatorFrame(tk.Frame):

    def __init__(self, parent, gui):
        """
        Construct the page displaying the output of the critic.
        :param parent: the parent of the page.
        :param gui: the graphical user interface.
        """
        # Call the constructor of the super class.
        super().__init__(parent)

        # Store gui data and empty image.
        self.gui = gui

        # Store the number of actions.
        self.n_actions = gui.config["env"]["n_actions"]

        # Create the gallery.
        self.gallery = Gallery(self, gui, self.refresh) \
            .add_image_column("Input image") \
            .add_empty_column() \
            .add_image_column("Image down") \
            .add_image_column("Image up") \
            .add_image_column("Image left") \
            .add_image_column("Image right") \
            .add_empty_column() \
            .add_data_column("Discriminator", self.n_actions, ["D(down)", "D(up)", "D(left)", "D(right)"]) \
            .add_control_bar()
        self.gallery.grid(row=0, column=0, sticky=tk.NSEW)

    def refresh(self):
        """
        Refresh the sample displayed in the galery.
        :return: nothing.
        """
        # Get input images.
        in_imgs = self.gallery.get_current_images()
        if in_imgs is None:
            self.gallery.reset()
            return

        # Extract the current state from the current observation.
        state, _, _ = self.gui.model.encoder(in_imgs)

        # Generate the next observations under each possible action.
        nb_samples = state.shape[0]
        actions = torch.IntTensor([i / nb_samples for i in range(0, self.n_actions * nb_samples)]).to(Device.get())
        next_state, _ = self.gui.model.transition(state.repeat(self.n_actions, 1), actions)
        next_obs = self.gui.model.decode_images(next_state)

        # Ask the dicriminator whether the images are realistic.
        are_real = self.gui.model.encoder(next_obs)[2]

        # Retrieve the images.
        images = []
        for act in range(0, self.n_actions):
            index = torch.LongTensor([i for i in range(act * nb_samples, (act + 1) * nb_samples)])
            images.append(next_obs.index_select(0, index))

        # Update the gallery.
        self.gallery.refresh({
            "Input image": in_imgs,
            "Image down": images[0],
            "Image up": images[1],
            "Image left": images[2],
            "Image right": images[3],
        }, {
            "Discriminator": are_real,
        })
