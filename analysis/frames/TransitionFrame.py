import tkinter as tk
from analysis.widgets.Gallery import Gallery
from tkinter import ttk
import torch


#
# Class representing the page displaying the output of the transition network.
#
class TransitionFrame(tk.Frame):

    def __init__(self, parent, gui):
        """
        Construct the page displaying the output of the transition network
        :param parent: the parent of the page.
        :param gui: the graphical user interface.
        """
        # Call the constructor of the super class.
        super().__init__(parent)

        # Store the graphical user interface
        self.gui = gui

        # Store the number of latent dimensions.
        self.n_dims = gui.config["agent"]["n_states"]

        # Create the gallery
        self.gallery = Gallery(self, gui, self.refresh) \
            .add_image_column("Input image") \
            .add_data_column("Input latent representation", self.n_dims) \
            .add_image_column("Output image") \
            .add_data_column("Output latent representation", self.n_dims)

        # Create the comboBox for selecting action and the corresponding label.
        self.action_label = tk.Label(self.gallery, text="Action:")

        self.selected_action = tk.StringVar()
        self.action_cb = ttk.Combobox(self.gallery, textvariable=self.selected_action, state='readonly', width=10)
        self.action_cb['values'] = ["down", "up", "left", "right"]
        self.action_cb.current(0)
        self.action_cb.bind("<<ComboboxSelected>>", self.refresh_displayed_images)

        # Create the control bar of the gallery.
        self.gallery.add_control_bar([self.action_label, self.action_cb])
        self.gallery.grid(row=0, column=0, sticky=tk.NSEW)

    def refresh(self):
        """
        Refresh the sample displayed in the galery.
        :return: nothing.
        """
        # Check if the transition model is available.
        if not hasattr(self.gui.model, 'transition'):
            return

        # Get input images.
        in_imgs = self.gallery.get_current_images()
        if in_imgs is None:
            self.gallery.reset()
            return

        # Compute the input and output state representations as well as the output images
        in_states = self.gui.model.encoder(in_imgs)[0]
        actions = torch.ones([in_imgs.shape[0]]) * self.action_cb['values'].index(self.action_cb.get())
        out_states, _ = self.gui.model.transition(in_states, actions)
        out_imgs = self.gui.model.decoder(out_states)

        # Update the gallery.
        self.gallery.refresh({
            "Input image": in_imgs,
            "Output image": out_imgs
        }, {
            "Input latent representation": in_states,
            "Output latent representation": out_states
        })

    def refresh_displayed_images(self, event):
        """
        Refresh the frame.
        :param event: the event that triggered the callback.
        :return: nothing.
        """
        self.refresh()
