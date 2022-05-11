import tkinter as tk
import torch
from analysis.widgets.Gallery import Gallery
from environments.dSpritesEnv import dSpritesEnv


#
# Class representing the page displaying the output of the critic.
#
class CriticWithoutEncoderFrame(tk.Frame):

    def __init__(self, parent, gui):
        """
        Construct the page displaying the output of the critic.
        :param parent: the parent of the page.
        :param gui: the graphical user interface.
        """
        # Call the constructor of the super class.
        super().__init__(parent)

        # Store gui data.
        self.gui = gui

        # Store the number of actions.
        self.n_actions = gui.config["env"]["n_actions"]

        # Create the gallery.
        self.gallery = Gallery(self, gui, self.refresh) \
            .add_image_column("Input image") \
            .add_empty_column() \
            .add_data_column("G values", self.n_actions, ["G(down)", "G(up)", "G(left)", "G(right)"]) \
            .add_empty_column() \
            .add_data_column("P actions", self.n_actions, ["P(down)", "P(up)", "P(left)", "P(right)"]) \
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

        # Get input states.
        states = self.gallery.get_current_states()
        if states is None:
            self.gallery.reset()
            return

        # Compute the state representations, the G-values and the probability of each action.
        states = states.to(torch.int64)
        states = [dSpritesEnv.state_to_one_hot(states[i]) for i in range(0, states.shape[0])]
        states = [torch.unsqueeze(state, dim=0) for state in states]
        states = torch.cat(states)
        g_values = self.gui.model.critic(states)
        p_actions = torch.softmax(g_values, dim=1)

        # Update the gallery.
        self.gallery.refresh({
            "Input image": in_imgs
        }, {
            "G values": g_values,
            "P actions": p_actions,
        })
