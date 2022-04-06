import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import torch


class ClickableImage(tk.Button):

    def __init__(self, parent, image=None, index=-1, width=10, height=10):
        """
        Construct a clickable image.
        :param parent: the parent of the clickable image.
        :param image: the image to be displayed.
        :param index: the image index.
        :param width: the width of the clickable image.
        :param height: the height of the clickable image.
        """
        self.image = torch.zeros([64, 64]).numpy() if image is None else image
        self.image = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        super().__init__(parent, image=self.image, command=self.on_click, activebackground=parent.gui.orange)
        self.configure(width=width, height=height)
        self.parent = parent
        self.index = index
        self.added_to_set = False

    def on_click(self):
        """
        Add the image to the list of image to be added when the image is clicked.
        :return: nothing.
        """
        if self.index == -1:
            return
        if self.added_to_set is False:
            self.parent.images_to_be_added.append(self.index)
            self.configure(bg=self.parent.gui.green)
            self.added_to_set = True
        else:
            self.parent.images_to_be_added.remove(self.index)
            self.configure(bg=self.parent.gui.white)
            self.added_to_set = False

    def set_image(self, image, index):
        """
        Change the image and index of the clickable image.
        :param image: the new image.
        :param index: the new index.
        :return: nothing.
        """
        self.index = index
        self.image = ImageTk.PhotoImage(image=Image.fromarray(np.squeeze(image*255)))
        self.configure(image=self.image, bg=self.parent.gui.white)
        self.added_to_set = False
