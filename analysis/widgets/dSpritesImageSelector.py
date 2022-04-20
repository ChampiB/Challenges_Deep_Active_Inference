import tkinter as tk
import numpy as np
import torch
from tkinter import ttk
from PIL import Image, ImageTk


#
# Class allowing the user to select targeted images from the dSprites dataset.
#
class dSpritesImageSelector(tk.Frame):

    def __init__(self, parent, gui, callback):
        """
        Construct the main navigation bar.
        :param parent: the parent of the gallery.
        :param gui: the graphical user interface.
        :param callback: the function to be called when an image is selected.
        """
        # Call super class contructor.
        super().__init__(parent)

        # Store the graphical user interface, parent and callback.
        self.gui = gui
        self.parent = parent
        self.callback = callback

        # Class attributes.
        self.current_column = 0
        self.labels = []
        self.cboxes = []
        self.selected_values = []

        # Add value selector for each dimension of the dSprites dataset.
        self.add_value_selector(text="Shape:", values=["square", "ellipse", "heart"])
        self.add_value_selector(text="Scale:", values=[str(i) for i in range(0, 6)])
        self.add_value_selector(text="Orientation:", values=[str(i) for i in range(0, 40)])
        self.add_value_selector(text="X position:", values=[str(i) for i in range(0, 32)])
        self.add_value_selector(text="Y position:", values=[str(i) for i in range(0, 32)])

        # Add button to select add target image
        self.load_button = tk.Button(
            self, text='add', height=2, bg=self.gui.white, width=10,
            command=self.callback
        )
        self.load_button.grid(row=2, column=7, sticky=tk.NSEW)

        # Add preview image of the targeted image
        self.target_image = torch.zeros([64, 64]).numpy()
        self.target_image = ImageTk.PhotoImage(image=Image.fromarray(self.target_image))

        self.target_image_label = tk.Label(self, image=self.target_image)
        self.target_image_label.grid(row=0, column=2, sticky=tk.NSEW)

    def add_value_selector(self, text, values):
        # Create label explaining which value is being selected.
        label = tk.Label(self, text=text)
        label.grid(row=1, column=self.current_column, sticky=tk.NSEW)
        self.labels.append(label)

        # Create variable storing the selected value.
        selected_val = tk.StringVar()
        self.selected_values.append(selected_val)

        # Create the combo box allowing the user to select a targeted value.
        cbox = ttk.Combobox(self, textvariable=selected_val, state='readonly', width=10)
        cbox['values'] = values
        cbox.current(0)
        cbox.grid(row=2, column=self.current_column, sticky=tk.NSEW)
        cbox.bind("<<ComboboxSelected>>", self.refresh_target_image)
        self.cboxes.append(cbox)

        # Increase current column index.
        self.current_column += 1

    def get_state(self):
        """
        Getter.
        :return: the latent representation of the selected image.
        """
        # Create latent representation
        state = np.zeros(6)
        for i in range(0, len(self.labels)):
            state[i + 1] = self.cboxes[i]['values'].index(self.cboxes[i].get())
        return state

    def refresh_target_image(self, event):
        """
        Refresh the preview of the targeted image.
        :param event: the event that triggered the call of this function.
        :return: nothing.
        """

        # Check that dataset is available
        if self.gui.dataset is None:
            return

        # Retreive the index of the target image
        i = self.get_index_of_target_image()

        # Retreive target image
        image = self.get_images_from_dataset([i])
        image = np.squeeze(image.numpy() * 255)
        self.target_image = ImageTk.PhotoImage(image=Image.fromarray(image))

        # Update target image
        self.target_image_label.configure(image=self.target_image)

    def get_index_of_target_image(self):
        """
        Getter.
        :return: the index of the target image.
        """
        # Retreive image's index and image
        return np.dot(self.get_state(), self.gui.dataset[3]).astype(int)

    def get_images_from_dataset(self, indices):
        """
        Retreive an images from the dataset.
        :param indices: the indices of the images to be retreived.
        :return: the retieved images.
        """
        images = self.gui.dataset[0][indices]
        images = np.moveaxis(images, [0, 1, 2, 3], [0, 2, 3, 1])
        return torch.from_numpy(images).to(torch.float32)
