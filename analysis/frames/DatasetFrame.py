import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch
from analysis.widgets.ClickableImage import ClickableImage


#
# Class representing the dataset page.
#
class DatasetFrame(tk.Frame):

    def __init__(self, parent, gui):
        """
        Create the dataset page.
        :param parent: the parent of the frame.
        :param gui: the gui's data.
        """
        tk.Frame.__init__(self, parent)

        # Remember config, parent and gui data
        self.parent = parent
        self.gui = gui

        # Colors
        self.white = gui.config["colors"]["white"]
        self.green = gui.config["colors"]["green"]
        self.orange = gui.config["colors"]["orange"]

        # The list of indices of all the images that needs to be added to the sample
        self.images_to_be_added = []

        # Attributes for the grid of images
        self.width = 7
        self.height = 7
        self.curr_index = 0
        self.buttons = [[]]
        self.btn_width = 100
        self.btn_height = 100

        # Display grid of empty images
        for x in range(0, self.width):
            for y in range(0, self.height):
                button = ClickableImage(self, width=self.btn_width, height=self.btn_height)
                button.grid(row=x, column=y, sticky=tk.NSEW)
                self.buttons[x].append(button)
            self.buttons.append([])

        # Prev button
        self.load_button = tk.Button(
            self, text='prev', height=2, bg=self.white,
            command=self.display_previous_images
        )
        self.load_button.grid(row=self.height+1, column=0, sticky=tk.NSEW)

        # Next button
        self.load_button = tk.Button(
            self, text='next', height=2, bg=self.white,
            command=self.display_next_images
        )
        self.load_button.grid(row=self.height+1, column=self.width-1, sticky=tk.NSEW)

        # Add button
        self.load_button = tk.Button(
            self, text='add', height=2, bg=self.white,
            command=self.add_images_to_sample
        )
        self.load_button.grid(row=self.height+1, column=int(self.width/2), sticky=tk.NSEW)

        # Create separation between left and right side of the frame
        self.grid_columnconfigure(self.width+1, minsize=100)

        # Add combobox to select specific shape
        self.label_shape = tk.Label(self, text="Shape:")
        self.label_shape.grid(row=self.height+1, column=self.width+2, sticky=tk.NSEW)

        self.selected_shape = tk.StringVar()
        self.shape_cb = ttk.Combobox(self, textvariable=self.selected_shape, state='readonly', width=10)
        self.shape_cb['values'] = ["square", "ellipse", "heart"]
        self.shape_cb.current(0)
        self.shape_cb.grid(row=self.height+2, column=self.width+2, sticky=tk.NSEW)
        self.shape_cb.bind("<<ComboboxSelected>>", self.refresh_target_image)

        # Add combobox to select specific scale
        self.label_scale = tk.Label(self, text="Scale:")
        self.label_scale.grid(row=self.height+1, column=self.width+3, sticky=tk.NSEW)

        self.selected_scale = tk.StringVar()
        self.scale_cb = ttk.Combobox(self, textvariable=self.selected_scale, state='readonly', width=10)
        self.scale_cb['values'] = [str(i) for i in range(0, 6)]
        self.scale_cb.current(0)
        self.scale_cb.grid(row=self.height+2, column=self.width+3, sticky=tk.NSEW)
        self.scale_cb.bind("<<ComboboxSelected>>", self.refresh_target_image)

        # Add combobox to select specific orientation
        self.label_orientation = tk.Label(self, text="Orientation:")
        self.label_orientation.grid(row=self.height+1, column=self.width+4, sticky=tk.NSEW)

        self.selected_orientation = tk.StringVar()
        self.orientation_cb = ttk.Combobox(self, textvariable=self.selected_orientation, state='readonly', width=10)
        self.orientation_cb['values'] = [str(i) for i in range(0, 40)]
        self.orientation_cb.current(0)
        self.orientation_cb.grid(row=self.height+2, column=self.width+4, sticky=tk.NSEW)
        self.orientation_cb.bind("<<ComboboxSelected>>", self.refresh_target_image)

        # Add combobox to select specific x position
        self.label_x_pos = tk.Label(self, text="X position:")
        self.label_x_pos.grid(row=self.height+1, column=self.width+5, sticky=tk.NSEW)

        self.selected_x_pos = tk.StringVar()
        self.x_pos_cb = ttk.Combobox(self, textvariable=self.selected_x_pos, state='readonly', width=10)
        self.x_pos_cb['values'] = [str(i) for i in range(0, 32)]
        self.x_pos_cb.current(0)
        self.x_pos_cb.grid(row=self.height+2, column=self.width+5, sticky=tk.NSEW)
        self.x_pos_cb.bind("<<ComboboxSelected>>", self.refresh_target_image)

        # Add combobox to select specific y position
        self.label_y_pos = tk.Label(self, text="Y position:")
        self.label_y_pos.grid(row=self.height+1, column=self.width+6, sticky=tk.NSEW)

        self.selected_y_pos = tk.StringVar()
        self.y_pos_cb = ttk.Combobox(self, textvariable=self.selected_y_pos, state='readonly', width=10)
        self.y_pos_cb['values'] = [str(i) for i in range(0, 32)]
        self.y_pos_cb.current(0)
        self.y_pos_cb.grid(row=self.height+2, column=self.width+6, sticky=tk.NSEW)
        self.y_pos_cb.bind("<<ComboboxSelected>>", self.refresh_target_image)

        # Add button to select add target image
        self.load_button_2 = tk.Button(
            self, text='add', height=2, bg=self.white, width=10,
            command=self.add_target_image_to_sample
        )
        self.load_button_2.grid(row=self.height+2, column=self.width+7, sticky=tk.NSEW)

        # Add preview image of the targeted image
        self.target_image = torch.zeros([64, 64]).numpy()
        self.target_image = ImageTk.PhotoImage(image=Image.fromarray(self.target_image))

        self.target_image_label = tk.Label(self, image=self.target_image)
        self.target_image_label.grid(row=self.height-1, column=self.width+4, sticky=tk.NSEW)

    def display_previous_images(self):
        """
        Display the previous set of images.
        :return: nothing.
        """
        self.images_to_be_added = []
        self.curr_index -= self.width * self.height
        self.curr_index = 0 if self.curr_index < 0 else self.curr_index
        self.refresh()

    def display_next_images(self):
        """
        Display the next set of images.
        :return: nothing.
        """
        self.images_to_be_added = []
        self.curr_index += self.width * self.height
        if self.curr_index > len(self.gui.dataset[0]):
            self.curr_index -= self.width * self.height
        self.refresh()

    def add_target_image_to_sample(self):
        """
        Add the target image to the sample.
        :return: nothing.
        """
        images = self.get_images_from_dataset([self.get_index_of_target_image()])
        states, _ = self.gui.model.encoder(images)
        self.gui.samples.append((images[0], states[0]))

    def get_index_of_target_image(self):
        """
        Getter.
        :return: the index of the target image.
        """
        # Create latent representation
        state = np.zeros(6)
        state[1] = self.shape_cb['values'].index(self.shape_cb.get())
        state[2] = float(self.scale_cb.get())
        state[3] = float(self.orientation_cb.get())
        state[4] = float(self.x_pos_cb.get())
        state[5] = float(self.y_pos_cb.get())

        # Retreive image's index and image
        return np.dot(state, self.gui.dataset[3]).astype(int)

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
        image = np.squeeze(image * 255)
        self.target_image = ImageTk.PhotoImage(image=Image.fromarray(image.numpy()))

        # Update target image
        self.target_image_label.configure(image=self.target_image)

    def get_images_from_dataset(self, indices):
        """
        Retreive a images from the dataset.
        :param indices: the indices of the images to be retreived.
        :return: the retieved images.
        """
        images = self.gui.dataset[0][indices]
        images = np.moveaxis(images, [0, 1, 2, 3], [0, 2, 3, 1])
        return torch.from_numpy(images).to(torch.float32)

    def add_images_to_sample(self):
        """
        Add the selected images to the sample.
        :return: nothing.
        """
        images = self.get_images_from_dataset(self.images_to_be_added)
        states, _ = self.gui.model.encoder(images)
        for i in range(0, len(self.images_to_be_added)):
            self.gui.samples.append((images[i], states[i]))
        self.refresh()

    def refresh(self):
        """
        Refresh the images displayed in the galery.
        :return: nothing.
        """
        # Check that dataset if loaded
        if self.gui.dataset is None:
            return

        # Refresh image grid
        for x in range(0, self.width):
            for y in range(0, self.height):
                image_id = self.curr_index + x + y * self.width
                image = self.gui.dataset[0][image_id]
                self.buttons[x][y].set_image(image=image, index=image_id)
        self.images_to_be_added = []

        # Refresh target image
        self.refresh_target_image(None)

