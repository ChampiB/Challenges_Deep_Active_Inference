from PIL import Image, ImageTk
import numpy as np
import torch
import tkinter as tk


#
# Class representing a gallery in which images and data can be displayed.
#
class Gallery(tk.Frame):

    def __init__(self, parent, gui, refresh_fc):
        """
        Create a gallery to display data and images.
        :param parent: the parent of the gallery.
        :param gui: the graphical user interface.
        :param refresh_fc: the function to call to refresh the gallery.
        """
        # Call constructor of super class.
        super().__init__(parent)

        # Store the graphical user interface and the gallery's parent.
        self.gui = gui

        # The dictionaries storing the images and the label used to display them.
        self.images = {}
        self.images_labels = {}

        # The dictionary storing the label used to display the data.
        self.data_labels = {}

        # The list of labels used to display the columns names.
        self.columns_names = []

        # The elements of the control bar.
        self.prev_button = None
        self.middle_widgets = []
        self.next_button = None

        # Default image.
        self.empty_image = ImageTk.PhotoImage(image=Image.fromarray(torch.zeros([64, 64]).numpy()))

        # Default data element.
        self.empty_data = "0.000"

        # The current column index.
        self.column_id = 0

        # The index of the currently displayed sample.
        self.curr_index = 0

        # Store the refresh function.
        self.refresh_callback = refresh_fc

    def add_image_column(self, col_name):
        """
        Add a column used to display images to the gallery.
        :param col_name: the column's name.
        :return: self.
        """
        # Add the column's name.
        self.__add_column_name(col_name)

        # Initialise the list of images and labels of the new column with default values.
        self.images[col_name] = []
        self.images_labels[col_name] = []
        for y in range(0, self.gui.n_samples_per_page):
            self.images[col_name].append(self.empty_image)
            label = tk.Label(self, image=self.empty_image)
            label.grid(row=y+1, column=self.column_id, sticky=tk.NSEW)
            self.images_labels[col_name].append(label)

        # Increase current column index.
        self.column_id += 1
        return self

    def add_data_column(self, col_name, data_dim, sub_cols_names=None):
        """
        Add a column used to display some numerical data.
        :param col_name: the column name.
        :param data_dim: the dimensionality of the displayed data.
        :param sub_cols_names: the list of the sub columns' names. The number of elements
        in the list is equal to data_dim. If sub_cols_names is None, the col_name is used insteed.
        :return: self.
        """
        # Add the column's name.
        if sub_cols_names is None or len(sub_cols_names) != data_dim:
            self.__add_column_name(col_name, data_dim)
        else:
            for i in range(0, len(sub_cols_names)):
                self.__add_column_name(sub_cols_names[i], shift=i)

        # Initialise the list labels of the new column.
        self.data_labels[col_name] = []

        # Fill up the new column with default values.
        for y in range(0, self.gui.n_samples_per_page):
            self.data_labels[col_name].append([])
            for x in range(0, data_dim):
                label = tk.Label(self, text=self.empty_data)
                label.grid(row=y+1, column=x+self.column_id, sticky=tk.NSEW, padx=5, pady=5)
                self.data_labels[col_name][y].append(label)

        # Increase current column index.
        self.column_id += data_dim
        return self

    def add_empty_column(self):
        """
        Create an empty column.
        :return: self.
        """
        # Set the minimum width of the empty column.
        self.grid_columnconfigure(self.column_id, minsize=100)

        # Increase current column index.
        self.column_id += 1
        return self

    def add_control_bar(self, middle_widgets=None):
        """
        Add the control bar at the bottom of the gallery.
        :param middle_widgets: the widgets to display in the middle of the control bar.
        :return: self.
        """
        # Insert the middle widgets in the middle of the control bar.
        if middle_widgets is not None:
            index = int((self.column_id - len(middle_widgets) + 1) / 2)
            for i in range(0, len(middle_widgets)):
                self.middle_widgets.append(middle_widgets[i])
                self.middle_widgets[i].grid(row=self.gui.n_samples_per_page+2, column=index+i, sticky=tk.NSEW)

        # Create prev button
        self.prev_button = tk.Button(
            self, text='prev', height=2, bg=self.gui.white,
            command=self.__display_previous_samples
        )
        self.prev_button.grid(row=self.gui.n_samples_per_page+2, column=0, sticky=tk.NSEW)

        # Create next button
        self.next_button = tk.Button(
            self, text='next', height=2, bg=self.gui.white,
            command=self.__display_next_samples
        )
        self.next_button.grid(row=self.gui.n_samples_per_page+2, column=self.column_id-1, sticky=tk.NSEW)
        return self

    def get_current_indices(self):
        """
        Getter.
        :return: the current indices of the sample displayed in the gallery.
        """
        indices = []
        for y in range(0, self.gui.n_samples_per_page):
            index = self.curr_index + y
            if index < len(self.gui.samples):
                indices.append(index)
            else:
                break
        return indices

    def get_current_states(self):
        """
        Getter.
        :return: the current states of the sample displayed in the gallery.
        """
        # Reset the current index if it exceeds the number of samples.
        if self.curr_index >= len(self.gui.samples):
            self.curr_index = 0

        # Retreive the indices of the current states.
        indices = self.get_current_indices()

        # If no states are available, return None.
        if len(indices) == 0:
            return None

        # Gather the current states.
        return torch.cat([torch.unsqueeze(self.gui.samples[i][1], dim=0) for i in indices])

    def get_current_images(self):
        """
        Getter.
        :return: the current images of the sample displayed in the gallery.
        """
        # Reset the current index if it exceeds the number of samples.
        if self.curr_index >= len(self.gui.samples):
            self.curr_index = 0

        # Retreive the indices of the current images.
        indices = self.get_current_indices()

        # If no images are available, return None.
        if len(indices) == 0:
            return None

        # Gather the current images.
        return torch.cat([torch.unsqueeze(self.gui.samples[i][0], dim=0) for i in indices])

    def reset_images(self, col_name):
        """
        Reset the images of the specified column to the default value.
        :param col_name: the name of the column to reset.
        :return: nothing.
        """
        for y in range(0, len(self.images[col_name])):
            self.images[col_name][y] = self.empty_image
            self.images_labels[col_name][y].configure(image=self.images[col_name][y])

    def reset_data(self, col_name):
        """
        Reset the data of the specified column to the default value.
        :param col_name: the name of the column to reset.
        :return: nothing.
        """
        for y in range(0, self.gui.n_samples_per_page):
            for x in range(0, len(self.data_labels[col_name][y])):
                self.data_labels[col_name][y][x].configure(text=self.empty_data)

    def reset(self):
        """
        Reset the images and data of all columns.
        :return: nothing.
        """
        # Reset all the images to their default values.
        for col_name, _ in self.images.items():
            self.reset_images(col_name)

        # Reset all the data to their default values.
        for col_name, _ in self.data_labels.items():
            self.reset_data(col_name)

    def refresh(self, img_dict, data_dict):
        """
        Refresh the data displayed
        :param img_dict: the dictionary containing the new images to be displayed.
        :param data_dict: the dictionary containing the new data to be displayed.
        :return: nothing.
        """
        # Display the new images.
        for col_name, imgs in img_dict.items():
            for y in range(0, len(self.images[col_name])):
                new_img = self.__to_photo_image(imgs[y]) if y < imgs.shape[0] else self.empty_image
                self.images[col_name][y] = new_img
                self.images_labels[col_name][y].configure(image=self.images[col_name][y])

        # Display the new data.
        for col_name, data in data_dict.items():
            for y in range(0, self.gui.n_samples_per_page):
                for x in range(0, data.shape[1]):
                    new_data = str(round(data[y][x].item(), 3)) if y < data.shape[0] else self.empty_data
                    self.data_labels[col_name][y][x].configure(text=new_data)

    @staticmethod
    def __to_photo_image(image):
        """
        Transform the input image into a PhotoImage.
        :param image: a pytorch tensor.
        :return: the PhotoImage.
        """
        image = image[0].detach().numpy() * 255
        return ImageTk.PhotoImage(image=Image.fromarray(image))

    def __display_previous_samples(self):
        """
        Display the previous set of selected samples.
        :return: nothing.
        """
        self.curr_index -= self.gui.n_samples_per_page
        self.curr_index = 0 if self.curr_index < 0 else self.curr_index
        self.refresh_callback()

    def __display_next_samples(self):
        """
        Display the next set of samples.
        :return: nothing.
        """
        self.curr_index += self.gui.n_samples_per_page
        if self.curr_index >= len(self.gui.samples):
            self.curr_index -= self.gui.n_samples_per_page
        self.refresh_callback()

    def __add_column_name(self, col_name, columnspan=1, shift=0):
        """
        Add the column name to the gallery.
        :param col_name: the name of the new column.
        :param columnspan: the width of the column.
        :param shift: the shift to apply to the current column index.
        :return: nothing.
        """
        label = tk.Label(self, text=col_name)
        label.grid(row=0, column=self.column_id+shift, columnspan=columnspan, sticky=tk.NSEW)
        self.columns_names.append(label)
