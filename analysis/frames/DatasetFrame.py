import tkinter as tk
from analysis.widgets.ClickableImage import ClickableImage
from analysis.widgets.dSpritesImageSelector import dSpritesImageSelector


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
            self, text='prev', height=2, bg=self.gui.white,
            command=self.display_previous_images
        )
        self.load_button.grid(row=self.height+1, column=0, sticky=tk.NSEW)

        # Next button
        self.load_button = tk.Button(
            self, text='next', height=2, bg=self.gui.white,
            command=self.display_next_images
        )
        self.load_button.grid(row=self.height+1, column=self.width-1, sticky=tk.NSEW)

        # Add button
        self.load_button = tk.Button(
            self, text='add', height=2, bg=self.gui.white,
            command=self.add_images_to_sample
        )
        self.load_button.grid(row=self.height+1, column=int(self.width/2), sticky=tk.NSEW)

        # Create separation between left and right side of the frame
        self.grid_columnconfigure(self.width+1, minsize=100)

        # Create the image selector.
        self.image_selector = dSpritesImageSelector(self, gui, self.add_target_image_to_sample)
        self.image_selector.grid(row=int(self.height/2), column=self.width+2, rowspan=2)

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
        images = self.image_selector.get_images_from_dataset(
            [self.image_selector.get_index_of_target_image()]
        ).repeat(1, self.gui.config["images"]["shape"][0], 1, 1)
        self.gui.add_sample((images[0], None))

    def add_images_to_sample(self):
        """
        Add the selected images to the sample.
        :return: nothing.
        """
        images = self.image_selector.get_images_from_dataset(self.images_to_be_added)
        images = images.repeat(1, self.gui.config["images"]["shape"][0], 1, 1)
        for i in range(0, len(self.images_to_be_added)):
            self.gui.add_sample((images[i], None))
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
        self.image_selector.refresh_target_image(None)
