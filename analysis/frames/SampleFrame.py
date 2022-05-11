import tkinter as tk
from analysis.widgets.GridGallery import GridGallery


#
# Class representing the sample page.
#
class SampleFrame(tk.Frame):

    def __init__(self, parent, gui):
        # Call the constructor of the super class.
        super().__init__(parent)

        # Store gui data.
        self.gui = gui

        # Create the gallery.
        self.gallery = GridGallery(self, gui, self.refresh) \
            .add_image_grid("Images")

        # Create the clear button.
        self.clear_button = tk.Button(
            self.gallery, text='clear', height=2, bg=gui.white,
            command=self.clear_all_samples
        )

        # Create the control bar of the gallery.
        self.gallery.add_control_bar([self.clear_button])
        self.gallery.grid(row=0, column=0, sticky="")

    def clear_all_samples(self):
        """
        Clear all samples.
        :return: nothing.
        """
        self.gui.samples = []
        self.gui.update_navbar()
        self.refresh()

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

        # Update the gallery.
        self.gallery.refresh({"Images": in_imgs}, {})
