import tkinter as tk
from analysis.widgets.Gallery import Gallery


#
# Class representing the page displaying the output of the encoder and decoder networks.
#
class EncoderDecoderFrame(tk.Frame):

    def __init__(self, parent, gui):
        """
        Construct the page displaying the output of the encoder and decoder networks.
        :param parent: the parent of the page.
        :param gui: the graphical user interface.
        """
        # Call the constructor of the super class.
        super().__init__(parent)

        # Store gui data.
        self.gui = gui

        # Create the gallery.
        self.gallery = Gallery(self, gui, self.refresh) \
            .add_image_column("Input image") \
            .add_data_column("Latent representation", gui.config["agent"]["n_states"]) \
            .add_image_column("Output image") \
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

        # Compute the state representations and output images
        states = self.gui.model.encoder(in_imgs)[0]
        out_imgs = self.gui.model.decoder(states)

        # Update the gallery.
        self.gallery.refresh({
            "Input image": in_imgs,
            "Output image": out_imgs
        }, {
            "Latent representation": states
        })
