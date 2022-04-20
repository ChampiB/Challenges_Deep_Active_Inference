import tkinter as tk
from analysis.widgets.dSpritesImageSelector import dSpritesImageSelector


class DisentanglementFrame(tk.Frame):

    def __init__(self, parent, gui):
        tk.Frame.__init__(self, parent)

        # Store GUI.
        self.gui = gui

        # Create label and text for the dimension's name.
        self.label = tk.Label(self, text="Dimension name:")
        self.label.grid(row=1, column=0, sticky=tk.NSEW)
        self.text_dim_name = tk.Text(self, width=1, height=1)
        self.text_dim_name.grid(row=1, column=1, sticky=tk.NSEW)

        # Add button to select add new traversal
        self.add_button = tk.Button(
            self, text='add', height=2, bg=gui.white, width=10,
            command=self.add_new_traversal
        )
        self.add_button.grid(row=1, column=2, sticky=tk.NSEW)

        # Create the image selector.
        self.image_selector = dSpritesImageSelector(self, gui, self.add_target_image_to_traversal)
        self.image_selector.grid(row=0, column=4, rowspan=2)

        # TODO

    def add_new_traversal(self):
        print("add_new_traversal")
        # TODO
        pass

    def add_target_image_to_traversal(self):
        print("add_target_image_to_traversal")
        # TODO
        pass

    def refresh(self):
        """
        Refresh the disentanglement frame.
        :return: nothing.
        """
        # Check that dataset if loaded
        if self.gui.dataset is None:
            return

        # Refresh target image
        self.image_selector.refresh_target_image(None)
