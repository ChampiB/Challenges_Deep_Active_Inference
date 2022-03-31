import tkinter as tk
import torch
import numpy as np
from PIL import Image, ImageTk
from analysis.widgets.CheckBoxSample import CheckBoxSample


#
# Class representing the sample page.
#
class SampleFrame(tk.Frame):

    def __init__(self, parent, controller, config, gui_data):
        tk.Frame.__init__(self, parent)

        # Save gui data
        self.gui_data = gui_data
        self.empty_image = ImageTk.PhotoImage(image=Image.fromarray(torch.zeros([64, 64]).numpy()))

        # Display samples
        self.n_latent = config["agent"]["n_states"]
        self.n_samples = 10
        self.curr_index = 0
        self.labels = [[]]
        self.images = []
        self.images_data = []
        self.check_boxes = []
        for y in range(0, self.n_samples):
            for x in range(0, self.n_latent):
                label = tk.Label(self, text="0.000")
                label.grid(row=y, column=x, sticky=tk.NSEW, padx=10, pady=10)
                self.labels[y].append(label)
            self.labels.append([])
            label = tk.Label(self, image=self.empty_image)
            label.grid(row=y, column=self.n_latent, sticky=tk.NSEW)
            self.images.append(label)
            self.images_data.append(self.empty_image)
            check_box = CheckBoxSample(self)
            check_box.grid(row=y, column=self.n_latent+1, sticky=tk.NSEW)
            self.check_boxes.append(check_box)

        # Prev button
        self.load_button = tk.Button(
            self, text='prev', height=2, bg='white',
            command=self.display_previous_samples
        )
        self.load_button.grid(row=self.n_samples + 1, column=0, sticky=tk.NSEW)

        # Next button
        self.load_button = tk.Button(
            self, text='next', height=2, bg='white',
            command=self.display_next_samples
        )
        self.load_button.grid(row=self.n_samples + 1, column=self.n_latent, sticky=tk.NSEW)

        # Clear button
        self.load_button = tk.Button(
            self, text='clear', height=2, bg='white',
            command=self.clear_all_samples
        )
        self.load_button.grid(row=self.n_samples + 1, column=int(self.n_latent/2), sticky=tk.NSEW)

    def display_previous_samples(self):
        """
        Display the previous set of samples.
        :return: nothing.
        """
        self.curr_index -= self.n_samples
        self.curr_index = 0 if self.curr_index < 0 else self.curr_index
        self.refresh()

    def display_next_samples(self):
        """
        Display the next set of samples.
        :return: nothing.
        """
        self.curr_index += self.n_samples
        if self.curr_index >= len(self.gui_data.samples):
            self.curr_index -= self.n_samples
        self.refresh()

    def clear_all_samples(self):
        """
        Clear all samples.
        :return: nothing.
        """
        self.gui_data.samples = []
        self.gui_data.selected_samples = []
        for check_box in self.check_boxes:
            check_box.set_index(-1)
        self.refresh()

    def refresh(self):
        """
        Refresh the sample displayed in the galery.
        :return: nothing.
        """
        if len(self.labels) == 0:
            return
        for y in range(0, self.n_samples):
            index = self.curr_index + y
            if index < len(self.gui_data.samples):
                image = self.gui_data.samples[index][0]
                image = np.squeeze(image.numpy()*255)
                image = ImageTk.PhotoImage(image=Image.fromarray(image))
                state = self.gui_data.samples[index][1]
                for x in range(0, self.n_latent):
                    self.labels[y][x].configure(text=str(round(state[x].item(), 3)))
                self.images_data[y] = image
                self.images[y].configure(image=self.images_data[y])
                if index in self.gui_data.selected_samples:
                    self.check_boxes[y].check()
                else:
                    self.check_boxes[y].uncheck()
                self.check_boxes[y].set_index(index)
            else:
                for x in range(0, self.n_latent):
                    self.labels[y][x].configure(text="0.000")
                self.images[y].configure(image=self.empty_image)
                self.check_boxes[y].uncheck()
                self.check_boxes[y].set_index(-1)

