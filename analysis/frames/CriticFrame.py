import tkinter as tk
from PIL import Image, ImageTk
import torch
import numpy as np


class CriticFrame(tk.Frame):

    def __init__(self, parent, controller, config, gui_data):
        tk.Frame.__init__(self, parent)

        # Store gui data and empty image
        self.gui_data = gui_data
        self.empty_image = ImageTk.PhotoImage(image=Image.fromarray(torch.zeros([64, 64]).numpy()))

        # Display samples
        self.n_latent = config["agent"]["n_states"]
        self.n_actions = 4
        self.n_samples = 10
        self.curr_index = 0
        self.labels = [[]]
        self.in_img = []
        self.in_img_data = []
        self.g_val_labels = [[]]
        for y in range(0, self.n_samples):
            # Create input image
            label = tk.Label(self, image=self.empty_image)
            label.grid(row=y+1, column=0, sticky=tk.NSEW)
            self.in_img.append(label)
            self.in_img_data.append(self.empty_image)
            # Create latent representation
            for x in range(0, self.n_latent):
                label = tk.Label(self, text="0.000")
                label.grid(row=y+1, column=x+1, sticky=tk.NSEW, padx=10, pady=10)
                self.labels[y].append(label)
            self.labels.append([])
            # Create G values labels
            for x in range(0, self.n_actions):
                label = tk.Label(self, text="0.000")
                label.grid(row=y + 1, column=x + self.n_latent + 2, sticky=tk.NSEW, padx=10, pady=10)
                self.g_val_labels[y].append(label)
            self.g_val_labels.append([])

        # Create separation between left and right side of the frame
        self.grid_columnconfigure(self.n_latent+1, minsize=100)

        # Create the labels
        self.input_img_label = tk.Label(self, text="Input image")
        self.input_img_label.grid(row=0, column=0, sticky=tk.NSEW)
        self.latent_state_label = tk.Label(self, text="Latent representation")
        self.latent_state_label.grid(row=0, column=1, columnspan=self.n_latent, sticky=tk.NSEW)
        self.g_down_label = tk.Label(self, text="G(down)")
        self.g_down_label.grid(row=0, column=self.n_latent+2, sticky=tk.NSEW)
        self.g_up_label = tk.Label(self, text="G(up)")
        self.g_up_label.grid(row=0, column=self.n_latent+3, sticky=tk.NSEW)
        self.g_left_label = tk.Label(self, text="G(left)")
        self.g_left_label.grid(row=0, column=self.n_latent+4, sticky=tk.NSEW)
        self.g_right_label = tk.Label(self, text="G(right)")
        self.g_right_label.grid(row=0, column=self.n_latent+5, sticky=tk.NSEW)

        # Prev button
        self.load_button = tk.Button(
            self, text='prev', height=2, bg='white',
            command=self.display_previous_samples
        )
        self.load_button.grid(row=self.n_samples+2, column=0, sticky=tk.NSEW)

        # Next button
        self.load_button = tk.Button(
            self, text='next', height=2, bg='white',
            command=self.display_next_samples
        )
        self.load_button.grid(row=self.n_samples+2, column=self.n_latent+5, sticky=tk.NSEW)

    def display_previous_samples(self):
        """
        Display the previous set of selected samples.
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
        if self.curr_index >= len(self.gui_data.selected_samples):
            self.curr_index -= self.n_samples
        self.refresh()

    def refresh(self):
        """
        Refresh the sample displayed in the galery.
        :return: nothing.
        """
        # If no samples are selected, return
        if len(self.gui_data.selected_samples) == 0:
            return

        # Retreive the indices of the input images
        in_indices = []
        for y in range(0, self.n_samples):
            index = self.curr_index + y
            if index < len(self.gui_data.selected_samples):
                in_indices.append(self.gui_data.selected_samples[index])
            else:
                break

        # Compute the state representations and output images
        selected_samples = [self.gui_data.samples[i] for i in in_indices]
        in_imgs = torch.cat([img for img, _ in selected_samples])
        in_imgs = torch.unsqueeze(in_imgs, dim=1)
        states, _ = self.gui_data.model.encoder(in_imgs)
        g_values = self.gui_data.model.critic(states)

        # Display the input images, the latent representation and the output images
        for y in range(0, self.n_samples):
            if y < in_imgs.shape[0]:
                self.in_img_data[y] = self.to_photo_image(in_imgs[y])
                self.in_img[y].configure(image=self.in_img_data[y])
                for x in range(0, self.n_latent):
                    self.labels[y][x].configure(text=str(round(states[y][x].item(), 3)))
                for x in range(0, self.n_actions):
                    self.g_val_labels[y][x].configure(text=str(round(g_values[y][x].item(), 3)))
            else:
                self.in_img[y].configure(image=self.empty_image)
                for x in range(0, self.n_latent):
                    self.labels[y][x].configure(text="0.000")
                for x in range(0, self.n_actions):
                    self.g_val_labels[y][x].configure(text="0.000")

    @staticmethod
    def to_photo_image(image):
        """
        Transform the input image into a PhotoImage.
        :param image: a pytorch tensor.
        :return: the PhotoImage.
        """
        image = np.squeeze(image.detach().numpy() * 255)
        return ImageTk.PhotoImage(image=Image.fromarray(image))
