import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import numpy as np


class TransitionFrame(tk.Frame):

    def __init__(self, parent, controller, config, gui_data):
        tk.Frame.__init__(self, parent)

        # Store gui data and empty image
        self.gui_data = gui_data
        self.empty_image = ImageTk.PhotoImage(image=Image.fromarray(torch.zeros([64, 64]).numpy()))

        # Display samples
        self.n_latent = config["agent"]["n_states"]
        self.n_samples = 10
        self.curr_index = 0
        self.in_labels = [[]]
        self.in_img = []
        self.in_img_data = []
        self.out_labels = [[]]
        self.out_img = []
        self.out_img_data = []
        for y in range(0, self.n_samples):
            # Create input image
            label = tk.Label(self, image=self.empty_image)
            label.grid(row=y+1, column=0, sticky=tk.NSEW)
            self.in_img.append(label)
            self.in_img_data.append(self.empty_image)
            # Create input latent representation
            for x in range(0, self.n_latent):
                label = tk.Label(self, text="0.000")
                label.grid(row=y+1, column=x+1, sticky=tk.NSEW, padx=10, pady=10)
                self.in_labels[y].append(label)
            self.in_labels.append([])
            # Create output image
            label = tk.Label(self, image=self.empty_image)
            label.grid(row=y+1, column=self.n_latent+1, sticky=tk.NSEW)
            self.out_img.append(label)
            self.out_img_data.append(self.empty_image)
            # Create output latent representation
            for x in range(0, self.n_latent):
                label = tk.Label(self, text="0.000")
                label.grid(row=y+1, column=x+self.n_latent+2, sticky=tk.NSEW, padx=10, pady=10)
                self.out_labels[y].append(label)
            self.out_labels.append([])

        # Create the labels
        self.input_img_label = tk.Label(self, text="Input image")
        self.input_img_label.grid(row=0, column=0, sticky=tk.NSEW)
        self.input_latent_state_label = tk.Label(self, text="Input latent representation")
        self.input_latent_state_label.grid(row=0, column=1, columnspan=self.n_latent, sticky=tk.NSEW)
        self.output_img_label = tk.Label(self, text="Output image")
        self.output_img_label.grid(row=0, column=self.n_latent+1, sticky=tk.NSEW)
        self.output_latent_state_label = tk.Label(self, text="Output latent representation")
        self.output_latent_state_label.grid(row=0, column=self.n_latent+2, columnspan=self.n_latent, sticky=tk.NSEW)

        # ComboBox for selecting action
        self.action_label = tk.Label(self, text="Action:")
        self.action_label.grid(row=self.n_samples+2, column=self.n_latent, sticky=tk.NSEW)

        self.selected_action = tk.StringVar()
        self.action_cb = ttk.Combobox(self, textvariable=self.selected_action, state='readonly', width=10)
        self.action_cb['values'] = ["down", "up", "left", "right"]
        self.action_cb.current(0)
        self.action_cb.grid(row=self.n_samples+2, column=self.n_latent+1, sticky=tk.NSEW)
        self.action_cb.bind("<<ComboboxSelected>>", self.refresh_displayed_images)

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
        self.load_button.grid(row=self.n_samples+2, column=2*self.n_latent+1, sticky=tk.NSEW)

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

        # Compute the input and output state representations as well as the output images
        selected_samples = [self.gui_data.samples[i] for i in in_indices]
        in_imgs = torch.cat([img for img, state in selected_samples])
        in_imgs = torch.unsqueeze(in_imgs, dim=1)
        in_states, _ = self.gui_data.model.encoder(in_imgs)
        actions = torch.ones([len(selected_samples)]) * self.action_cb['values'].index(self.action_cb.get())
        out_states, _ = self.gui_data.model.transition(in_states, actions)
        out_imgs = self.gui_data.model.decoder(out_states)

        # Display the input images, the latent representation and the output images
        for y in range(0, self.n_samples):
            if y < in_imgs.shape[0]:
                self.in_img_data[y] = self.to_photo_image(in_imgs[y])
                self.in_img[y].configure(image=self.in_img_data[y])
                for x in range(0, self.n_latent):
                    self.in_labels[y][x].configure(text=str(round(in_states[y][x].item(), 2)))
                self.out_img_data[y] = self.to_photo_image(out_imgs[y])
                self.out_img[y].configure(image=self.out_img_data[y])
                for x in range(0, self.n_latent):
                    self.out_labels[y][x].configure(text=str(round(out_states[y][x].item(), 2)))
            else:
                self.in_img[y].configure(image=self.empty_image)
                for x in range(0, self.n_latent):
                    self.in_labels[y][x].configure(text="0.00")
                self.out_img[y].configure(image=self.empty_image)
                for x in range(0, self.n_latent):
                    self.out_labels[y][x].configure(text="0.00")

    def refresh_displayed_images(self, event):
        """
        Refresh the frame.
        :param event: the event that triggered the callback.
        :return: nothing.
        """
        self.refresh()

    @staticmethod
    def to_photo_image(image):
        """
        Transform the input image into a PhotoImage.
        :param image: a pytorch tensor.
        :return: the PhotoImage.
        """
        image = np.squeeze(image.detach().numpy() * 255)
        return ImageTk.PhotoImage(image=Image.fromarray(image))
