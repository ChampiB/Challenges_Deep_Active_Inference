import tkinter as tk
from tkinter import messagebox
import torch
from PIL import Image, ImageTk


class LatentSpaceCanvas(tk.Canvas):

    def __init__(self, parent, gui):
        """
        Construct a canvas displaying the latent space.
        :param parent: the parent of the clickable image.
        :param gui: the gui data to be displayed.
        """

        # Width and height of the canvas.
        self.width = 1820
        self.height = 930

        # Maximum and mimimum value of the latent space we are interested in.
        self.min_x = -10
        self.max_x = 10
        self.min_y = -10
        self.max_y = 10

        # Initial right click position and flag indicating wheter the right click is being pressed.
        self.pos_x = 0
        self.pos_y = 0
        self.right_click_on = False

        super().__init__(parent, width=self.width, height=self.height, bg=gui.white)
        self.parent = parent
        self.gui = gui
        self.images_data = []
        self.bind("<Button-1>", self.on_click)
        self.bind("<Button-3>", self.on_right_click)
        self.bind("<ButtonRelease-3>", self.on_right_click_release)
        self.bind('<Motion>', self.on_motion)

        # with Windows OS
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        # with Linux OS
        self.bind("<Button-4>", self.on_mouse_wheel)
        self.bind("<Button-5>", self.on_mouse_wheel)

        self.configure(cursor="tcross")
        self.refresh()

    def refresh(self):
        """
        Refresh the canvas.
        :return: nothing.
        """
        self.delete("all")
        self.draw_axes()
        if self.parent.check_box_val.get() != 0:
            self.draw_samples()
        else:
            self.draw_grid_samples()

    def draw_grid_samples(self):
        """
        Draw a 2d trversal of the latent space.
        :return: nothing.
        """
        # Check that the model is available
        if self.gui.model is None:
            error_msg = "You must provide the model before to be able add samples when clicking."
            messagebox.showerror("Error", error_msg)
            return

        steps = 10
        x_inc, y_inc = self.compute_increments(steps)

        self.images_data = []
        state = self.get_default_coords()
        x_pos = self.min_x + x_inc
        while x_pos < self.max_x:
            y_pos = self.min_y + y_inc
            while y_pos < self.max_y:
                # Compute the latent coordinate and the corresponding image
                state[0][int(self.parent.selected_dim_x.get())] = x_pos
                state[0][int(self.parent.selected_dim_y.get())] = y_pos
                image = self.to_photo_image(self.gui.model.decoder(state))
                self.images_data.append(image)
                self.draw_image(x_pos, y_pos, image)

                y_pos += y_inc
            x_pos += x_inc

    def get_default_coords(self):
        """
        Retrieve the default coordinate vector.
        :return: the default coordinate vector.
        """
        n_dims = len(self.parent.default_vector_coords)
        state = torch.zeros([1, n_dims])
        for i in range(0, n_dims):
            state[0][i] = float(self.parent.default_vector_coords[i].get("1.0", tk.END))
        return state

    def compute_increments(self, steps):
        """
        Compute the x and y increments.
        :param steps: the total number of step to be taken.
        :return: the x and y increments.
        """
        x_inc = 0
        y_inc = 0
        while x_inc == 0 or y_inc == 0:
            x_inc = (self.max_x - self.min_x) / steps
            y_inc = (self.max_y - self.min_y) / steps
            steps -= 1
            if steps == 0:
                return -1, -1
        return x_inc, y_inc

    def draw_axes(self):
        """
        Draw the axes of the latent space.
        :return: nothing.
        """
        # draw axes
        self.draw_line(self.min_x, 0, self.max_x, 0)
        self.draw_line(0, self.min_y, 0, self.max_y)

        # Compute x and y increments
        steps = 10
        x_inc, y_inc = self.compute_increments(steps)
        if x_inc == -1 and y_inc == -1:
            return

        # Draw axes' numbers
        x_pos = self.min_x + x_inc
        while x_pos < self.max_x:
            if x_pos == 0:
                pass
            self.draw_text(x_pos, -0.02 * (self.max_x - self.min_x), str(round(x_pos, 2)))
            x_pos += x_inc

        y_pos = self.min_y + y_inc
        while y_pos < self.max_y:
            if y_pos == 0:
                pass
            self.draw_text(-0.01 * (self.max_y - self.min_y), y_pos, str(round(y_pos, 2)))
            y_pos += y_inc

    def is_valid_pos(self, x, y):
        """
        Check whether the position (x,y) is valid.
        :param x: the x coordinate.
        :param y: the y coordinate.
        :return: true if (x,y) is a valid position, false otherwise.
        """
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def compute_true_pos(self, x, y):
        """
        Compute the true position of (x,y) in the canvas.
        :param x: the x coordinate in the latent space.
        :param y: the y coordinate in the latent space.
        :return: the x and y coordinate in the canvas.
        """
        ratio_x = (x - self.min_x) / (self.max_x - self.min_x)
        ratio_y = 1 - (y - self.min_y) / (self.max_y - self.min_y)
        return ratio_x * self.width, ratio_y * self.height

    def compute_latent_pos(self, x, y):
        """
        Compute the position of (x,y) in the latent space.
        :param x: the x coordinate in the canvas.
        :param y: the y coordinate in the canvas.
        :return: the x and y coordinate in the latent space.
        """
        ratio_x = x / self.width
        ratio_y = 1 - y / self.height
        latent_x = ratio_x * (self.max_x - self.min_x) + self.min_x
        latent_y = ratio_y * (self.max_y - self.min_y) + self.min_y
        return latent_x, latent_y

    def draw_line(self, x1, y1, x2, y2):
        """
        Draw line between (x1,y1) and (x2,y2).
        :param x1: the first x coordinate.
        :param y1: the first y coordinate.
        :param x2: the second x coordinate.
        :param y2: the second y coordinate.
        :return: nothing.
        """
        # Chech that input coordinate are valid
        if not self.is_valid_pos(x1, y1) or not self.is_valid_pos(x2, y2):
            return

        # Compute true coordinate of x1, x2, y1, and y2
        true_x1, true_y1 = self.compute_true_pos(x1, y1)
        true_x2, true_y2 = self.compute_true_pos(x2, y2)

        # Draw line in canvas
        self.create_line(true_x1, true_y1, true_x2, true_y2)

    def draw_text(self, x, y, text):
        """
        Draw the text at position (x,y).
        :param x: the x coordinate.
        :param y: the y coordinate.
        :param text: the text to be displayed.
        :return: nothing.
        """
        if not self.is_valid_pos(x, y):
            return
        true_x, true_y = self.compute_true_pos(x, y)
        self.create_text(true_x, true_y, text=text)

    def draw_samples(self):
        """
        Draw the samples on the canvas.
        :return: nothing.
        """
        if self.gui.samples is None or len(self.gui.samples) == 0:
            return
        self.images_data = []
        for i in range(0, len(self.gui.samples)):
            image = self.to_photo_image(self.gui.samples[i][0])
            self.images_data.append(image)
            state = self.gui.samples[i][1]
            if state is None:
                img = torch.unsqueeze(self.gui.samples[i][0], dim=0)
                state = self.gui.model.encoder(img)[0][0]
            x = state[int(self.parent.selected_dim_x.get())].item()
            y = state[int(self.parent.selected_dim_y.get())].item()
            self.draw_image(x, y, image)

    @staticmethod
    def to_photo_image(image):
        """
        Transform the input image into a PhotoImage.
        :param image: a pytorch tensor.
        :return: the PhotoImage.
        """
        image = image[0].detach().numpy() * 255
        return ImageTk.PhotoImage(image=Image.fromarray(image))

    def draw_image(self, x, y, img):
        """
        Draw an image.
        :param x: the x coordinate.
        :param y: the y coordinate.
        :param img: the image to be displayed.
        :return: nothing.
        """
        if not self.is_valid_pos(x, y):
            return
        true_x, true_y = self.compute_true_pos(x, y)
        self.create_image(true_x, true_y, image=img, anchor="center")

    def on_mouse_wheel(self, event):
        shift = 0
        if event.num == 5 or event.delta == -120:
            shift = 1
        if event.num == 4 or event.delta == 120:
            shift = -1

        self.min_x += shift
        self.max_x -= shift
        if self.min_x >= self.max_x:
            self.min_x -= shift
            self.max_x += shift

        self.min_y += shift
        self.max_y -= shift
        if self.min_y >= self.max_y:
            self.min_y -= shift
            self.max_y += shift

        self.refresh()

    def on_click(self, event):
        """
        Add a sample as requested by the user's click.
        :param event: the event describing the user's click.
        :return: nothing.
        """
        # Retreive latent state
        state = self.get_default_coords()
        latent_x, latent_y = self.compute_latent_pos(event.x, event.y)
        state[0][int(self.parent.selected_dim_x.get())] = latent_x
        state[0][int(self.parent.selected_dim_y.get())] = latent_y

        # Retreive observation from state
        obs = self.gui.model.decoder(state) * 255

        # Add the state-observation pair to the samples of the GUI
        self.gui.add_sample((obs[0].detach(), state[0]))

        # Refresh the GUI
        self.refresh()

    def on_right_click(self, event):
        """
        Save mouse position when left click.
        :param event: the event that triggered the call to this function.
        :return: nothing.
        """
        self.pos_x = event.x
        self.pos_y = event.y
        self.right_click_on = True

    def on_right_click_release(self, event):
        """
        Update class attribute to make sure to stop movement of the displayed area.
        :param event: the event that triggered the call to this function.
        :return: nothing.
        """
        self.right_click_on = False

    def on_motion(self, event):
        """
        Move the axes according to the users request.
        :param event: the event that triggered the call to this function.
        :return: nothing.
        """
        if self.right_click_on:
            # Compute the delta in x and y
            old_x, old_y = self.compute_latent_pos(self.pos_x, self.pos_y)
            new_x, new_y = self.compute_latent_pos(event.x, event.y)

            delta_x = old_x - new_x
            delta_y = old_y - new_y

            # Apply delta to min and max values
            self.min_x += delta_x
            self.max_x += delta_x
            self.min_y += delta_y
            self.max_y += delta_y

            # Update last position of the mouse
            self.pos_x = event.x
            self.pos_y = event.y

            # Refresh the GUI
            self.refresh()
