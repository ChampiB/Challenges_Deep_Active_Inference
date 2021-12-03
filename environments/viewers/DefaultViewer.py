from tkinter import *
from PIL import Image, ImageTk
import numpy as np


#
# Class implementing a default viewer displaying an environment visually.
#
class DefaultViewer:

    def __init__(self, title, reward, img, image_size=(200, 200), resize_type=Image.ANTIALIAS):
        """
        Constructor.
        :param title: the window's title.
        :param reward: the current reward received by the agent.
        :param img: the current observation received by the agent.
        :param image_size: the size of the image to display.
        :param resize_type: the type of resize to perform.
        """

        # Size of the images to display.
        self.image_size = image_size
        self.resize_type = resize_type

        # Create the root window.
        self.root = Tk()
        self.root.title(title)

        # Add the image to the root window.
        img = self.to_photo_image(img)
        self.img_label = Label(self.root, image=img)
        self.img_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        # Add the reward to the root window.
        self.text_label = Label(self.root, text="Reward:")
        self.text_label.grid(row=1, column=0, padx=10, pady=10)

        self.reward_label_data = StringVar()
        self.reward_label_data.set(str(reward))
        self.reward_label = Label(self.root, textvariable=self.reward_label_data)
        self.reward_label.grid(row=1, column=1, padx=10, pady=10)

        # Refresh the main window.
        self.root.update()

    def update(self, reward, img):
        """
        Update the viewer.
        :param reward: the new reward to display.
        :param img: the new observation to display.
        :return: nothing.
        """

        # Update the image.
        img = self.to_photo_image(img)
        self.img_label.configure(image=img)
        self.img_label.image = img

        # Update the reward.
        self.reward_label_data.set(str(reward))

        # Refresh the main window.
        self.root.update()

    def to_photo_image(self, img):
        """
        Returns the input image as an PhotoImage, i.e. the format require for display.
        :return: the formatted input image required by pillow and tkinter for render.
        """
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize(self.image_size, self.resize_type)
        return ImageTk.PhotoImage(img)
