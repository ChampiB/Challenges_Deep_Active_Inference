from tkinter import ttk
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from hydra.utils import instantiate
from singletons.dSpritesDataset import DataSet
from omegaconf import open_dict


#
# A class representing the page used to load the model and dataset.
#
class LoadFrame(tk.Frame):

    def __init__(self, parent, gui):
        """
        Constructor of the home page.
        :param parent: the parent.
        :param gui: the class containing the data of the GUI.
        """

        tk.Frame.__init__(self, parent)

        # Remember parent frame and configuration
        self.gui = gui
        self.parent = parent

        # Model button and label
        self.model_label = tk.Label(self, text="Model:")
        self.model_label.grid(row=0, column=0, sticky=tk.NSEW)
        self.model_button = tk.Button(
            self, text='Select a model...', width=30, height=3, bg=self.gui.red,
            activebackground=self.gui.orange, command=self.open_model_file
        )
        self.model_button.grid(row=0, column=1, sticky=tk.NSEW)

        # Model's number of dimension label and text box
        self.model_n_latent = tk.Label(self, text="Number of latent dimensions:")
        self.model_n_latent.grid(row=0, column=2, sticky=tk.NSEW)

        self.selected_n_latent = tk.StringVar()
        self.n_latent_cb = ttk.Combobox(self, textvariable=self.selected_n_latent, state='readonly', width=10)
        self.n_latent_cb['values'] = [i for i in range(1, gui.max_latent_dims)]
        self.n_latent_cb.current(9)
        self.n_latent_cb.grid(row=0, column=3, sticky=tk.NSEW)

        # Load model button
        self.load_model_button = tk.Button(
            self, text='load', width=20, height=3, bg=self.gui.white,
            command=self.load_model
        )
        self.load_model_button.grid(row=0, column=4, sticky=tk.NSEW)

        # Dataset button and label
        self.dataset_label = tk.Label(self, text="Dataset:")
        self.dataset_label.grid(row=1, column=0, sticky=tk.NSEW)
        self.dataset_button = tk.Button(
            self, text='Select a dataset...', width=30, height=3, bg=self.gui.red,
            activebackground=self.gui.orange, command=self.open_dataset_file
        )
        self.dataset_button.grid(row=1, column=1, sticky=tk.NSEW)

        # Load data button
        self.load_data_button = tk.Button(
            self, text='load', width=20, height=3, bg=self.gui.white,
            command=self.load_dataset
        )
        self.load_data_button.grid(row=1, column=2, sticky=tk.NSEW)

        # Model and dataset files
        self.model_file = None
        self.dataset_file = None

    def open_model_file(self):
        """
        Ask the user to speficy the path of the model's file.
        :return: nothing.
        """
        # Ask user to select the model file
        model_file = fd.askopenfile(title="Select model", filetypes=(
            ('model files', '*.pt'),
            ('All files', '*.*')
        ))

        # If the user did not select any file, return
        if model_file is None:
            return

        # Otherwise update buttun
        self.model_file = model_file
        self.model_button['text'] = self.model_file.name.split('/')[-1]
        self.model_button['bg'] = self.gui.green

    def open_dataset_file(self):
        """
        Ask the user to speficy the path of the model's file.
        :return: nothing.
        """
        # Ask user to select the dataset file
        dataset_file = fd.askopenfile(title="Select dataset", filetypes=(
            ('dataset files', '*.npz'),
            ('All files', '*.*')
        ))

        # If the user did not select any file, return
        if dataset_file is None:
            return

        # Otherwise update buttun
        self.dataset_file = dataset_file
        self.dataset_button['text'] = self.dataset_file.name.split('/')[-1]
        self.dataset_button['bg'] = self.gui.green

    def load_model(self):
        """
        Load the model in the GUI.
        :return: nothing.
        """
        # Display an error if the user did not provide the model file
        if self.model_file is None:
            error_msg = "You must provide the model file before to click the load button."
            messagebox.showerror("Error", error_msg)
            return

        # Load the model and store it in the GUI
        with open_dict(self.gui.config):
            self.gui.config.agent.n_states = int(self.selected_n_latent.get())
        with open_dict(self.gui.config):
            self.gui.config.agent.n_states = int(self.selected_n_latent.get())
        self.gui.model = instantiate(self.gui.config["agent"])
        self.gui.model.load(self.model_file.name)

    def load_dataset(self):
        """
        Load the dataset in the GUI.
        :return: nothing.
        """
        # Display an error if the user did not provide the model or the dataset file.
        if self.dataset_file is None:
            error_msg = "You must provide the dataset file before to click the load button."
            messagebox.showerror("Error", error_msg)
            return

        # Load the dataset and store it in the GUI.
        self.gui.dataset = DataSet.get(self.dataset_file.name)

    def refresh(self):
        pass
