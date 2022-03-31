import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from hydra.utils import instantiate
from singletons.dSpritesDataset import DataSet


#
# A class representing the home page.
#
class HomeFrame(tk.Frame):

    def __init__(self, parent, controller, config, gui_data):
        """
        Constructor of the home page.
        :param parent: the parent.
        :param controller: the controller.
        :param config: the hydra config.
        :param gui_data: the class containing the data of the GUI.
        """

        tk.Frame.__init__(self, parent)

        # Remember parent frame and configuration
        self.gui_data = gui_data
        self.parent = parent
        self.config = config

        # Colors
        self.red = "#e00000"
        self.green = "#1d7500"
        self.orange = "#de9b00"

        # Model button and label
        self.model_label = tk.Label(self, text="Model:")
        self.model_label.grid(row=0, column=0, sticky=tk.NSEW)
        self.model_button = tk.Button(
            self, text='Select a model...', width=30, height=3, bg=self.red,
            activebackground=self.orange, command=self.open_model_file
        )
        self.model_button.grid(row=0, column=1, sticky=tk.NSEW)

        # Dataset button and label
        self.dataset_label = tk.Label(self, text="Dataset:")
        self.dataset_label.grid(row=1, column=0, sticky=tk.NSEW)
        self.dataset_button = tk.Button(
            self, text='Select a dataset...', width=30, height=3, bg=self.red,
            activebackground=self.orange, command=self.open_dataset_file
        )
        self.dataset_button.grid(row=1, column=1, sticky=tk.NSEW)

        # Load button
        self.load_button = tk.Button(
            self, text='load', width=20, height=3, bg='white',
            command=self.load_data_and_model
        )
        self.load_button.grid(row=2, column=1, sticky=tk.NSEW)

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
        self.model_button['bg'] = self.green

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
        self.dataset_button['bg'] = self.green

    def load_data_and_model(self):
        """
        Load the dataset and the model in the GUI.
        :return: nothing.
        """
        # Display an error if the user did not provide the model or the dataset file
        if self.model_file is None or self.dataset_file is None:
            error_msg = "You must provide the model and dataset files before to click the load button."
            messagebox.showerror("Error", error_msg)
            return

        # Load and save the model and dataset in the GUI
        self.gui_data.model = instantiate(self.config["agent"])
        self.gui_data.model.load(self.model_file.name)
        self.gui_data.dataset = DataSet.get(self.dataset_file.name)

    def refresh(self):
        pass
