import tkinter as tk
from analysis.frames.LoadFrame import LoadFrame
from analysis.frames.EncoderDecoderFrame import EncoderDecoderFrame
from analysis.frames.TransitionFrame import TransitionFrame
from analysis.frames.DatasetFrame import DatasetFrame
from analysis.frames.SampleFrame import SampleFrame
from analysis.frames.VisualisationFrame import VisualisationFrame
from analysis.frames.CriticFrame import CriticFrame
from analysis.frames.DiscriminatorFrame import DiscriminatorFrame
from analysis.frames.CriticWithoutEncoderFrame import CriticWithoutEncoderFrame
from analysis.frames.DisentanglementFrame import DisentanglementFrame
from analysis.widgets.NavBar import NavBar


#
# A class representing the GUI of the model analysis.
#
class AnalysisGUI:

    def __init__(self, config):
        """
        Construct the graphical user interface used to analyse the model.
        :param config: the hydra configuration.
        """

        # Store the hydra configuration.
        self.config = config

        # Load the GUI attributes from the hydra configuration.
        self.n_samples_per_page = config["gallery"]["n_samples_per_page"]
        self.max_latent_dims = config["gui"]["max_latent_dims"]
        self.white = config["colors"]["white"]
        self.red = config["colors"]["red"]
        self.green = config["colors"]["green"]
        self.orange = config["colors"]["orange"]
        self.darkgray = config["colors"]["darkgray"]
        self.lightgray = config["colors"]["lightgray"]

        # Create the main window.
        self.window = tk.Tk()

        self.window.title(config["gui"]["title"])
        self.window.geometry(self.get_screen_size())

        # The model, dataset and sample of the graphical user interface.
        self.model = None
        self.dataset = None
        self.samples = []

        # Create the navigation bar.
        self.navbar = NavBar(self)
        self.window.config(menu=self.navbar)

        # Create the frame container.
        self.container = tk.Frame(self.window)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(2, weight=1)
        self.container.grid_columnconfigure(2, weight=1)

        # The dictionary of frames' constructor.
        self.frames_classes = {
            "LoadFrame": LoadFrame,
            "EncoderDecoderFrame": EncoderDecoderFrame,
            "TransitionFrame": TransitionFrame,
            "CriticFrame": CriticFrame,
            "CriticWithoutEncoderFrame": CriticWithoutEncoderFrame,
            "DiscriminatorFrame": DiscriminatorFrame,
            "DatasetFrame": DatasetFrame,
            "SampleFrame": SampleFrame,
            "VisualisationFrame": VisualisationFrame,
            "DisentanglementFrame": DisentanglementFrame
        }

        # The list of currently loaded frames.
        self.frames = {}
        self.current_frame = None

        # Show the page used to load the model and dataset.
        self.show_frame("LoadFrame")

    def get_screen_size(self):
        """
        Getter.
        :return: the screen' size.
        """
        screen_size = str(self.window.winfo_screenwidth() - 85)
        screen_size += "x"
        screen_size += str(self.window.winfo_screenheight() - 75)
        screen_size += "+85+35"
        return screen_size

    def add_sample(self, sample):
        """
        Add a sample to the gui.
        :param sample: the sample to be added.
        :return: nothing.
        """
        self.samples.append(sample)
        if len(self.samples) == 1:
            self.update_navbar()

    def update_navbar(self):
        """
        Update the navigation bar.
        :return: nothing.
        """
        self.navbar = NavBar(self)
        self.window.config(menu=self.navbar)

    def show_frame(self, frame_name):
        """
        Show a frame for the given frame name.
        :param frame_name: the name of the frame to show.
        :return: nothing.
        """
        # Construct the frame if it does not already exist.
        if frame_name not in self.frames.keys():
            frame = self.frames_classes[frame_name](parent=self.container, gui=self)
            self.frames[frame_name] = frame

        # Display the requested frame.
        if self.current_frame is not None:
            self.current_frame.grid_forget()
        self.current_frame = self.frames[frame_name]
        self.current_frame.grid(row=1, column=1, sticky="")
        self.current_frame.refresh()
        self.current_frame.tkraise()

    def loop(self):
        """
        Launch the main loop of the graphical user interface.
        :return: nothing.
        """
        self.window.mainloop()
