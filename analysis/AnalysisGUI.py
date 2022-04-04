import tkinter as tk
from analysis.frames.LoadFrame import LoadFrame
from analysis.frames.EncoderDecoderFrame import EncoderDecoderFrame
from analysis.frames.TransitionFrame import TransitionFrame
from analysis.frames.DatasetFrame import DatasetFrame
from analysis.frames.SampleFrame import SampleFrame
from analysis.frames.VisualisationFrame import VisualisationFrame
from analysis.frames.CriticFrame import CriticFrame
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

        # Create the main window.
        self.window = tk.Tk()
        self.window.title(config["gui"]["title"])
        self.window.geometry(self.get_screen_size())

        # Create the navigation bar.
        self.navbar = NavBar(self)
        self.window.config(menu=self.navbar)

        # The model, dataset and sample of the graphical user interface.
        self.model = None
        self.dataset = None
        self.samples = []

        # Create the frame container.
        self.container = tk.Frame(self.window)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # The dictionary of frames' constructor.
        self.frames_classes = {
            "LoadFrame": LoadFrame,
            "EncoderDecoderFrame": EncoderDecoderFrame,
            "TransitionFrame": TransitionFrame,
            "CriticFrame": CriticFrame,
            "DatasetFrame": DatasetFrame,
            "SampleFrame": SampleFrame,
            "VisualisationFrame": VisualisationFrame
        }

        # The list of currently loaded frames.
        self.frames = {}

        # Show the page used to load the model and dataset.
        self.show_frame("LoadFrame")

    def get_screen_size(self):
        """
        Getter.
        :return: the screen' size.
        """
        screen_size = str(self.window.winfo_screenwidth())
        screen_size += "x"
        screen_size += str(self.window.winfo_screenheight())
        return screen_size

    def show_frame(self, frame_name):
        """
        Show a frame for the given frame name.
        :param frame_name: the name of the frame to show.
        :return: nothing.
        """
        # Construct the frame if it does not already exist.
        if frame_name not in self.frames.keys():
            frame = self.frames_classes[frame_name](parent=self.container, gui=self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[frame_name] = frame

        # Display the requested frame.
        frame = self.frames[frame_name]
        frame.refresh()
        frame.tkraise()

    def loop(self):
        """
        Launch the main loop of the graphical user interface.
        :return: nothing.
        """
        self.window.mainloop()
