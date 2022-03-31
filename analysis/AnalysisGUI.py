import tkinter as tk
from analysis.frames.HomeFrame import HomeFrame
from analysis.frames.EncoderDecoderFrame import EncoderDecoderFrame
from analysis.frames.TransitionFrame import TransitionFrame
from analysis.frames.DatasetFrame import DatasetFrame
from analysis.frames.SampleFrame import SampleFrame
from analysis.frames.VisualisationFrame import VisualisationFrame
from analysis.frames.CriticFrame import CriticFrame


#
# A class representing the GUI of the model analysis.
#
class AnalysisGUI:

    def __init__(self, config):
        # Store config
        self.config = config

        # Create main window
        self.window = tk.Tk()
        self.window.title("Model analysis")

        screen_size = str(self.window.winfo_screenwidth())
        screen_size += "x"
        screen_size += str(self.window.winfo_screenheight())
        self.window.geometry(screen_size)

        # Create navigation bar
        self.navbar = tk.Menu(self.window)

        self.navbar.add_command(label="Home", command=self.home_cmd)

        self.modelbar = tk.Menu(self.navbar, tearoff=0)
        self.modelbar.add_command(label="Encoder/Decoder", command=self.encoder_decoder_cmd)
        self.modelbar.add_command(label="Transition", command=self.transition_cmd)
        self.modelbar.add_command(label="Critic", command=self.critic_cmd)
        self.navbar.add_cascade(label="Model", menu=self.modelbar)

        self.navbar.add_command(label="Dataset", command=self.dataset_cmd)

        self.navbar.add_command(label="Sample", command=self.sample_cmd)
        self.navbar.add_command(label="Visualisation", command=self.visualisation_cmd)

        self.window.config(menu=self.navbar)

        # The GUI's model, dataset and sample
        self.model = None
        self.dataset = None
        self.samples = []
        self.selected_samples = []

        # Create the GUI's frames
        self.container = tk.Frame(self.window)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.show_frame("HomeFrame")

    def show_frame(self, frame_name):
        """
        Show a frame for the given frame name.
        :param frame_name: the name of the frame to show.
        :return: nothing.
        """
        # Construct the frame that does not already exist
        if frame_name not in self.frames.keys():
            frames = {
                "HomeFrame": HomeFrame,
                "EncoderDecoderFrame": EncoderDecoderFrame,
                "TransitionFrame": TransitionFrame,
                "CriticFrame": CriticFrame,
                "DatasetFrame": DatasetFrame,
                "SampleFrame": SampleFrame,
                "VisualisationFrame": VisualisationFrame
            }
            frame = frames[frame_name](parent=self.container, controller=self.window, config=self.config, gui_data=self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[frame_name] = frame

        # Display the requested frame
        frame = self.frames[frame_name]
        frame.refresh()
        frame.tkraise()

    def home_cmd(self):
        self.show_frame("HomeFrame")

    def encoder_decoder_cmd(self):
        self.show_frame("EncoderDecoderFrame")

    def transition_cmd(self):
        self.show_frame("TransitionFrame")

    def critic_cmd(self):
        self.show_frame("CriticFrame")

    def dataset_cmd(self):
        self.show_frame("DatasetFrame")

    def sample_cmd(self):
        self.show_frame("SampleFrame")

    def visualisation_cmd(self):
        self.show_frame("VisualisationFrame")

    def loop(self):
        self.window.mainloop()
