import tkinter as tk
from tkinter import ttk
from analysis.widgets.LatentSpaceCanvas import LatentSpaceCanvas


class VisualisationFrame(tk.Frame):

    def __init__(self, parent, gui):
        tk.Frame.__init__(self, parent)

        # Save the number of latent dimensions.
        self.n_latent = gui.config["agent"]["n_states"]

        # Colors
        self.white = gui.config["colors"]["white"]

        # Label and combobox for the x dimension.
        self.label_dim_x = tk.Label(self, text="Dim x:")
        self.label_dim_x.grid(row=0, column=0, sticky=tk.NSEW)

        self.selected_dim_x = tk.StringVar()
        self.cb_dim_x = ttk.Combobox(self, textvariable=self.selected_dim_x)
        self.cb_dim_x['values'] = [str(i) for i in range(0, self.n_latent)]
        self.cb_dim_x['state'] = 'readonly'
        self.cb_dim_x.grid(row=0, column=1, sticky=tk.NSEW)
        self.cb_dim_x.current(0)
        self.cb_dim_x.bind("<<ComboboxSelected>>", self.refresh_callback)

        # Label and combobox for the y dimension.
        self.label_dim_y = tk.Label(self, text="Dim y:")
        self.label_dim_y.grid(row=0, column=2, sticky=tk.NSEW)

        self.selected_dim_y = tk.StringVar()
        self.cb_dim_y = ttk.Combobox(self, textvariable=self.selected_dim_y)
        self.cb_dim_y['values'] = [str(i) for i in range(0, self.n_latent)]
        self.cb_dim_y['state'] = 'readonly'
        self.cb_dim_y.grid(row=0, column=3, sticky=tk.NSEW)
        self.cb_dim_y.current(1)
        self.cb_dim_y.bind("<<ComboboxSelected>>", self.refresh_callback)

        # Add check box to switch between samples and grid.
        self.check_box_val = tk.IntVar(value=1)
        self.check_box = tk.Checkbutton(self, text="Sample/Grid", variable=self.check_box_val, command=self.refresh)
        self.check_box.grid(row=0, column=4, sticky=tk.NSEW)

        # Add default vector.
        self.default_vector_label = tk.Label(self, text="Default vector:")
        self.default_vector_label.grid(row=0, column=5, sticky=tk.NSEW)
        self.default_vector_coords = []
        for i in range(0, self.n_latent):
            coord_box = tk.Text(self, width=1, height=1)
            coord_box.insert('end', "0.0")
            coord_box.grid(row=0, column=6+i, sticky=tk.NSEW)
            self.default_vector_coords.append(coord_box)

        # Create the canvas.
        self.canvas = LatentSpaceCanvas(self, gui)
        self.canvas.grid(row=1, column=0, columnspan=6+self.n_latent, sticky=tk.NSEW, padx=5, pady=5)

    def refresh_callback(self, event):
        self.refresh()

    def refresh(self):
        self.canvas.refresh()
