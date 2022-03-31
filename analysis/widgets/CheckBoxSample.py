import tkinter as tk


class CheckBoxSample(tk.Checkbutton):

    def __init__(self, parent, index=-1):
        """
        Construct a checkbox tracking whether a sample is selected or not.
        :param parent: the parent of the clickable image.
        :param index: the sample index.
        """
        self.is_ticked = tk.IntVar()
        super().__init__(parent, variable=self.is_ticked, command=self.on_click)
        self.parent = parent
        self.index = index

    def on_click(self):
        """
        Add or remove the sample from the list of selected samples.
        :return: nothing.
        """
        if self.index == -1:
            self.uncheck()
            return
        if self.is_ticked.get() == 1:
            if self.index not in self.parent.gui_data.selected_samples:
                self.parent.gui_data.selected_samples.append(self.index)
        if self.is_ticked.get() == 0:
            if self.index in self.parent.gui_data.selected_samples:
                self.parent.gui_data.selected_samples.remove(self.index)

    def uncheck(self):
        """
        Uncheck the checkbox.
        :return: nothing.
        """
        self.is_ticked.set(0)

    def check(self):
        """
        Check the checkbox.
        :return: nothing.
        """
        self.is_ticked.set(1)

    def set_index(self, index):
        """
        Change the index of the checkbox.
        :param index: the new index.
        :return: nothing.
        """
        self.index = index
