from tkinter import *
import os


class GUI(Tk):

    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.title('Taiko Master v0.3.1')
        self.geometry('400x600')

        """Create Submit Button"""
        self._submitButton = Button(master, text="start")
        self._submitButton.bind('<Button-1>', self.click_start_button)
        self._submitButton.place(x=100, y=100, width=200, height=50)

        """Create stop Button"""
        self._stopButton = Button(master, text="stop")
        self._stopButton.bind('<Button-1>', self.click_stop_button)
        self._stopButton.place_forget()

    def click_start_button(self, event):
        """ handle button click event and output text from entry area"""
        os.system('python ssh_remote.py')
        self._submitButton.place_forget()
        self._stopButton.place(x=100, y=100, width=200, height=50)

    def click_stop_button(self, event):
        """ handle button click event and output text from entry area"""
        os.system('python ssh_remote.py -kill')
        self._stopButton.place_forget()
        self._submitButton.place(x=100, y=100, width=200, height=50)


if __name__ == "__main__":
    window = GUI()
    window.mainloop()
