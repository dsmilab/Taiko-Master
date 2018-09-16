from tkinter import *
import os


class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()

        """Create Submit Button"""
        self.submitButton = Button(master, command=self.click_start_button, text="start", height=5, width=15)
        self.submitButton.grid()
        """Create stop Button"""
        self.stopButton = Button(master, command=self.click_stop_button, text="stop")
        self.stopButton.grid()

    @staticmethod
    def click_start_button():
        """ handle button click event and output text from entry area"""
        os.system('python ssh_remote.py')

    @staticmethod
    def click_stop_button():
        """ handle button click event and output text from entry area"""
        os.system('python ssh_remote.py -kill')


if __name__ == "__main__":
    guiFrame = GUI()
    guiFrame.mainloop()
