from tkinter import *
from taiko.network.client import TaikoClient
import taiko as tk
import threading
import os


class GUI(Tk):

    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.title('Taiko Master v0.3.2')
        self.geometry('1024x768')
        self._client = TaikoClient()
        self._buttons = {
            'start': Button(master, text="start"),
            'stop': Button(master, text="stop"),
            'spider': Button(master, text="spider"),
            'upload': Button(master, text='upload')
        }

        self.__create_start_btn()
        self.__create_stop_btn()
        self.__create_upload_btn()

    def __create_start_btn(self):
        self._buttons['start'].bind('<Button-1>', self.click_start_button)
        self._buttons['start'].place(x=100, y=100, width=200, height=50)

    def __create_stop_btn(self):
        self._buttons['stop'].bind('<Button-1>', self.click_stop_button)
        self._buttons['stop'].place(x=100, y=300, width=200, height=50)

    def __create_spider_btn(self):
        self._buttons['spider'].bind('<Button-1>', self.click_spider_button)
        self._buttons['spider'].place(x=100, y=500, width=200, height=50)

    def __create_upload_btn(self):
        self._buttons['upload'].bind('<Button-1>', self.click_upload_button)
        self._buttons['upload'].place(x=100, y=400, width=200, height=50)

    def click_start_button(self, event):
        self._client.record_sensor()
        self._client.record_screenshot()
        # self._buttons['start'].place_forget()
        self.__create_stop_btn()
        self.__create_spider_btn()

    def click_stop_button(self, event):
        self._client.record_sensor(is_kill=True)
        self._client.record_screenshot(is_kill=True)
        self._client.download_sensor()
        # self._buttons['stop'].place_forget()
        self.__create_start_btn()
        # os.system('python putter.py')

    def click_upload_button(self, event):
        self._client.upload_sensor()
        self._client.upload_screenshot()

    def click_spider_button(self, event):
        os.system('python spider.py')


if __name__ == "__main__":
    window = GUI()
    window.mainloop()
