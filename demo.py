from taiko.gui import *
from taiko.interface import *

class Demo(object):

    def __init__(self):
        self._window = GUI()
        self._window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._window.mainloop()

    def _on_closing(self):
        Interface().record_sensor(True)
        self._window.destroy()


if __name__ == "__main__":
    Demo()