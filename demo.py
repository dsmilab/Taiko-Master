from taiko.gui import *
from taiko.tools.singleton import *


class Demo(metaclass=_Singleton):

    def __init__(self):
        self._window = GUI()
        self._window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._window.mainloop()

    def _on_closing(self):
        self._window.client.clear()
        self._window.destroy()


if __name__ == "__main__":
    Demo()
