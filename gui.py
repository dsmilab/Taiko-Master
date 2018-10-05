from tkinter import *
from taiko.client import TaikoClient
import platform

if platform.system() == 'Windows':
    ACTIVE = 'normal'
elif platform.system() == 'Linux':
    ACTIVE = 'active'


class GUI(Tk):

    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.title('Taiko Master v0.3.2')
        self.geometry('800x600')
        self._client = TaikoClient()
        self._stage = 0

        self.__init_window()
        self.switch_screen('_StartScreen')

    def __init_window(self):
        container = Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._screens = {}
        for scr in [_StartScreen, _RunScreen, _ResultScreen]:
            scr_name = scr.__name__
            screen = scr(parent=container, controller=self)
            screen.grid(row=0, column=0, sticky="nsew")
            self._screens[scr_name] = screen

    def switch_screen(self, scr_name):
        screen = self._screens[scr_name]
        screen.tkraise()


class _StartScreen(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}
        self._images = {}
        self._var = {}
        self._labels = {}
        self._entries = {}

        self._selected_difficulty = 'easy'
        self.__init_screen()

    def __init_screen(self):
        self.__create_buttons()
        self.__create_entry_tips()
        self.__refresh_difficulty_buttons()

    def __create_buttons(self):
        self._buttons['start'] = Button(self, text='start')
        self._buttons['start'].bind('<Button-1>', self.__click_start_button)
        self._buttons['start'].place(x=300, y=520, width=250, height=70)

        self._var['difficulty'] = StringVar()
        for i_, difficulty in enumerate(['easy', 'normal', 'hard', 'extreme']):
            self._images[difficulty] = PhotoImage(file='data/pic/' + difficulty + '.png')
            self._buttons[difficulty] = Button(self,
                                               image=self._images[difficulty],
                                               text=difficulty)
            self._buttons[difficulty].bind('<Button-1>', self.__click_difficulty_button)
            self._buttons[difficulty].place(x=130 + i_ * 140, y=300, width=120, height=120)

    def __create_entry_tips(self):
        self._labels['drummer_name'] = Label(self, text='drummer\'s name:')
        self._labels['drummer_name'].place(x=40, y=10, height=80)
        self._labels['drummer_name'].config(font=("Helvetica", 30))

        self._entries['drummer_name'] = Entry(self, bg='lightgray')
        self._entries['drummer_name'].place(x=40, y=80, height=80, width=300)
        self._entries['drummer_name'].config(font=("Helvetica", 30))

        self._labels['song_id'] = Label(self, text='song ID:')
        self._labels['song_id'].place(x=450, y=80)
        self._labels['song_id'].config(font=("Helvetica", 25))

        self._entries['song_id'] = Entry(self, bg='lightblue')
        self._entries['song_id'].place(x=600, y=80, height=40, width=70)
        self._entries['song_id'].config(font=("Helvetica", 20))

        self._labels['difficulty'] = Label(self, text='difficulty')
        self._labels['difficulty'].place(x=330, y=250)
        self._labels['difficulty'].config(font=("Times", 20))

    def __click_difficulty_button(self, e):
        self._selected_difficulty = e.widget['text']
        self.__refresh_difficulty_buttons()

    def __refresh_difficulty_buttons(self):
        for i_, difficulty in enumerate(['easy', 'normal', 'hard', 'extreme']):
            self._buttons[difficulty].configure(bd=5, bg='snow')
        self._buttons[self._selected_difficulty].configure(bd=5, bg='red')

    def __click_start_button(self, e):
        self._controller.switch_screen('_RunScreen')


class _RunScreen(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}
        self.__init_screen()

    def __init_screen(self):
        self.__create_stop_button()

    def __create_stop_button(self):
        self._buttons['stop'] = Button(self, text='stop')
        self._buttons['stop'].bind('<Button-1>', self.click_stop_button)
        self._buttons['stop'].place(x=300, y=520, width=250, height=70)

    def click_stop_button(self, event):
        self._controller.switch_screen('_ResultScreen')


class _ResultScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}

        self.__init_screen()

    def __init_screen(self):
        self.__create_back_button()

    def __create_back_button(self):
        self._buttons['back'] = Button(self, text='back')
        self._buttons['back'].bind('<Button-1>', self.click_back_button)
        self._buttons['back'].place(x=520, y=520, width=250, height=70)

    def click_back_button(self, event):
        self._controller.switch_screen('_StartScreen')


if __name__ == "__main__":
    window = GUI()
    window.mainloop()
