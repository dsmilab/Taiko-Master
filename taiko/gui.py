from .client import *
from .tools.validate import *

from tkinter import *
from tkinter import ttk

import platform
import threading
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# if platform.system() == 'Windows':
#     ACTIVE = 'normal'
# elif platform.system() == 'Linux':
#     ACTIVE = 'normal'


class GUI(Tk):

    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.title('Taiko Master v0.3.3')
        self.geometry('800x600')
        self.resizable(width=False, height=False)
        self._stage = 0
        self._client = TaikoClient()
        self._screen = None

        self.__init_window()

    def __init_window(self):
        self._container = Frame(self)
        self._container.pack(side='top', fill='both', expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)
        self._switch_screen(_StartScreen)

    def goto_next_screen(self, now_scr):
        if now_scr == _StartScreen:
            self._switch_screen(_RunScreen)
        elif now_scr == _RunScreen:
            try:
                self._switch_screen(_LoadingScreen)
            except (KeyError, TypeError):
                self._switch_screen(_ErrorScreen)
        elif now_scr == _LoadingScreen:
            self._switch_screen(_ResultScreen)
        elif now_scr == _ResultScreen:
            self._switch_screen(_StartScreen)
        elif now_scr == _ErrorScreen:
            self._switch_screen(_StartScreen)

    def _switch_screen(self, scr):
        if self._screen:
            self._screen.grid_forget()
            self._screen.destroy()

        self._screen = scr(parent=self._container, controller=self)
        self._screen.grid(row=0, column=0, sticky="nsew")
        self._screen.tkraise()

    @property
    def client(self):
        return self._client


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
        self._controller.client.clear()

    def __create_buttons(self):
        self._buttons['start'] = Button(self, text='start')
        self._buttons['start'].bind('<Button-1>', self.__click_start_button)
        self._buttons['start'].place(x=280, y=520, width=250, height=70)

        self._var['difficulty'] = StringVar()
        for i_, difficulty in enumerate(self._controller.client.DIFFICULTIES):
            self._images[difficulty] = PhotoImage(file=self._controller.client.pic_path[difficulty])
            self._buttons[difficulty] = Button(self,
                                               image=self._images[difficulty],
                                               text=difficulty)
            self._buttons[difficulty].bind('<Button-1>', self.__click_difficulty_button)
            self._buttons[difficulty].place(x=130 + i_ * 140, y=300, width=120, height=120)

    def __create_entry_tips(self):
        self._labels['drummer_name'] = Label(self, text='drummer\'s name:')
        self._labels['drummer_name'].place(x=40, y=10, height=80)
        self._labels['drummer_name'].config(font=("Times", 30))

        vcmd = (self.register(validate_alpha_digit), '%P', '%S')
        self._entries['drummer_name'] = Entry(self, bg='lightgray', validate='key', validatecommand=vcmd)
        self._entries['drummer_name'].place(x=40, y=80, height=80, width=300)
        self._entries['drummer_name'].config(font=("Times", 30))

        self._labels['song_id'] = Label(self, text='song ID:')
        self._labels['song_id'].place(x=450, y=80)
        self._labels['song_id'].config(font=("Times", 25))

        vcmd = (self.register(validate_integer), '%P', '%S')
        self._entries['song_id'] = Entry(self, bg='lightblue', validate='key', validatecommand=vcmd)
        self._entries['song_id'].place(x=600, y=80, height=40, width=70)
        self._entries['song_id'].config(font=("Times", 20))

        self._labels['difficulty'] = Label(self, text='difficulty')
        self._labels['difficulty'].place(x=330, y=250)
        self._labels['difficulty'].config(font=("Times", 20))

    def __click_difficulty_button(self, e):
        self._selected_difficulty = e.widget['text']
        self.__refresh_difficulty_buttons()

    def __refresh_difficulty_buttons(self):
        for i_, difficulty in enumerate(self._controller.client.DIFFICULTIES):
            self._buttons[difficulty].configure(bd=5, bg='snow')
        self._buttons[self._selected_difficulty].configure(bd=5, bg='red')

    def __click_start_button(self, e):
        self._controller.client.set_drummer_name(self._entries['drummer_name'].get())
        self._controller.client.set_song_id(self._entries['song_id'].get())
        self._controller.goto_next_screen(self.__class__)


class _RunScreen(Frame):
    LABEL = ['L', 'R']

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}
        self._labels = {}
        self._images = {}
        self.__init_screen()
        self.__capture_sensor()
        self.__capture_screenshot()
        self.__update_raw_canvas()
        # self._draw_thread = threading.Thread(target=self.__update_raw_canvas)
        # self._draw_thread.start()

    def __init_screen(self):
        self.__create_stop_button()
        self.__create_raw_canvas()

    def __create_stop_button(self):
        self._buttons['stop'] = Button(self, text='stop')
        self._buttons['stop'].bind('<Button-1>', self.__click_stop_button)
        self._buttons['stop'].place(x=280, y=520, width=250, height=70)

    def __create_raw_canvas(self):
        f = Figure()
        self._ax = f.subplots(nrows=6, ncols=2, sharex='all')
        self._canvas = FigureCanvasTkAgg(f, self)
        self._canvas.get_tk_widget().place(x=0, y=0, width=800, height=500)

    def __update_raw_canvas(self):
        self.__draw_raw_canvas(0)
        self.__draw_raw_canvas(1)
        self._canvas.draw()
        self.after(50, self.__update_raw_canvas)

    def __draw_raw_canvas(self, handedness):
        label = _RunScreen.LABEL[handedness]
        df = self._controller.client.query_sensor(label)
        if df is None:
            return

        for i_, col in enumerate(df.columns[1:]):
            self._ax[i_, handedness].clear()
            if i_ < 3:
                self._ax[i_, handedness].set_ylim(-20, 20)
            else:
                self._ax[i_, handedness].set_ylim(-200, 200)

            self._ax[i_, handedness].plot(df['timestamp'], df[col])
            self._ax[i_, handedness].set_ylabel(col)

        handedness_label = 'Left' if handedness == 0 else 'Right'
        self._ax[0, handedness].set_title(handedness_label + ' raw')
        self._ax[-1, handedness].set_xlabel('timestamp')

    def __capture_screenshot(self):
        self._controller.client.record_screenshot()

    def __capture_sensor(self):
        self._controller.client.record_sensor()

    def __click_stop_button(self, e):
        self._controller.client.stop_sensor()
        self._controller.client.stop_screenshot()
        self._controller.client.download_sensor()
        self._controller.client.update_local_record_table()
        self._controller.goto_next_screen(self.__class__)


class _LoadingScreen(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}
        self._labels = {}
        self._images = {}

        self.__init_screen()
        self._process_thread = threading.Thread(target=self.__process)
        self._process_thread.start()

    def __init_screen(self):
        self.__create_progress_bar()
        self.__create_tips()

    def __create_progress_bar(self):
        self._prog_bar = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self._prog_bar['maximum'] = self._controller.client.progress['maximum']
        self._prog_bar.place(x=100, y=300, width=600, height=50)

    def __create_tips(self):
        self._labels['tips'] = Label(self, text=self._controller.client.progress_tips)
        self._labels['tips'].place(x=100, y=350, width=600, height=80)
        self._labels['tips'].config(font=("Times", 12))

    def __process(self):
        self.after(200, self._process_queue)
        self._controller.client.upload_screenshot()

    def _process_queue(self):
        self._prog_bar['value'] = self._controller.client.progress['value']
        self._labels['tips'].configure(text=self._controller.client.progress_tips)
        if self._prog_bar['value'] < self._prog_bar['maximum']:
            self.after(200, self._process_queue)
        else:
            self._process_thread.join()
            self._controller.goto_next_screen(self.__class__)


class _ResultScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}
        self._labels = {}
        self._images = {}
        self.__init_screen()

    def __init_screen(self):
        self.__create_back_button()
        self.__create_score_canvas()
        self.__create_radar_canvas()
        self.__create_label_tips()

    def __create_back_button(self):
        self._buttons['back'] = Button(self, text='back')
        self._buttons['back'].bind('<Button-1>', self.__click_back_button)
        self._buttons['back'].place(x=520, y=520, width=250, height=70)

    def __create_score_canvas(self):
        img = Image.open(self._controller.client.pic_path['score_curve'])
        img = img.resize((800, 300), Image.ANTIALIAS)
        self._images['score_curve'] = ImageTk.PhotoImage(img)
        self._labels['score_curve'] = Label(self, image=self._images['score_curve'])
        self._labels['score_curve'].place(x=0, y=0, width=800, height=300)

    def __create_radar_canvas(self):
        img = Image.open(self._controller.client.pic_path['radar'])
        img = img.resize((250, 250), Image.ANTIALIAS)
        self._images['radar'] = ImageTk.PhotoImage(img)
        self._labels['radar'] = Label(self, image=self._images['radar'])
        self._labels['radar'].place(x=25, y=325, width=250, height=250)

    def __create_label_tips(self):
        times = self._controller.client.remained_play_times
        self._labels['remained_times'] = Label(self, text='Need to play %d times more' % times)
        self._labels['remained_times'].config(font=("Times", 20))
        self._labels['remained_times'].place(x=300, y=400, width=500, height=50)

    def __click_back_button(self, e):
        self._controller.goto_next_screen(self.__class__)


class _ErrorScreen(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self._controller = controller

        self._buttons = {}
        self._labels = {}
        self._images = {}
        self.__init_screen()

    def __init_screen(self):
        self.__create_back_button()
        self.__create_error_canvas()

    def __create_error_canvas(self):
        img = Image.open(self._controller.client.pic_path['error'])
        img = img.resize((800, 500), Image.ANTIALIAS)
        self._images['error'] = ImageTk.PhotoImage(img)
        self._labels['error'] = Label(self, image=self._images['error'])
        self._labels['error'].place(x=0, y=0, width=800, height=500)

    def __create_back_button(self):
        self._buttons['back'] = Button(self, text='back')
        self._buttons['back'].bind('<Button-1>', self.__click_back_button)
        self._buttons['back'].place(x=520, y=520, width=250, height=70)

    def __click_back_button(self, e):
        self._controller.goto_next_screen(self.__class__)
