from tkinter import *
from taiko.network.client import TaikoClient
from taiko.config import *
import platform

if platform.system() == 'Windows':
    ACTIVE = 'normal'
elif platform.system() == 'Linux':
    ACTIVE = 'active'


class GUI(Tk):

    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.title('Taiko Master v0.3.2')
        self.geometry('1024x768')
        self._client = TaikoClient()
        self._stage = 0

        self._buttons = {
            'start': Button(master, text="start"),
            'stop': Button(master, text="stop"),
            'upload': Button(master, text='upload'),
            'analyze': Button(master, text='analyze'),
            'update_db': Button(master, text='update_db'),
        }

        self._labels = {
            'drummer_name': Label(master, text='drummer\'s name:'),
            'song_id': Label(master, text='song ID:')
        }

        self._entry = {
            'drummer_name': Entry(master, bg='lightgray'),
            'song_id': Entry(master, bg='lightblue'),
        }

        self._frame = {
            'difficulty': LabelFrame(master),
            'gender': LabelFrame(master),
        }

        self._var = {
            'difficulty': StringVar(),
            'gender': StringVar(),
            # 'song_id': IntVar(),
        }
        self._var['difficulty'].set('easy')
        self._var['gender'].set('M')
        # self._var['song_id'].set(0)

        self.init_screen()

    def init_screen(self):
        self.__create_start_btn()
        self.__create_stop_btn()
        self.__create_upload_btn()
        self.__create_analyze_btn()
        self.__create_update_db_btn()
        self.__create_player_tbx()
        self.__create_song_id_tbx()
        self.__create_difficulty_menu()
        self.__create_gender_menu()
        self.refresh()

    def __create_start_btn(self):
        self._buttons['start'].bind('<Button-1>', self.click_start_button)
        self._buttons['start'].place(x=70, y=650, width=150, height=50)

    def __create_stop_btn(self):
        self._buttons['stop'].bind('<Button-1>', self.click_stop_button)
        self._buttons['stop'].place(x=250, y=650, width=150, height=50)

    def __create_upload_btn(self):
        self._buttons['upload'].bind('<Button-1>', self.click_upload_button)
        self._buttons['upload'].place(x=430, y=650, width=150, height=50)

    def __create_analyze_btn(self):
        self._buttons['analyze'].bind('<Button-1>', self.click_analyze_button)
        self._buttons['analyze'].place(x=610, y=650, width=150, height=50)
        self._buttons['analyze'].config(state='disabled')

    def __create_update_db_btn(self):
        self._buttons['update_db'].bind('<Button-1>', self.click_update_db_button)
        self._buttons['update_db'].place(x=790, y=650, width=150, height=50)
        self._buttons['analyze'].config(state='disabled')

    def __create_difficulty_menu(self):
        radio_btns = {
            'easy': Radiobutton(self._frame['difficulty'],
                                text='easy', value='easy', variable=self._var['difficulty']),
            'normal': Radiobutton(self._frame['difficulty'],
                                  text='normal', value='normal', variable=self._var['difficulty']),
        }
        self._frame['difficulty'].place(x=200, y=400, width=100, height=100)
        for i_, (key, widget) in enumerate(radio_btns.items()):
            widget.place(x=0, y=0 + i_ * 30, width=100, height=20)
            widget.pack(anchor=W)

    def __create_gender_menu(self):
        radio_btns = {
            'male': Radiobutton(self._frame['gender'], text='Male', value='M', variable=self._var['gender']),
            'female': Radiobutton(self._frame['gender'], text='Female', value='F', variable=self._var['gender']),
        }
        self._frame['gender'].place(x=400, y=400, width=100, height=100)
        for i_, (key, widget) in enumerate(radio_btns.items()):
            widget.place(x=0, y=0 + i_ * 30, width=100, height=20)
            widget.pack(anchor=W)

    def __create_player_tbx(self):
        self._labels['drummer_name'].place(x=150, y=100, height=80)
        self._labels['drummer_name'].config(font=("Helvetica", 40))
        self._entry['drummer_name'].place(x=400, y=200, height=80, width=300)
        self._entry['drummer_name'].config(font=("Helvetica", 30))

    def __create_song_id_tbx(self):
        self._labels['song_id'].place(x=600, y=420)
        self._labels['song_id'].config(font=("Helvetica", 16))
        self._entry['song_id'].place(x=700, y=400, height=80, width=50)
        self._entry['song_id'].config(font=("Helvetica", 20))

    def click_start_button(self, event):
        if self._buttons['start']['state'] == ACTIVE:
            self._client.record_sensor()
            self._client.record_screenshot()
            # self._buttons['start'].place_forget()
            self.__create_stop_btn()
            self._stage = 1
        self.refresh()

    def click_stop_button(self, event):
        if self._buttons['stop']['state'] == ACTIVE:
            self._client.record_sensor(is_kill=True)
            self._client.record_screenshot(is_kill=True)
            self._client.download_sensor()
            # self._buttons['stop'].place_forget()
            self.__create_start_btn()
            self._stage = 2
        self.refresh()

    def click_upload_button(self, event):
        if self._buttons['upload']['state'] == ACTIVE:
            self._client.upload_sensor()
            self._client.upload_screenshot()
            self._stage = 3
        self.refresh()

    def click_analyze_button(self, event):
        if self._buttons['analyze']['state'] == ACTIVE:
            sys.stderr.write(self._entry['drummer_name'].get())
            sys.stderr.write('\n')

            sys.stderr.write(self._var['difficulty'].get())
            sys.stderr.write('\n')

            sys.stderr.write(self._var['gender'].get())
            sys.stderr.write('\n')

            sys.stderr.write(self._entry['song_id'].get())
            sys.stderr.write('\n')
            sys.stderr.flush()
        self.refresh()

    def click_update_db_button(self, event):
        if self._buttons['update_db']['state'] == ACTIVE:
            player_name = self._entry['drummer_name'].get()
            gender = self._var['gender'].get()
            song_id = self._entry['song_id'].get()
            difficulty = self._var['difficulty'].get()
            self._client.update_database(player_name, gender, song_id, difficulty)
        self.refresh()

    def refresh(self):
        if self._entry['drummer_name'].get() == '':
            self._buttons['update_db'].config(state='disabled')
        else:
            if self._entry['song_id'].get().isdigit() and self._stage >= 3:
                self._buttons['update_db'].config(state=ACTIVE)


if __name__ == "__main__":
    window = GUI()
    window.mainloop()
