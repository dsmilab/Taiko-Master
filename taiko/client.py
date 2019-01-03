from .config import *
from .visualize import *
from .tools.timestamp import *
from .tools.converter import *
from .tools.realtime import *
from .play import *
from .AAE_prediction import *

from glob import glob
import pandas as pd
import paramiko
import threading
import sys
import os
import re
import mss.tools
import socket
import time
import pickle
import logging

__all__ = ['TaikoClient']

# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(levelname)s] (%(threadName)-10s) %(funcName)10s() %(message)s',
#                     )

# linux
LINUX_BB_COMMAND = "cd %s; python read10axis.py -6;" % REMOTE_BASE_PATH

GPU_LOGIN_COMMAND = "PATH='/usr/bin/anaconda3/bin:" \
                    "/usr/local/sbin:" \
                    "/usr/local/bin:" \
                    "/usr/sbin:" \
                    "/usr/bin:" \
                    "/sbin:" \
                    "/bin';" \
                    "export PATH;"

LINUX_SERVER_EXE_COMMAND = "python %s -d" % SERVER_EXE_PATH

# linux
LINUX_KILL_COMMAND = "pkill -f python;"


class _SSHTaiko(object):
    WINDOW_SIZE = 100

    def __init__(self, ssh, socket_, ip_addr, label):
        self._ssh = ssh
        self._socket = socket_
        self._ip_addr = ip_addr
        self._label = label
        self._analog = AnalogData(_SSHTaiko.WINDOW_SIZE)
        self._connection = None

    def start(self):
        logging.debug('_SSHTaiko start() => %s' % threading.current_thread())
        logging.info('wait for connect')
        while True:
            self._connection, address = self._socket.accept()

            try:
                sys.stdout.write('%s:%s connected!\n' % (address[0], address[1]))
                sys.stdout.flush()
                self.run()

            except Exception as e:
                sys.stderr.write(str(e) + '\n')
                sys.stderr.flush()
                break

    def run(self):
        logging.debug('_SSHTaiko run() => %s' % threading.current_thread())

        while self._connection:
            try:
                buf = self._connection.recv(1024)

                try:
                    data = pickle.loads(buf)
                except Exception as e:
                    continue

                if data[-1] == 'Q':
                    logging.info("client request to quit")
                    break

                elif data[-1] == self._label + '\r':
                    self._analog.add(data[:-2])

            except KeyboardInterrupt:
                break

        self._connection.close()

    def close(self):
        logging.debug('_SSHTaiko close() => %s' % threading.current_thread())
        self._ssh.close()
        self._connection.close()
        self._socket.close()

    def get_window_df(self):
        data = {
            'timestamp': [tm for tm in range(_SSHTaiko.WINDOW_SIZE)],
            'acc_x': self._analog.window[1],
            'acc_y': self._analog.window[2],
            'acc_z': self._analog.window[3],
            'gyr_x': self._analog.window[4],
            'gyr_y': self._analog.window[5],
            'gyr_z': self._analog.window[6],
        }

        df = pd.DataFrame(data=data)
        return df

    @property
    def ssh(self):
        return self._ssh

    @property
    def socket(self):
        return self._socket

    @property
    def ip_addr(self):
        return self._ip_addr


class _Client(object):
    DIFFICULTIES = ['easy', 'normal', 'hard', 'extreme']

    def __init__(self):
        self._local_sensor_filename = {}
        self._local_capture_dirname = None
        self._capture_alive = False
        self._remained_play_times = None
        self._taiko_ssh = {}
        self._ip_addr = '127.0.0.1'
        self._port = {}
        self.__load_socket_config()

        self._progress = {
            'maximum': 100,
            'value': 0,
        }
        self._progress_tips = ''

        self._prog_max = {
            # 'compress_screenshot': 30,
            # 'upload_to_server': 10,
            # 'server_process': 50,
            # 'download_from_server': 10,
            'process_screenshot': 50,
            'process_radar': 50,
        }

        self._pic_path = {
            'error': posixpath.join(PIC_DIR_PATH, 'curve_not_found.jpg'),
            # 'radar': posixpath.join(PIC_DIR_PATH, 'radar.png'),
        }
        for difficulty in _Client.DIFFICULTIES:
            self._pic_path[difficulty] = posixpath.join(PIC_DIR_PATH, difficulty + '.png')

    def __load_socket_config(self):
        config = glob(posixpath.join(SSH_CONFIG_PATH, 'config.sock'))[0]
        with open(config, 'r') as f:
            self._ip_addr = f.readline()[:-1]
            self._port['L'] = int(f.readline()[:-1])
            self._port['R'] = int(f.readline()[:-1])
            f.close()

    def _record_sensor(self, host_ip_, username_, pwd_, command_, label_):
        print('_Client _record_sensor() =>', threading.current_thread())
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self._ip_addr, self._port[label_]))
            sock.listen(2)

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            ssh.exec_command(command_)

            sys.stdout.write('connect %s ok\n' % host_ip_)
            sys.stdout.flush()

            taiko_ssh = _SSHTaiko(ssh, sock, host_ip_, label_)
            self._taiko_ssh[label_] = taiko_ssh

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

    def _stop_sensor(self, host_ip_, username_, pwd_, command_, label_):
        logging.debug('_Client _stop_sensor() => %s' % threading.current_thread())
        try:
            taiko_ssh = self._taiko_ssh.pop(label_, None)
            if taiko_ssh is not None:
                taiko_ssh.close()
                sys.stdout.write('stop %s ok\n' % taiko_ssh.ip_addr)
                sys.stdout.flush()

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            ssh.exec_command(command_)
            ssh.close()

            sys.stdout.write('kill %s ok\n' % host_ip_)
            sys.stdout.flush()

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

    def _download_sensor(self, host_ip_, username_, pwd_, prefix_):
        logging.debug('_Client _download_sensor() => %s' % threading.current_thread())
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            sftp = ssh.open_sftp()

            remote_items = sftp.listdir(REMOTE_BASE_PATH)
            csv_items = list(filter(lambda name: name[-4:] == '.csv', remote_items))
            remote_filename = max(csv_items)

            remote_file = posixpath.join(REMOTE_BASE_PATH, remote_filename)
            local_file = posixpath.join(LOCAL_SENSOR_DIR_PATH, prefix_ + '_' + remote_filename)

            sys.stdout.write('Reading from %s ...\n' % host_ip_)
            sys.stdout.flush()

            sftp.get(remote_file, local_file)
            self._local_sensor_filename[prefix_] = prefix_ + '_' + remote_filename

            sys.stdout.write('Reading %s done.\n' % host_ip_)
            sys.stdout.flush()

            sftp.close()
            ssh.close()

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

    def _record_screenshot(self):
        logging.debug('_Client _record_screenshot() => %s' % threading.current_thread())
        with mss.mss() as sct:
            ts = time.time()
            st = 'capture_' + get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')

            local_dir = posixpath.join(LOCAL_SCREENSHOT_PATH, st)
            os.makedirs(local_dir, exist_ok=True)

            self._local_capture_dirname = st
            sys.stdout.write('[%s] Start capturing screenshot...\n' % st)
            sys.stdout.flush()

            count = 0
            self._capture_alive = True
            while self._capture_alive:
                try:
                    monitor = {'top': 40, 'left': 0, 'width': 640, 'height': 360}
                    img = sct.grab(monitor)
                    now_time = time.time()
                    save_filename = '%04d-%.4f.png' % (count, now_time)
                    local_file = posixpath.join(local_dir, save_filename)
                    mss.tools.to_png(img.rgb, img.size, output=local_file)
                    count += 1

                except KeyboardInterrupt:
                    break

            ts = time.time()
            st = get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')
            sys.stdout.write('[%s] Stop capturing.\n' % st)
            sys.stdout.flush()

    @property
    def remained_play_times(self):
        return self._remained_play_times

    @property
    def pic_path(self):
        return self._pic_path

    @property
    def progress(self):
        return self._progress

    @property
    def progress_tips(self):
        return self._progress_tips

    @property
    def taiko_ssh(self):
        return self._taiko_ssh


class TaikoClient(_Client):

    def __init__(self):
        super(TaikoClient, self).__init__()
        self._capture_thread = None
        self._taiko_ssh_thread = []
        self._song_id = None
        self._drummer_name = None
        # !!!
        # local_curve_path = posixpath.join(PIC_DIR_PATH, "curve_not_found.jpg")
        # self._pic_path['score_curve'] = local_curve_path
        # local_result_path = posixpath.join(PIC_DIR_PATH, "result.jpg")
        # self._pic_path['result'] = local_result_path

    def clear(self):
        self.stop_sensor()
        self.stop_screenshot()
        self.clear_tmp_dir_png()
        self._progress = {
            'maximum': 100,
            'value': 0,
        }
        self._progress_tips = ''

    def record_sensor(self):
        logging.debug('TaikoClient record_sensor() => %s' % threading.current_thread())
        sensor_settings = glob(posixpath.join(SSH_CONFIG_PATH, '*.bb'))

        threads = []
        for file_path in sensor_settings:
            res = re.search('(\d){,3}.(\d){,3}.(\d){,3}.(\d){,3}.bb', file_path)
            filename = res.group(0)

            host_ip = filename[:-3]
            try:
                with open(file_path, 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    label = f.readline()[:-1]
                    command = LINUX_BB_COMMAND
                    thread = threading.Thread(target=self._record_sensor,
                                              args=(host_ip, username, pwd, command, label))
                    thread.start()
                    threads.append(thread)

                    f.close()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()

        for _, taiko_ssh in self._taiko_ssh.items():
            thread = threading.Thread(target=taiko_ssh.start)
            thread.start()
            self._taiko_ssh_thread.append(thread)

    def query_sensor(self, label):
        try:
            taiko_ssh = self._taiko_ssh[label]
            window_df = taiko_ssh.get_window_df()

            return window_df

        except KeyError:
            return None

    def stop_sensor(self):
        logging.debug('TaikoClient stop_sensor() => %s' % threading.current_thread())
        sensor_settings = glob(posixpath.join(SSH_CONFIG_PATH, '*.bb'))

        threads = []
        for file_path in sensor_settings:
            res = re.search('(\d){,3}.(\d){,3}.(\d){,3}.(\d){,3}.bb', file_path)
            filename = res.group(0)

            host_ip = filename[:-3]
            try:
                with open(file_path, 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    label = f.readline()[:-1]
                    command = LINUX_KILL_COMMAND
                    thread = threading.Thread(target=self._stop_sensor,
                                              args=(host_ip, username, pwd, command, label))
                    thread.start()
                    threads.append(thread)
                    f.close()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()
            logging.debug('TaikoClient join() stop_sensor() => %s' % thread)

    def download_sensor(self):
        logging.debug('TaikoClient download_sensor() => %s' % threading.current_thread())
        sensor_settings = glob(posixpath.join(SSH_CONFIG_PATH, '*.bb'))

        self._progress_tips = 'Downloading raw data from sensors ...'
        # threads = []
        for file_path in sensor_settings:
            res = re.search('(\d){,3}.(\d){,3}.(\d){,3}.(\d){,3}.bb', file_path)
            filename = res.group(0)

            host_ip = filename[:-3]
            try:
                threads = []
                with open(file_path, 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    _prefix = f.readline()[:-1]

                    self._download_sensor(host_ip, username, pwd, _prefix)
                    # thread = threading.Thread(target=self._download_sensor, args=(host_ip, username, pwd, _prefix,))
                    # thread.start()
                    # threads.append(thread)
                    # f.close()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()
        #
        # for thread in threads:
        #     logging.debug('TaikoClient join() download_sensor() => %s' % thread)
        #     thread.join()

    def record_screenshot(self):
        logging.debug('TaikoClient record_screenshot() => %s' % threading.current_thread())
        self._capture_thread = threading.Thread(target=self._record_screenshot)
        self._capture_thread.start()

    def stop_screenshot(self):
        logging.debug('TaikoClient stop_screenshot() => %s' % threading.current_thread())
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_alive = False
            self._capture_thread.join()
            logging.debug('TaikoClient join() stop_screenshot() =>', self._capture_thread)

    def process_screenshot(self):
        logging.debug('TaikoClient process_screenshot() => %s' % threading.current_thread())
        local_dir_path = posixpath.join(LOCAL_SCREENSHOT_PATH, self._local_capture_dirname)
        self._progress_tips = 'Processing screenshot for plotting ...'
        plot_play_score(local_dir_path, self._song_id, True, True)
        logging.debug('TaikoClient join() process_screenshot() => %s' % self._capture_thread)
        self._progress['value'] += self._prog_max['process_screenshot']

        local_curve_path = glob(posixpath.join(TMP_DIR_PATH, '*.png'))[0]
        pic_path = re.search('curve_(\d)+.png$', local_curve_path).group(0)
        self._remained_play_times = int(re.search('\d+', pic_path).group(0))
        self._pic_path['score_curve'] = local_curve_path

    def process_radar(self):
        logging.debug('TaikoClient process_radar() => %s' % threading.current_thread())

        row = {
            'drummer_name': self.drummer_name,
            'song_id': self.song_id,
            'left_sensor_datetime': self._local_sensor_filename['L'],
            'right_sensor_datetime': self._local_sensor_filename['R'],
            'capture_datetime': self._local_capture_dirname,
        }

        self._progress_tips = 'Finding play start time ...'
        play = get_play(row, from_tmp_dir=True)
        self._progress['value'] += self._prog_max['process_radar'] // 5 * 2
        self._progress_tips = 'Cropping raw data in need ...'
        play.crop_near_raw_data(0.1)
        self._progress['value'] += self._prog_max['process_radar'] // 5

        self._progress_tips = 'Processing sensor data ...'
        idx, sm_temp = execute()

        local_radar_path = glob(posixpath.join(TMP_DIR_PATH, 'result.jpg'))[0]
        self._pic_path['radar'] = local_radar_path

        self._progress['value'] += self._prog_max['process_radar'] // 5 * 2

    def clear_tmp_dir_png(self):
        local_curve_paths = glob(posixpath.join(TMP_DIR_PATH, '*.png'))
        for local_curve_path in local_curve_paths:
            os.remove(local_curve_path)

    def update_local_record_table(self):
        self._local_sensor_filename['L'] = 'L_2018-09-28_112912'
        self._local_sensor_filename['R'] = 'R_2018-09-28_112913'
        left_sensor_datetime = self._local_sensor_filename['L']
        right_sensor_datetime = self._local_sensor_filename['R']
        capture_datetime = self._local_capture_dirname

        try:
            record_df = pd.read_csv(LOCAL_RECORD_TABLE_PATH)
            index_ = 0
            if len(record_df) > 0:
                index_ = record_df.index[-1] + 1
            record_df.loc[index_] = [self.drummer_name,
                                     self.song_id,
                                     left_sensor_datetime,
                                     right_sensor_datetime,
                                     capture_datetime]
            record_df.to_csv(LOCAL_RECORD_TABLE_PATH, index=False)

            sys.stdout.write('Update local table ok\n')
            sys.stdout.flush()

        except Exception as e:
            sys.stderr.write("error: {0}\n".format(e))
            sys.stderr.flush()

    def set_song_id(self, song_id):
        self._song_id = int(song_id)

    def set_drummer_name(self, drummer_name):
        self._drummer_name = drummer_name

    @property
    def drummer_name(self):
        if self._drummer_name is None:
            return 'anonymous user'
        return self._drummer_name

    @property
    def song_id(self):
        if self._song_id is None:
            return -1
        return self._song_id
