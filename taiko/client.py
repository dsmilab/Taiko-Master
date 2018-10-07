from .config import *
from .tools.timestamp import *
from .tools.singleton import *

import platform
import paramiko
import threading
import subprocess
import sys
import os
from glob import glob
import re
import mss.tools
import socket
import time
import pickle

__all__ = ['TaikoClient']

# linux
LINUX_BB_COMMAND = "cd %s; python read10axis.py -6;" % REMOTE_BASE_PATH

# linux
LINUX_KILL_COMMAND = "pkill -f python;"


class _Client(object):

    def __init__(self):
        self._local_filename = {}
        self._capture_alive = False

    def __create_socket(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        config = glob(posixpath.join(SSH_CONFIG_PATH, 'config.sock'))[0]
        with open(config, 'r') as f:
            self._ip_addr = f.readline()[:-1]
            self._port = int(f.readline()[:-1])

        self._socket.bind((self._ip_addr, self._port))

    @staticmethod
    def _record_sensor(host_ip_, username_, pwd_, command_, tips_=''):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            ssh.exec_command(command_)

            sys.stdout.write('%s\n' % tips_)
            sys.stdout.flush()

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

    def _download_sensor(self, host_ip_, username_, pwd_, prefix_):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            sftp = ssh.open_sftp()

            remote_items = sftp.listdir(REMOTE_BASE_PATH)
            csv_items = list(filter(lambda name: name[-4:] == '.csv', remote_items))
            remote_filename = max(csv_items)

            remote_file = posixpath.join(REMOTE_BASE_PATH, remote_filename)
            local_file = posixpath.join(LOCAL_SENSOR_PATH, prefix_ + '_' + remote_filename)

            sys.stdout.write('Reading from %s ...\n' % host_ip_)
            sys.stdout.flush()

            sftp.get(remote_file, local_file)
            self._local_filename[prefix_] = prefix_ + '_' + remote_filename

            sys.stdout.write('Reading %s done.\n' % host_ip_)
            sys.stdout.flush()

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

    def _record_screenshot(self):
        with mss.mss() as sct:
            ts = time.time()
            st = 'capture_' + get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')

            local_dir = posixpath.join(LOCAL_SCREENSHOT_PATH, st)
            os.makedirs(local_dir, exist_ok=True)

            self._capture_alive = True
            sys.stdout.write('[%s] Start capturing screenshot...\n' % st)
            sys.stdout.flush()

            count = 0
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


class TaikoClient(_Client):

    def __init__(self):
        super(TaikoClient, self).__init__()
        self._capture_thread = None

    def clear(self):
        self.record_sensor(True)
        self.record_screenshot(True)

    def record_sensor(self, is_kill=False):
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
                    command = LINUX_KILL_COMMAND if is_kill else LINUX_BB_COMMAND
                    tips = 'kill %s ok' % host_ip if is_kill else 'connect %s ok' % host_ip
                    thread = threading.Thread(target=self._record_sensor, args=(host_ip, username, pwd, command, tips))
                    thread.start()
                    threads.append(thread)

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()

    def download_sensor(self):
        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        sensor_settings = list(filter(lambda name: name[-3:] == '.bb', settings))

        threads = []
        for filename in sensor_settings:
            host_ip = filename[:-3]
            try:
                threads = []
                with open(posixpath.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    _prefix = f.readline()[:-1]

                    thread = threading.Thread(target=self._download_sensor, args=(host_ip, username, pwd, _prefix,))
                    thread.start()
                    threads.append(thread)

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()

    def record_screenshot(self, is_kill=False):
        try:
            if is_kill:
                self._capture_alive = False
                if self._capture_thread is not None:
                    self._capture_thread.join()

                    screenshot_dirs = next(os.walk(LOCAL_SCREENSHOT_PATH))[1]
                    img_dir = max(screenshot_dirs)
                    self._local_filename['capture'] = img_dir

            else:
                self._capture_thread = threading.Thread(target=self._record_screenshot)
                self._capture_thread.start()

        except Exception as e:
            sys.stderr.write('error: %s\n' % str(e))
            sys.stderr.flush()
