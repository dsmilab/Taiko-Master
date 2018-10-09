from .config import *
from .tools.timestamp import *
from .tools.converter import *

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
import queue

__all__ = ['TaikoClient']

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

    def __init__(self, ssh, ip_addr):
        self._ssh = ssh
        self._ip_addr = ip_addr

    def close(self):
        self._ssh.close()

    @property
    def ssh(self):
        return self._ssh

    @property
    def ip_addr(self):
        return self._ip_addr


class _Client(object):

    def __init__(self):
        self._local_sensor_filename = {}
        self._local_capture_dirname = None
        self._capture_alive = False
        self._remained_play_times = None
        self._taiko_ssh_queue = queue.Queue()

        self._progress = {
            'maximum': 100,
            'value': 0,
        }
        self._tips = ''

        self._pic_path = {
            'error': posixpath.join(PIC_DIR_PATH, 'curve_not_found.jpg'),
        }

    def __create_socket(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        config = glob(posixpath.join(SSH_CONFIG_PATH, 'config.sock'))[0]
        with open(config, 'r') as f:
            self._ip_addr = f.readline()[:-1]
            self._port = int(f.readline()[:-1])

        self._socket.bind((self._ip_addr, self._port))

    def _record_sensor(self, host_ip_, username_, pwd_, command_):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            ssh.exec_command(command_)

            sys.stdout.write('connect %s ok\n' % host_ip_)
            sys.stdout.flush()

            taiko_ssh = _SSHTaiko(ssh, host_ip_)
            self._taiko_ssh_queue.put(taiko_ssh)

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

    def _stop_sensor(self):
        try:
            while not self._taiko_ssh_queue.empty():
                taiko_ssh = self._taiko_ssh_queue.get()
                taiko_ssh.close()

                sys.stdout.write('stop %s ok\n' % taiko_ssh.ip_addr)
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
            self._progress['value'] += 10
            self._local_sensor_filename[prefix_] = prefix_ + '_' + remote_filename

            sys.stdout.write('Reading %s done.\n' % host_ip_)
            sys.stdout.flush()

            sftp.close()
            ssh.close()

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
            self._local_capture_dirname = st
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

    def _upload_screenshot(self, host_ip_, username_, pwd_, command_, upload_tasks_, remote_dir_):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host_ip_, username=username_, password=pwd_)
            sftp = ssh.open_sftp()

            # try to create tmp dir if it does not exit
            try:
                sftp.mkdir(remote_dir_)
            except IOError:
                pass

            # first delete remote files in tmp dir.
            remote_files = sftp.listdir(remote_dir_)
            for remote_file in remote_files:
                sftp.remove(posixpath.join(remote_dir_, remote_file))
            self._progress['value'] += 10

            sys.stdout.write('Uploading screenshot to %s ...\n' % host_ip_)
            sys.stdout.flush()

            for local_file_, remote_file_ in upload_tasks_:
                print(local_file_, remote_file_)
                sftp.put(local_file_, remote_file_)
            self._progress['value'] += 10

            sys.stdout.write('Upload screenshot done.\n')
            sys.stdout.flush()

            sys.stdout.write('Processing screenshot on GPU....\n')
            sys.stdout.flush()

            _, stdout_, _ = ssh.exec_command(GPU_LOGIN_COMMAND + command_, get_pty=True)
            for line in stdout_.readlines():
                sys.stdout.write(str(line))
                sys.stdout.flush()

            remote_files = sftp.listdir(SERVER_TMP_DIR_PATH)
            pic_filename = [filename for filename in remote_files if re.match('^curve_(\d)+.png$', filename)][0]

            remote_pic_file = posixpath.join(SERVER_TMP_DIR_PATH, pic_filename)
            local_pic_file = posixpath.join(PIC_DIR_PATH, pic_filename)

            self._remained_play_times = int(re.search('\d+', pic_filename).group(0))
            self._pic_path['score_curve'] = local_pic_file

            sftp.get(remote_pic_file, local_pic_file)
            sftp.remove(remote_pic_file)
            self._progress['value'] += 10

            sys.stdout.write('Download result from GPU, ok!\n')
            sys.stdout.flush()

            sftp.close()
            ssh.close()

        except Exception as e:
            sys.stderr.write('SSH connection error: %s\n' % str(e))
            sys.stderr.flush()

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
    def tips(self):
        return self._tips


class TaikoClient(_Client):

    def __init__(self):
        super(TaikoClient, self).__init__()
        self._capture_thread = None
        self._song_id = None

    def clear(self):
        self.stop_sensor()
        self.stop_screenshot()
        self._progress = {
            'maximum': 100,
            'value': 0,
        }
        self._tips = ''

    def record_sensor(self):
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
                    command = LINUX_BB_COMMAND
                    thread = threading.Thread(target=self._record_sensor,
                                              args=(host_ip, username, pwd, command))
                    thread.start()
                    threads.append(thread)

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()

    def stop_sensor(self):
        thread = threading.Thread(target=self._stop_sensor)
        thread.start()
        thread.join()

    def download_sensor(self):
        sensor_settings = glob(posixpath.join(SSH_CONFIG_PATH, '*.bb'))

        threads = []
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

                    thread = threading.Thread(target=self._download_sensor, args=(host_ip, username, pwd, _prefix,))
                    thread.start()
                    threads.append(thread)

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()

    def record_screenshot(self):
        self._capture_thread = threading.Thread(target=self._record_screenshot)
        self._capture_thread.start()
        self._capture_alive = True

    def stop_screenshot(self):
        self._capture_alive = False
        if self._capture_thread is not None:
            self._capture_thread.join()

    def upload_screenshot(self):
        server_settings = glob(posixpath.join(SSH_CONFIG_PATH, '*.gpu'))

        for file_path in server_settings:
            res = re.search('(\d){,3}.(\d){,3}.(\d){,3}.(\d){,3}.gpu', file_path)
            filename = res.group(0)

            host_ip = filename[:-4]
            try:
                with open(file_path, 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]

                    local_dir_path = posixpath.join(LOCAL_SCREENSHOT_PATH, self._local_capture_dirname)
                    convert_images_to_video(local_dir_path, local_dir_path)

                    remote_dir_path = SERVER_SCREENSHOT_PATH

                    flv_filename = self._local_capture_dirname + '.flv'
                    csv_filename = self._local_capture_dirname + '.csv'

                    files = [flv_filename, csv_filename]
                    upload_tasks = []
                    for filename_ in files:
                        local_file = posixpath.join(local_dir_path, filename_)
                        remote_file = posixpath.join(remote_dir_path, filename_)
                        upload_tasks.append((local_file, remote_file))

                    command = LINUX_SERVER_EXE_COMMAND + " %s %s %d" % (upload_tasks[0][1],
                                                                        remote_dir_path,
                                                                        self._song_id)

                    thread = threading.Thread(target=self._upload_screenshot,
                                              args=(host_ip, username, pwd, command, upload_tasks, remote_dir_path))
                    thread.start()
                    thread.join()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

    def set_song_id(self, song_id):
        self._song_id = int(song_id)

    @property
    def song_id(self):
        return self._song_id
