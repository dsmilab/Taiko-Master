from .config import *

import platform
import paramiko
import threading
import subprocess
import sys
import os
from glob import glob
import re
import socket
import pickle

__all__ = ['TaikoClient']

# linux
LINUX_BB_COMMAND = "cd %s; python read10axis.py -6;" % REMOTE_BASE_PATH

# linux
LINUX_KILL_COMMAND = "pkill -f python;"


class _Client(object):

    def __init__(self):
        self._local_filename = {}

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

        except Exception as ee:
            sys.stderr.write("SSH connection error: {0}\n".format(ee))
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

            sys.stdout.write('Reading done.\n' % host_ip_)
            sys.stdout.flush()

        except Exception as ee:
            sys.stderr.write("SSH connection error: {0}\n".format(ee))
            sys.stderr.flush()


class TaikoClient(_Client):

    def __init__(self):
        super(TaikoClient, self).__init__()

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
                sys.stderr.write("error: {0}\n".format(e))
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
                    thread.daemon = True
                    thread.start()
                    threads.append(thread)

            except Exception as e:
                sys.stderr.write("error: {0}\n".format(e))
                sys.stderr.flush()

        for thread in threads:
            thread.join()