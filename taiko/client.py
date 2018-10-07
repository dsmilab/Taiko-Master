from config import *

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
        self.__create_ssh()
        self.__create_socket()

    def __exit__(self, exc_type, exc_value, traceback):
        self._socket.close()

    def __create_ssh(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def __create_socket(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        config = glob(posixpath.join(SSH_CONFIG_PATH, 'config.sock'))[0]
        with open(config, 'r') as f:
            self._ip_addr = f.readline()[:-1]
            self._port = int(f.readline()[:-1])
        print('to bind')
        self._socket.bind((self._ip_addr, self._port))
        print('bind done')

    def _record_sensor(self, host_ip_, username_, pwd_, command_, tips_=''):
        try:
            # self._socket.listen(2)
            self._ssh.connect(host_ip_, username=username_, password=pwd_)
            self._ssh.exec_command(command_)

            sys.stdout.write('%s\n' % tips_)
            sys.stdout.flush()

            while True:
                connection, address = self._socket.accept()
                try:
                    threading.Thread(target=self.__record_sensor, args=(connection, self._ip_addr, self._port)).start()
                except Exception:
                    print("Thread did not start")
                    break

        except Exception as ee:
            sys.stderr.write("SSH connection error: {0}\n".format(ee))
            sys.stderr.flush()

    def __record_sensor(self, connection, ip, port):
        is_activate = True
        count = 0
        while is_activate:
            buf = connection.recv(1024)
            try:
                data = pickle.loads(buf)
            except Exception:
                print("Thread load buf fail")
                return

            if data[-1] == 'Q':
                print("client request to quit")
                is_activate = False
                connection.close()
            else:
                print("client sent:", data)


class TaikoClient(_Client):

    def __init__(self):
        super(TaikoClient, self).__init__()

    def record_sensor(self, is_kill=False):

        sensor_settings = glob(posixpath.join(SSH_CONFIG_PATH, '*.bb'))

        for file_path in sensor_settings[:1]:
            res = re.search('(\d){,3}.(\d){,3}.(\d){,3}.(\d){,3}.bb', file_path)
            filename = res.group(0)

            host_ip = filename[:-3]
            try:
                threads = []
                with open(file_path, 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    command = LINUX_KILL_COMMAND if is_kill else LINUX_BB_COMMAND
                    tips = 'kill %s ok' % host_ip if is_kill else 'connect %s ok' % host_ip
                    thread = threading.Thread(target=self._record_sensor, args=(host_ip, username, pwd, command, tips))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            except Exception as e:
                sys.stderr.write("error: {0}\n".format(e))
                sys.stderr.flush()
