from ..config import *
import platform
import argparse
import paramiko
import threading
import subprocess
import sys
import os

__all__ = ['TaikoClient']
# SSH_BB_ADDRESS_1 = "10.0.1.39"
# SSH_BB_ADDRESS_2 = "10.0.1.44"

# linux
LINUX_BB_COMMAND = "cd Projects/beagle; python read10axis.py -6;"

# linux
LINUX_KILL_COMMAND = "pkill -f python;"


class TaikoClient(object):
    def __init__(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._video_pid = None
        print('client esb')

    def record_sensor(self, is_kill):
        def __record_sensor(host_ip_, username_, pwd_, command_, tips_=''):
            try:
                self._ssh.connect(host_ip_, username=username_, password=pwd_)
                self._ssh.exec_command(command_)

                sys.stdout.write('%s\n' % tips_)
                sys.stdout.flush()

            except Exception as ee:
                sys.stderr.write("SSH connection error: {0}\n".format(ee))
                sys.stderr.flush()

        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        sensor_settings = list(filter(lambda name: name[-3:] == '.bb', settings))

        for filename in sensor_settings:
            host_ip = filename[:-3]
            try:
                with open(os.path.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    command = LINUX_KILL_COMMAND if is_kill else LINUX_BB_COMMAND
                    tips = 'kill %s ok' % host_ip if is_kill else 'connect %s ok' % host_ip
                    p = threading.Thread(target=__record_sensor, args=(host_ip, username, pwd, command, tips))
                    p.start()
                    p.join()

            except Exception as e:
                sys.stderr.write("IO error: {0}\n".format(e))
                sys.stderr.flush()

    def record_video(self, is_kill):
        try:
            if is_kill:
                with open('~tmp.tmp', 'r') as f:
                    pid = int(f.readline())
                    self._video_pid = pid

                    if platform.system() == 'Windows':
                        os.system('taskkill /F /PID ' + str(pid))
                        print('kill pid:%d ok' % pid)

                    elif platform.system() == 'Linux':
                        os.system('kill -9 ' + str(pid))
                        print('kill pid:%d ok' % pid)

                    else:
                        sys.stderr.write('Unknown OS\n')
                        sys.stderr.flush()

                    f.close()
                os.remove('~tmp.tmp')

            else:
                proc = subprocess.Popen(['python', 'capture.py'], stdout=subprocess.PIPE)
                with open('~tmp.tmp', 'w') as f:
                    f.write(str(proc.pid))
                    f.close()

        except Exception as e:
            sys.stderr.write("IO error: {0}\n".format(e))
            sys.stderr.flush()
