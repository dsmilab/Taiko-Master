from ..config import *
import platform
import paramiko
import threading
import subprocess
import sys
import os

__all__ = ['TaikoClient']

# linux
LINUX_BB_COMMAND = "cd %s; python read10axis.py -6;" % REMOTE_BASE_PATH

# linux
LINUX_KILL_COMMAND = "pkill -f python;"


class TaikoClient(object):
    def __init__(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._video_pid = None

    def record_sensor(self, is_kill=False):
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
                sys.stderr.write("error: {0}\n".format(e))
                sys.stderr.flush()

    def record_screenshot(self, is_kill=False):
        try:
            if is_kill:
                with open('~tmp.tmp', 'r') as f:
                    if self._video_pid is None:
                        self._video_pid = int(f.readline())

                    pid = self._video_pid
                    self._video_pid = None

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
                    sys.stdout.write('[pid=%d] stop capturing screenshot locally.\n' % pid)
                    sys.stdout.flush()

                os.remove('~tmp.tmp')

            else:
                capture_exe_path = os.path.join(BASE_PATH, 'capture.py')
                proc = subprocess.Popen(['python', capture_exe_path], stdout=subprocess.PIPE)
                self._video_pid = proc.pid
                with open('~tmp.tmp', 'w') as f:
                    f.write(str(self._video_pid))
                    f.close()
                    sys.stdout.write('[pid=%d] start capturing screenshot locally...\n' % self._video_pid)
                    sys.stdout.flush()

        except Exception as e:
            sys.stderr.write("IO error: {0}\n".format(e))
            sys.stderr.flush()

    def download_sensor(self):
        def __download_sensor(host_ip_, username_, pwd_, prefix_):
            try:
                self._ssh.connect(host_ip_, username=username_, password=pwd_)
                sftp = self._ssh.open_sftp()

                remote_items = sftp.listdir(REMOTE_BASE_PATH)
                csv_items = list(filter(lambda name: name[-4:] == '.csv', remote_items))
                remote_filename = max(csv_items)

                sys.stdout.write('Reading from %s ...\n' % host_ip)
                sys.stdout.flush()

                remote_file = os.path.join(REMOTE_BASE_PATH, remote_filename)
                local_file = os.path.join(LOCAL_SENSOR_PATH, prefix_ + '_' + remote_filename)
                sftp.get(remote_file, local_file)

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
                    _prefix = f.readline()[:-1]

                    p = threading.Thread(target=__download_sensor, args=(host_ip, username, pwd, _prefix,))
                    p.start()
                    p.join()

            except Exception as e:
                sys.stderr.write("error: {0}\n".format(e))
                sys.stderr.flush()

    def upload_sensor(self):
        def __upload_sensor(host_ip_, username_, pwd_, tasks_):
            try:
                self._ssh.connect(host_ip_, username=username_, password=pwd_)
                sftp = self._ssh.open_sftp()

                sys.stdout.write('Uploading sensor data to %s ...\n' % host_ip)
                sys.stdout.flush()

                for local_file_, remote_file_ in tasks_:
                    sftp.put(local_file_, remote_file_)

                sys.stdout.write('Upload sensor data done.\n')
                sys.stdout.flush()

            except Exception as ee:
                sys.stderr.write("SSH connection error: {0}\n".format(ee))
                sys.stderr.flush()

        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        server_settings = list(filter(lambda name: name[-4:] == '.gpu', settings))

        for filename in server_settings:
            host_ip = filename[:-4]
            try:
                with open(os.path.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]

                    files = next(os.walk(LOCAL_SENSOR_PATH))[2]
                    left_items = list(filter(lambda name: name[:2] == 'L_', files))
                    right_items = list(filter(lambda name: name[:2] == 'R_', files))

                    left_filename = max(left_items)
                    right_filename = max(right_items)

                    tasks = []
                    local_file = os.path.join(LOCAL_SENSOR_PATH, left_filename)
                    remote_file = os.path.join(SERVER_LEFT_PATH, left_filename)
                    tasks.append((local_file, remote_file))

                    local_file = os.path.join(LOCAL_SENSOR_PATH, right_filename)
                    remote_file = os.path.join(SERVER_RIGHT_PATH, right_filename)
                    tasks.append((local_file, remote_file))

                    p = threading.Thread(target=__upload_sensor, args=(host_ip, username, pwd, tasks))
                    p.start()
                    p.join()

            except Exception as e:
                sys.stderr.write("error: {0}\n".format(e))
                sys.stderr.flush()

    def upload_screenshot(self):
        def __upload_screenshot(host_ip_, username_, pwd_, tasks_, remote_dir_):
            try:
                self._ssh.connect(host_ip_, username=username_, password=pwd_)
                sftp = self._ssh.open_sftp()

                try:
                    sftp.mkdir(remote_dir_)
                except IOError:
                    pass

                sys.stdout.write('Uploading screenshot to %s ...\n' % host_ip)
                sys.stdout.flush()

                for local_file_, remote_file_ in tasks_:
                    sftp.put(local_file_, remote_file_)

                sys.stdout.write('Upload screenshot done.\n')
                sys.stdout.flush()

            except Exception as ee:
                sys.stderr.write("SSH connection error: {0}\n".format(ee))
                sys.stderr.flush()

        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        server_settings = list(filter(lambda name: name[-4:] == '.gpu', settings))

        for filename in server_settings:
            host_ip = filename[:-4]
            try:
                with open(os.path.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]

                    screenshot_dirs = next(os.walk(LOCAL_SCREENSHOT_PATH))[1]
                    img_dir = max(screenshot_dirs)

                    local_dir = os.path.join(LOCAL_SCREENSHOT_PATH, img_dir)
                    remote_dir = os.path.join(SERVER_SCREENSHOT_PATH, img_dir)

                    files = next(os.walk(local_dir))[2]

                    tasks = []

                    for filename_ in files:
                        local_file = os.path.join(local_dir, filename_)
                        remote_file = os.path.join(remote_dir, filename_)
                        tasks.append((local_file, remote_file))

                    p = threading.Thread(target=__upload_screenshot, args=(host_ip, username, pwd, tasks, remote_dir))
                    p.start()
                    p.join()

            except Exception as e:
                sys.stderr.write("error: {0}\n".format(e))
                sys.stderr.flush()
