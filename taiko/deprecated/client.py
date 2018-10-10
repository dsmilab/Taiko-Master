from taiko.config import *
import pandas as pd
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

# GPU
LOGIN_COMMAND = "PATH='/usr/bin/anaconda3/bin'; export PATH;" +\
                "cd %s; " % SERVER_PROJECT_PATH

UPDATE_DB_COMMAND = "python taiko/update_db.py"


class TaikoClient(object):
    def __init__(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._video_pid = None
        self._start_datetime = None
        self._local_filename = {}

    def reset(self):
        self._video_pid = None
        self._start_datetime = None
        self._local_filename = {}

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

        self.reset()
        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        sensor_settings = list(filter(lambda name: name[-3:] == '.bb', settings))

        for filename in sensor_settings:
            host_ip = filename[:-3]
            try:
                threads = []
                with open(posixpath.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    command = LINUX_KILL_COMMAND if is_kill else LINUX_BB_COMMAND
                    tips = 'kill %s ok' % host_ip if is_kill else 'connect %s ok' % host_ip
                    thread = threading.Thread(target=__record_sensor, args=(host_ip, username, pwd, command, tips))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

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

                    screenshot_dirs = next(os.walk(LOCAL_SCREENSHOT_PATH))[1]
                    img_dir = max(screenshot_dirs)
                    self._local_filename['capture'] = img_dir

                os.remove('~tmp.tmp')

            else:
                capture_exe_path = posixpath.join(BASE_PATH, 'capture.py')
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

                remote_file = posixpath.join(REMOTE_BASE_PATH, remote_filename)
                local_file = posixpath.join(LOCAL_SENSOR_DIR_PATH, prefix_ + '_' + remote_filename)

                sys.stdout.write('Reading from %s ...\n' % host_ip)
                sys.stdout.flush()
                sftp.get(remote_file, local_file)
                self._local_filename[prefix_] = prefix_ + '_' + remote_filename

                sys.stdout.write('Reading done.\n' % host_ip)
                sys.stdout.flush()

            except Exception as ee:
                sys.stderr.write("SSH connection error: {0}\n".format(ee))
                sys.stderr.flush()

        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        sensor_settings = list(filter(lambda name: name[-3:] == '.bb', settings))

        for filename in sensor_settings:
            host_ip = filename[:-3]
            try:
                threads = []
                with open(posixpath.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    _prefix = f.readline()[:-1]

                    thread = threading.Thread(target=__download_sensor, args=(host_ip, username, pwd, _prefix,))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
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
                threads = []
                with open(posixpath.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]

                    files = next(os.walk(LOCAL_SENSOR_DIR_PATH))[2]
                    left_items = list(filter(lambda name: name[:2] == 'L_', files))
                    right_items = list(filter(lambda name: name[:2] == 'R_', files))

                    left_filename = max(left_items)
                    right_filename = max(right_items)

                    tasks = []
                    local_file = posixpath.join(LOCAL_SENSOR_DIR_PATH, left_filename)
                    remote_file = posixpath.join(SERVER_LEFT_PATH, left_filename)
                    tasks.append((local_file, remote_file))

                    local_file = posixpath.join(LOCAL_SENSOR_DIR_PATH, right_filename)
                    remote_file = posixpath.join(SERVER_RIGHT_PATH, right_filename)
                    tasks.append((local_file, remote_file))

                    thread = threading.Thread(target=__upload_sensor, args=(host_ip, username, pwd, tasks))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
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
                threads = []
                with open(posixpath.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]

                    screenshot_dirs = next(os.walk(LOCAL_SCREENSHOT_PATH))[1]
                    img_dir = max(screenshot_dirs)
                    self._start_datetime = img_dir[-19:]

                    local_dir = posixpath.join(LOCAL_SCREENSHOT_PATH, img_dir)
                    remote_dir = posixpath.join(SERVER_SCREENSHOT_PATH, img_dir)

                    files = next(os.walk(local_dir))[2]

                    tasks = []

                    for filename_ in files:
                        local_file = posixpath.join(local_dir, filename_)
                        remote_file = posixpath.join(remote_dir, filename_)
                        tasks.append((local_file, remote_file))

                    thread = threading.Thread(target=__upload_screenshot,
                                              args=(host_ip, username, pwd, tasks, remote_dir))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

    def update_database(self, player_name, gender, song_id, difficulty):
        def __update_database(host_ip_, username_, pwd_, command_):
            try:
                self._ssh.connect(host_ip_, username=username_, password=pwd_)
                stdin, stdout, stderr = self._ssh.exec_command(command_)

                key_msg = str(stdout.read())[-4:-1]
                if key_msg == ENTRY_SUCCESS:
                    sys.stdout.write('Update database ok\n')
                else:
                    sys.stdout.write('update database fail\n')

                sys.stdout.flush()

            except Exception as ee:
                sys.stderr.write("SSH connection error: {0}\n".format(ee))
                sys.stderr.flush()

        settings = next(os.walk(SSH_CONFIG_PATH))[2]
        server_settings = list(filter(lambda name: name[-4:] == '.gpu', settings))

        for filename in server_settings:
            host_ip = filename[:-4]
            try:
                threads = []
                with open(posixpath.join(SSH_CONFIG_PATH, filename), 'r') as f:
                    username = f.readline()[:-1]
                    pwd = f.readline()[:-1]
                    command = LOGIN_COMMAND
                    command += UPDATE_DB_COMMAND
                    command += " %s %s %s %s %s" % (player_name, gender, str(song_id), difficulty, self._start_datetime)

                    thread = threading.Thread(target=__update_database, args=(host_ip, username, pwd, command,))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

    def update_local_record_table(self, player_name, song_id):
        left_sensor_datetime = self._local_filename['L']
        right_sensor_datetime = self._local_filename['R']
        capture_datetime = self._local_filename['capture']

        try:
            record_df = pd.read_csv(LOCAL_RECORD_TABLE_PATH)
            index_ = 0
            if len(record_df) > 0:
                index_ = record_df.index[-1] + 1
            record_df.loc[index_] = [player_name, song_id, left_sensor_datetime, right_sensor_datetime, capture_datetime]
            record_df.to_csv(LOCAL_RECORD_TABLE_PATH, index=False)

            sys.stdout.write('Update local table ok\n')
            sys.stdout.flush()

        except Exception as e:
                sys.stderr.write('error: %s\n' % str(e))
                sys.stderr.flush()

