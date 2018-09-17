import platform
import argparse
import paramiko
import threading
import subprocess
import sys
import os

# connect.ssh
SSH_CONFIG_PATH = '../data/connect_host/'


SSH_GPU_ADDRESS = "140.114.36.104"

REMOTE_SAVE_PATH = "PyCharmPojects/Taiko-Master/bb_capture_output"
BB_CAPTURE_PATH = 'bb_capture/'

LOGIN_COMMAND = "PATH='/usr/bin/anaconda3/bin'; export PATH;" +\
                "cd PyCharmPojects/Taiko-Master/; "

ENTRY_COMMAND = "python taiko/drum.py;"


def _get_img_dir():
    dirs = next(os.walk(BB_CAPTURE_PATH))[1]
    img_dir = max(dirs)
    return img_dir


def put_file(host_ip):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with open(SSH_CONFIG_PATH + host_ip + '.gpu', 'r') as f:
        username = f.readline()[:-1]
        pwd = f.readline()[:-1]

        try:
            ssh.connect(host_ip, username=username, password=pwd)
            sftp = ssh.open_sftp()

            img_dir = _get_img_dir()
            img_dir_path = os.path.join(BB_CAPTURE_PATH, img_dir)

            files = next(os.walk(img_dir_path))[2]
            remote_dir = os.path.join(REMOTE_SAVE_PATH, img_dir)
            print(remote_dir)

            try:
                sftp.mkdir(remote_dir)
            except IOError:
                pass

            for filename in files:
                local_file = os.path.join(img_dir_path, filename)
                remote_file = os.path.join(remote_dir, filename)
                print(filename)
                sftp.put(local_file, remote_file)

            command = LOGIN_COMMAND + "python taiko/drum.py %s %s %s %s" % ('howeverover', 'M', '1', img_dir[-19:])
            stdin, stdout, stderr = ssh.exec_command(command)
            song_start_time = str(stdout.read())[-16:-1]
            print(song_start_time)
            print(img_dir[-19:])
            print(str(stderr.read()))
        except Exception as e:
            sys.stderr.write("SSH connection error: {0}\n".format(e))
            sys.stderr.flush()


def main():
    put_file(SSH_GPU_ADDRESS)


if __name__ == '__main__':
    main()
