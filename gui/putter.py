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

REMOTE_SAVE_PATH = "PyCharmPojects/Taiko-Master/tmp_capture"
BB_CAPUTRE_PATH = 'bb_capture/'

LOGIN_COMMAND = "PATH='/usr/bin/anaconda3/bin'; export PATH;" +\
                "cd PyCharmPojects/Taiko-Master/; python taiko/drum.py;"


def _get_img_dir():
    dirs = next(os.walk(BB_CAPUTRE_PATH))[1]
    img_dir = max(dirs)
    return os.path.join(BB_CAPUTRE_PATH, img_dir)


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
            files = next(os.walk(img_dir))[2]
            for filename in files:
                local_file = os.path.join(img_dir, filename)
                remote_file = os.path.join(REMOTE_SAVE_PATH, filename)
                sftp.put(local_file, remote_file)
            stdin, stdout, stderr = ssh.exec_command(LOGIN_COMMAND)
            song_start_time = str(stdout.read())[-16:-1]
            print(song_start_time)
        except Exception as e:
            sys.stderr.write("SSH connection error: {0}\n".format(e))
            sys.stderr.flush()


def main():
    put_file(SSH_GPU_ADDRESS)


if __name__ == '__main__':
    main()
