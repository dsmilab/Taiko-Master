from taiko.tools.config import *
import os
import csv
import sys
import getpass
import posixpath


def create_bb_host(handedness):
    title = 'LEFT' if handedness == 0 else 'RIGHT'
    sys.stdout.write('Input the information about %s BeagleBone Blue:\n' % title)
    sys.stdout.write('%16s: ' % '[ip_address]')
    sys.stdout.flush()
    ip_addr = input()
    sys.stdout.write('%16s: ' % '[username]')
    sys.stdout.flush()
    username = input()
    password = getpass.getpass('%16s: ' % '[password]')
    sys.stdout.flush()

    os.makedirs(CONNECT_HOST_DIR_PATH, exist_ok=True)
    filename = posixpath.join(CONNECT_HOST_DIR_PATH, ip_addr + '.bb')
    with open(filename, 'w') as f:
        f.write(username + '\n')
        f.write(password + '\n')
        label = 'L' if handedness == 0 else 'R'
        f.write(label + '\n')


def create_local_host():
    sys.stdout.write('Input the information about this machine:\n')
    sys.stdout.write('%16s: ' % '[ip_address]')
    sys.stdout.flush()
    ip_addr = input()
    sys.stdout.write('%16s: ' % '[port 1]')
    sys.stdout.flush()
    port1 = input()
    sys.stdout.write('%16s: ' % '[port 2]')
    sys.stdout.flush()
    port2 = input()

    os.makedirs(CONNECT_HOST_DIR_PATH, exist_ok=True)
    filename = posixpath.join(CONNECT_HOST_DIR_PATH, 'config.sock')
    with open(filename, 'w') as f:
        f.write(ip_addr + '\n')
        f.write(port1 + '\n')
        f.write(port2 + '\n')


def create_tmp_dir():
    os.makedirs(LOCAL_SCREENSHOT_PATH, exist_ok=True)
    os.makedirs(LOCAL_SENSOR_DIR_PATH, exist_ok=True)
    record_table_csv = posixpath.join(TMP_DIR_PATH, 'record_table.csv')
    if os.path.exists(record_table_csv):
        return

    with open(record_table_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['drummer_name',
                         'song_id',
                         'left_sensor_datetime',
                         'right_sensor_datetime',
                         'capture_datetime'])


def main():
    create_bb_host(0)
    create_bb_host(1)
    create_local_host()
    create_tmp_dir()
    sys.stdout.write('[Taiko-Master] setup done!\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
