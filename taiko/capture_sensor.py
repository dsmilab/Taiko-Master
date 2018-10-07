from config import *
from tools.timestamp import *

import platform
import paramiko
import threading
import time
import csv
import subprocess
import sys
import os
from glob import glob
import re
import socket
import pickle


def __send_to_local(axis_10, connection):
    ts = time.time()
    st = get_datetime(ts).strftime('%Y-%m-%d_%H%M%S')
    start_st = st

    sys.stdout.write('[%s] Start reading sensor data to local...\n' % st)
    sys.stdout.flush()

    try:
        with open(st + '.csv', 'w') as f:
            writer = csv.writer(f)
            if axis_10:
                writer.writerow(['timestamp', 'temp',
                                 'imu_ax', 'imu_ay', 'imu_az',
                                 'imu_gx', 'imu_gy', 'imu_gz',
                                 'imu_mx', 'imu_my', 'imu_mz'])
            else:
                writer.writerow(['timestamp',
                                 'imu_ax', 'imu_ay', 'imu_az',
                                 'imu_gx', 'imu_gy', 'imu_gz'])

            data_rows = 0
            while True:
                try:
                    buffer_ = connection.recv(1024)
                    try:
                        row = pickle.loads(buffer_)

                        writer.writerow(row)
                        print(row)

                        data_rows += 1
                        if data_rows % 10000 == 0:
                            ts = time.time()
                            st = get_datetime(ts).strftime('%Y-%m-%d_%H%M%S')
                            sys.stdout.write('[%s] %6d rows have been collected.\n' % (st, data_rows))
                            sys.stdout.flush()

                    except Exception:
                        print("Thread load buf fail")

                except KeyboardInterrupt:
                    f.close()
                    connection.close()

                    ts = time.time()
                    st = get_datetime(ts).strftime('%Y-%m-%d_%H%M%S')
                    sys.stdout.write('[%s] \"%s\" was saved.\n' % (st, start_st + '.csv'))
                    sys.stdout.flush()
                    break

    except IOError:
        sys.stderr.write('Failed to build CSV.\n')
        sys.stderr.flush()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', 21))
    sock.listen(2)

    print('wait for connect...\n')

    while True:
        connection, address = sock.accept()
        ip, port = str(address[0]), str(address[1])
        print('connect by: ', address)
        try:
            threading.Thread(target=__send_to_local, args=(False, connection,)).start()
        except Exception:
            print("Thread did not start")
            break

    sock.close()


if __name__ == '__main__':
    main()
