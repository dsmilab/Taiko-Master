import platform
import argparse
import paramiko
import threading
import subprocess
import sys
import os

# connect.ssh
SSH_CONFIG_PATH = '../data/connect_host/'


SSH_BB_ADDRESS_1 = "10.0.1.39"
SSH_BB_ADDRESS_2 = "10.0.1.44"

CSV_PATH = "Projects/beagle"


def transfer_file(host_ip, prefix=''):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with open(SSH_CONFIG_PATH + host_ip + '.dsmi', 'r') as f:
        username = f.readline()[:-1]
        pwd = f.readline()[:-1]

        try:
            ssh.connect(host_ip, username=username, password=pwd)
            sftp = ssh.open_sftp()
            remote_items = sftp.listdir(CSV_PATH)
            csv_items = list(filter(lambda name: name[-4:] == '.csv', remote_items))
            remote_file = max(csv_items)
            local_file = 'sensor_data/' + prefix + '_' + remote_file
            sys.stdout.write('Reading from %s ...\n' % host_ip)
            sys.stdout.flush()
            sftp.get(os.path.join(CSV_PATH, remote_file), local_file)

        except Exception as e:
            sys.stderr.write("SSH connection error: {0}\n".format(e))
            sys.stderr.flush()


def main():
    if not os.path.isdir('sensor_data'):
        os.mkdir('sensor_data')

    hosts = [SSH_BB_ADDRESS_1, SSH_BB_ADDRESS_2]
    handedness = ['left', 'right']

    threads = []
    for host, hand in zip(hosts, handedness):
        p = threading.Thread(target=transfer_file, args=(host, hand,))
        p.start()
        threads.append(p)

    for p_ in threads:
        p_.join()


if __name__ == '__main__':
    main()
