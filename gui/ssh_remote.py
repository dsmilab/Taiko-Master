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

# linux
LINUX_BB_COMMAND = "cd Projects/beagle; python read10axis.py -6;"

# linux
LINUX_KILL_COMMAND = "pkill -f python;"


def record_sensor(host_ip, is_kill):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with open(SSH_CONFIG_PATH + host_ip + '.dsmi', 'r') as f:
        username = f.readline()[:-1]
        pwd = f.readline()[:-1]
        try:
            ssh.connect(host_ip, username=username, password=pwd)
            if is_kill:
                ssh.exec_command(LINUX_KILL_COMMAND)
            else:
                ssh.exec_command(LINUX_BB_COMMAND)

            if is_kill:
                print('kill %s ok' % host_ip)
            else:
                print('connect %s ok' % host_ip)

        except Exception as e:
            sys.stderr.write("SSH connection error: {0}\n".format(e))
            sys.stderr.flush()


def record_video(is_kill):
    if is_kill:
        with open('~tmp.tmp', 'r') as f:
            pid = int(f.readline())
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


def main():
    parser = argparse.ArgumentParser(description='Collect data from BeagleBone Blue')
    parser.add_argument('-kill', help='stop record', action='store_true')

    args_ = vars(parser.parse_args())
    is_kill = args_['kill']

    hosts = [SSH_BB_ADDRESS_1, SSH_BB_ADDRESS_2]

    threads = []
    for host in hosts:
        p = threading.Thread(target=record_sensor, args=(host, is_kill,))
        p.start()
        threads.append(p)

    p = threading.Thread(target=record_video, args=(is_kill,))
    p.start()
    threads.append(p)

    for p_ in threads:
        p_.join()


if __name__ == '__main__':
    main()
