import argparse
import paramiko
import threading
import sys

SSH_ADDRESS_1 = "10.0.1.39"
SSH_ADDRESS_2 = "10.0.1.44"
SSH_USERNAME = "debian"
SSH_RECORD_COMMAND = "cd Projects/beagle; python read10axis.py -6;"
SSH_KILL_COMMAND = "pkill -f python;"


def work(host_ip, command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with open('data/passwd.tk', 'r') as f:
        pwd = f.readline()[:-1]

        try:
            ssh.connect(host_ip, username=SSH_USERNAME, password=pwd)
            ssh.exec_command(command)
            print('connect %s ok' % host_ip)

        except Exception as e:
            sys.stderr.write("SSH connection error: {0}".format(e))
            sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(description='Collect data from BeagleBone Blue')
    parser.add_argument('-kill', help='stop record', action='store_true')

    args = vars(parser.parse_args())
    is_kill = args['kill']

    hosts = [SSH_ADDRESS_1, SSH_ADDRESS_2]
    threads = []
    for h in hosts:
        command = SSH_RECORD_COMMAND
        if is_kill:
            command = SSH_KILL_COMMAND

        t = threading.Thread(target=work, args=(h, command,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
