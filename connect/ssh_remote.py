import argparse
import paramiko
import threading
import sys

SSH_BB_ADDRESS_1 = "10.0.1.39"
SSH_BB_ADDRESS_2 = "10.0.1.44"
SSH_VIDEO_ADDRESS = "140.113.25.42"

# linux
LINUX_BB_COMMAND = "cd Projects/beagle; python read10axis.py -6;"

# windows
WIN32_VIDEO_COMMAND = "cd Desktop/gui & python .\\capture.py"

# linux
LINUX_KILL_COMMAND = "pkill -f python;"
WIN32_KILL_COMMAND = "tskill python"


def work(host_ip, is_kill):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with open('../data/connect_host/' + host_ip + '.dsmi', 'r') as f:
        opcode = f.readline()[:-1]
        username = f.readline()[:-1]
        pwd = f.readline()[:-1]
        try:
            ssh.connect(host_ip, username=username, password=pwd)
            if opcode == 'sensor':
                if is_kill:
                    ssh.exec_command(LINUX_KILL_COMMAND)
                else:
                    ssh.exec_command(LINUX_BB_COMMAND)
            elif opcode == 'video':
                if is_kill:
                    ssh.exec_command(WIN32_KILL_COMMAND)
                else:
                    ssh.exec_command(WIN32_VIDEO_COMMAND)

            print('connect %s ok' % host_ip)

        except Exception as e:
            sys.stderr.write("SSH connection error: {0}".format(e))
            sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(description='Collect data from BeagleBone Blue')
    parser.add_argument('-kill', help='stop record', action='store_true')

    args = vars(parser.parse_args())
    is_kill = args['kill']

    hosts = [SSH_BB_ADDRESS_1, SSH_BB_ADDRESS_2, SSH_VIDEO_ADDRESS]

    threads = []
    for host in hosts:
        t = threading.Thread(target=work, args=(host, is_kill,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
