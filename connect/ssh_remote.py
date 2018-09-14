#   1. Install the paramikio module for python.
#       pip install paramiko
#   2. Edit the SSH details below.

import paramiko
import sys

## EDIT SSH DETAILS ##

SSH_ADDRESS = "10.0.1.44"
SSH_USERNAME = "debian"
SSH_PASSWORD = "beaglebone"
SSH_COMMAND = "cd Projects/beagle; python read10axis.py -6;"

## CODE BELOW ##

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh_stdin = ssh_stdout = ssh_stderr = None

try:
    ssh.connect(SSH_ADDRESS, username=SSH_USERNAME, password=SSH_PASSWORD)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(SSH_COMMAND)
except Exception as e:
    sys.stderr.write("SSH connection error: {0}".format(e))

if ssh_stdout:
    print(ssh_stdout.read())
    sys.stdout.write(ssh_stdout.read())
if ssh_stderr:
    sys.stderr.write(ssh_stderr.read())