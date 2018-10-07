from .config import *
from .tools.singleton import *
import subprocess
import platform

__all__ = ['Interface']


class Interface(metaclass=_Singleton):

    def __init__(self):
        self._pid = None

    def record_sensor(self, is_kill=False):
        if is_kill:
            if platform.system() == 'Windows':
                os.system('taskkill /F /PID ' + str(self._pid))
                print('kill pid:%d ok' % self._pid)

            elif platform.system() == 'Linux':
                os.system('kill -9 ' + str(self._pid))
                print('kill pid:%d ok' % self._pid)

        else:
            capture_exe_path = posixpath.join(BASE_PATH, 'capture_sensor.py')
            proc = subprocess.Popen(['python', capture_exe_path], stdout=subprocess.PIPE)
            self._pid = int(proc.pid)
            print('run pid = %d' % int(proc.pid))


