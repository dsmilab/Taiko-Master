from config import *
from tools.timestamp import *

import os
import sys
import time
import threading
import mss.tools

__all__ = ['record_screenshot']


def record_screenshot(bg_exe=True):
    def __record_screenshot():
        with mss.mss() as sct:
            ts = time.time()
            st = 'capture_' + get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')

            local_dir = os.path.join(LOCAL_SCREENSHOT_PATH, st)
            if not os.path.isdir(local_dir):
                os.mkdir(local_dir)

            sys.stdout.write('[%s] Start capturing screenshot...\n' % st)
            sys.stdout.flush()
            try:
                count = 0
                while True:
                    monitor = {'top': 40, 'left': 0, 'width': 640, 'height': 360}
                    img = sct.grab(monitor)
                    now_time = time.time()
                    save_filename = '%04d-%.4f.png' % (count, now_time)
                    local_file = os.path.join(local_dir, save_filename)
                    mss.tools.to_png(img.rgb, img.size, output=local_file)
                    count += 1

            except KeyboardInterrupt:
                ts = time.time()
                st = get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')
                sys.stdout.write('[%s] Stop capturing.\n' % st)
                sys.stdout.flush()

    if bg_exe:
        p = threading.Thread(target=__record_screenshot, args=())
        p.start()
        p.join()
    else:
        __record_screenshot()

    return 0


if __name__ == '__main__':
    record_screenshot(bg_exe=False)
