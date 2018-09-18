import mss.tools
import time
import os
from datetime import datetime, timedelta


def get_datetime(utc0_time, delta=0):
    d = datetime.fromtimestamp(utc0_time) + timedelta(seconds=int(delta))
    utc8 = datetime.fromtimestamp(time.mktime(d.timetuple()))
    return utc8


def main():
    with mss.mss() as sct:
        ts = time.time()
        st = 'capture_' + get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir('bb_capture/' + st)

        print('[%s] Start capturing screenshot...' % st)
        try:
            count = 0
            while True:
                monitor = {'top': 40, 'left': 0, 'width': 640, 'height': 360}
                img = sct.grab(monitor)
                now_time = time.time()
                save_filename = '%04d-%.4f.png' % (count, now_time)
                mss.tools.to_png(img.rgb, img.size, output='bb_capture/' + st + '/' + save_filename)
                count += 1

        except KeyboardInterrupt:
            ts = time.time()
            st = get_datetime(ts).strftime('%Y_%m_%d_%H_%M_%S')
            print('[%s] Stop capturing.' % st)

    return 0


if __name__ == '__main__':
    main()
