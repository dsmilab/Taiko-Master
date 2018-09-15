import mss.tools
import time


with mss.mss() as sct:
    count = 0
    last_time = time.time()
    while True:
        monitor = {'top': 40, 'left': 0, 'width': 640, 'height': 360}
        img = sct.grab(monitor)
        now_time = time.time()
        mss.tools.to_png(img.rgb, img.size, output='dir/%04d_%.2f.png' % (count, now_time))
        count += 1
        if now_time > last_time + 1:
            break
    print('FPS = %d' % count)
