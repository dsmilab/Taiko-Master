from config import *
from tools.converter import *
from glob import glob
import sys


def func1():
    replaced = 'capture_2018_09_27_16_39_45'
    screenshot_dir_path = glob(posixpath.join(LOCAL_SCREENSHOT_PATH, replaced))[0]
    output_dir = posixpath.join(LOCAL_SCREENSHOT_PATH, 'output_folder')
    convert_images_to_video(screenshot_dir_path, output_dir)


def func2():
    input_filename = posixpath.join(LOCAL_SCREENSHOT_PATH, 'output_folder', 'capture_2018_09_27_16_39_45.flv')
    timestamp_filename = posixpath.join(LOCAL_SCREENSHOT_PATH, 'output_folder', 'timestamp_capture_2018_09_27_16_39_45.csv')
    output_dir = posixpath.join(LOCAL_SCREENSHOT_PATH, 'pro_folder')
    convert_video_to_images(input_filename, output_dir)


def main(argv):
    if argv[0] == '-t':
        func1()
    else:
        func2()


if __name__ == '__main__':
    main(sys.argv[1:])
