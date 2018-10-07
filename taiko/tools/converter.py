import numpy as np
import ffmpeg
import cv2
import os
import glob
import matplotlib.pyplot as plt
import posixpath
import pandas as pd
import re

__all__ = ['convert_images_to_video',
           'convert_video_to_images']


def convert_images_to_video(input_dir_path, output_dir_path, verbose=0):
    os.makedirs(output_dir_path, exist_ok=True)
    res = re.search('capture_(\d){4}_(\d){2}_(\d){2}_(\d){2}_(\d){2}_(\d){2}', input_dir_path)

    input_dir_name = res.group(0)
    input_files = posixpath.join(input_dir_path + '/*.png')

    output_video_name = posixpath.join(output_dir_path, input_dir_name)
    output_filename = output_video_name + '.flv'

    if verbose > 0:
        print('convert image from:', input_files, 'into', output_filename)

    (
        ffmpeg
        .input(input_files, pattern_type='glob', framerate=20)
        .output(output_filename)
        .run()
    )

    if verbose > 0:
        print('finish..')


def convert_video_to_images(input_filename, output_dir_path, verbose=0):
    os.makedirs(output_dir_path, exist_ok=True)

    out, _ = (
        ffmpeg
        .input(input_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    img_array = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, 360, 640, 3])
    )

    for id_, img in enumerate(img_array):
        filename = '%04d.png' % id_
        file_path = posixpath.join(output_dir_path, filename)
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if verbose > 0:
            print('output:', file_path)

    if verbose > 0:
        print('finish')


def __get_img_filename(folder_path='./', filename_extension='png'):
    """ find all the filename below the folder, and sort them

    parameters:
        folder_path: folder path where images saved
        filename_extension: file type want to search

    return:
        list of filename with alphabet order
    """
    filename = glob.glob(folder_path + '/*.' + filename_extension)
    filename = sorted(filename)
    return filename


def __get_timestamp(filename, output_filename='timestamp.csv'):
    """ get the image timestamp and save as .csv

    parameters:
        filename: list of filename which was sorted by timestamp(get from function `get_img_filename`)
        output_filename: filename of csv

    return:
        dataframe of timestamp
    """
    timestamp = []
    for path in filename:
        temp = path.split('-')[1].split('.')[:-1]
        timestamp.append(str(temp[0]) + '.' + str(temp[1]))
    df = pd.DataFrame(timestamp, columns=['timestamp'])
    df.to_csv(output_filename, index=False)
    return df


def __images_to_video(input_folder='./', filename_extension='png', fr=20, output_filename='output',
                      output_filename_extension='flv', verbose=True):
    """ convert image which under the input_folder(with assigned filename_extension) into video

    parameters:
        input_folder: folder path where images saved, eg. taiko/capture_2018_09_25_12_21_56
        filename_extension: file type want to search
        fr: frame rate, eg. 20 means runs 20 picture during 1 second(frequency)
        output_filename: file name of output video
        output_filename_extension: filename_extension of output video
        verbose: if output message
    """

    filename_extension = '.' + filename_extension
    input_filename = input_folder + '/*' + filename_extension
    output_filename = output_filename + '.' + output_filename_extension
    # if os.path.isfile(output_filename) is True:
    #     print('video already exist..')
    #     return
    if verbose:
        print('convert image from:', input_filename, 'into', output_filename)

    (
        ffmpeg
        .input(input_filename, pattern_type='glob', framerate=fr)
        .output(output_filename)
        .run()
    )
    if verbose:
        print('finish..')
