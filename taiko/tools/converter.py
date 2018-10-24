import numpy as np
import ffmpeg
import cv2
import os
from glob import glob
import posixpath
import pandas as pd
import re

__all__ = ['convert_images_to_video',
           'convert_video_to_images']


def convert_images_to_video(input_dir_path, output_dir_path):
    res = re.search('capture_(\d){4}_(\d){2}_(\d){2}_(\d){2}_(\d){2}_(\d){2}', input_dir_path)

    input_dir_name = res.group(0)
    input_file_pattern = posixpath.join(input_dir_path, '*.png')

    output_video_name = posixpath.join(output_dir_path, input_dir_name)
    output_filename = output_video_name + '.flv'

    input_file_paths = sorted(glob(input_file_pattern))

    timestamps = []
    for file_path in input_file_paths:
        res = re.search('(\d){4}-(\d)+.(\d)+.png', file_path)
        filename = res.group(0)
        timestamps.append(filename[5: -4])

    df = pd.DataFrame(data={
        'timestamp': timestamps
    })
    csv_path = posixpath.join(output_dir_path, input_dir_name + '.csv')
    df.to_csv(csv_path, index=False)

    os.makedirs(output_dir_path, exist_ok=True)

    (
        ffmpeg
        .input(input_file_pattern, pattern_type='glob', framerate=20)
        .output(output_filename)
        .run()
    )


def convert_video_to_images(input_file_path, output_dir_path):
    csv_path = input_file_path[:-4] + '.csv'
    df = pd.read_csv(csv_path)
    timestamps = list(df['timestamp'])

    os.makedirs(output_dir_path, exist_ok=True)
    out, _ = (
        ffmpeg
        .input(input_file_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    img_array = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, 360, 640, 3])
    )
    for id_, img in enumerate(img_array):
        filename = '%04d-%s.png' % (id_, timestamps[id_])
        file_path = posixpath.join(output_dir_path, filename)
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
