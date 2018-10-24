import taiko.tools.converter
import taiko.visualize
import taiko.AAE_Main_test_Gui

from glob import glob
import posixpath
import sys
import os


def unzip(input_file_path, output_dir):
    taiko.tools.converter.convert_video_to_images(input_file_path, output_dir)


def main(argv):
    if argv[0] == '-d':
        input_file_path = argv[1]
        output_dir = argv[2]
        song_id = int(argv[3])
        unzip(input_file_path, output_dir)
        taiko.visualize.plot_play_score(output_dir, song_id, True, True)

        types = ['.png', '.flv', '.csv']
        for type_ in types:
            file_paths = glob(posixpath.join(output_dir, '*' + type_))
            for file_path in file_paths:
                os.remove(file_path)
    if argv[0] == '-e':
        taiko.AAE_Main_test_Gui.process_aae(1)


if __name__ == '__main__':
    main(sys.argv[1:])
