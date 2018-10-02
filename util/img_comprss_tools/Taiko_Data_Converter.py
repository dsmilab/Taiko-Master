
# coding: utf-8

# In[1]:


import numpy as np
import ffmpeg, cv2, os, glob
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


def get_img_filename(folder_path='./', filename_extension='png'):
    """ find all the filename below the folder, and sort them
    
    parameters: 
        folder_path: folder path where images saved
        filename_extension: file type want to search
        
    return:
        list of filename with alphabet order
    """
    
    filename_extension = '.' + filename_extension
    filename = glob.glob(folder_path + '/*' + filename_extension)
    filename = sorted(filename)
    return filename


# In[3]:


def get_timestamp(filename, output_filename='timestamp.csv'):
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
        timestamp.append( str(temp[0]) + '.' + str(temp[1]) )
    df = pd.DataFrame(timestamp, columns=['timestamp'])
    df.to_csv(output_filename, index=False)
    return df


# In[4]:


def images_to_video(input_folder='./', filename_extension='png', fr=20, output_filename='output', output_filename_extension='flv', verbose=True):
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
    if os.path.isfile(output_filename) is True:
        print('video already exist..')
        return
    if verbose:
        print('covert image from:', input_filename, 'into', output_filename)
    
    (
        ffmpeg
        .input(input_filename, pattern_type='glob', framerate=fr)
        .output(output_filename)
        .run()
    )
    if verbose:
        print('finish..')


# In[5]:


def convert_images_to_video(input_folder, output_folder):
    """ convert image which under the input_folder and output video & timestamp.csv on assign folder
    
    parameters: 
        input_folder: folder path where images saved, eg. taiko/capture_2018_09_25_12_21_56
        output_folder: folder to output video & timestamp.csv
    """        
    
    #  0.check if output_folder exist 
    if os.path.isdir(output_folder) is False:
        os.makedirs(output_folder)
        print('create folder:', output_folder)
        
    # 1.get all the path of images
    filename_list = get_img_filename(input_folder)
    
    # 2.get all the timestamp of image and save as .csv
    timestamp_filename = 'timestamp_' + input_folder.split('/')[1] + '.csv' # eg.timestamp_capture_2018_09_25_12_21_56.csv
    timestamp = get_timestamp(filename_list, output_filename= output_folder  + '/' + timestamp_filename)    
    
    # 3.convert images into video
    images_to_video(input_folder, output_filename= output_folder + '/' + input_folder.split('/')[1], output_filename_extension='flv')


# In[16]:


def convert_video_to_images(input_filename, timestamp_filename=None,  images_output_folder='./', output_filename_extension='png', verbose=False):
    """ convert video array into images, and save to assign folder
    
    parameters: 
        input_filename: video filename with filename_extension, eg. capture_2018_09_25_12_21_56.flv
        timestamp: list of timestamp, if use timestamp to assign image filename, None means filename will be assignd as 0.png 1.png 2.png ... N.png
        output_folder: output folder, if assigned output_folder dont exist, image wont be saved
        output_filename_extension: filename_extension of output image
        verbose: if output message
    """        
    # 0.check if output_folder exist 
    if os.path.isdir(images_output_folder) is False:
        os.makedirs(images_output_folder)
        print('create folder:', images_output_folder)
        
    # 1.convert video into array
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
    
    # 2. output img file
    print('video covert to image array....')
    filename_extension = '.' + output_filename_extension
    images_output_folder = images_output_folder + '/'
    if timestamp_filename is not None:
        timestamp = pd.read_csv(timestamp_filename).values
    for idx, img in enumerate(img_array):
        if timestamp_filename is None:
            if verbose:
                print('output:', images_output_folder + str(idx) + filename_extension)
            cv2.imwrite(images_output_folder + '%04d'%idx + filename_extension, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            if verbose:
                print('output:', images_output_folder + str(timestamp[idx][0]) + filename_extension)   
            cv2.imwrite(images_output_folder + '%04d-'%idx + str(timestamp[idx][0])+ filename_extension, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print('finish')


# # 教學
# 
# 就兩個指令:
# 
# - convert_images_to_video(input_folder, output_folder): 
# 
#   將圖片轉成影像(影像檔名為`[input_folder].flv`)，並輸出timestamp.csv(檔名為`[timestamp] + [input_folder].csv`)
#  - input_folder: 圖片們所在的資料夾位置
#  - output_folder: 輸出影片要存放的位置(如果給定的位置不存在會自動創建一個)
# - convert_video_to_images(): 將影像轉成圖片
#  - input_filename: 影片所在的位置
#  - timestamp_filename: timestamp.csv所在的位置，圖片檔名會參照timestamp.csv的order自動命名。如果參數設None會按照[0,N]順序編碼
#  - images_output_folder: 轉換後的圖片存放位置，如果指定的位置不存在會自動創建一個
#  - verbose: 要不要輸出詳細訊息

# ### 1. 將圖片轉成影像

# In[7]:


# 指定資料夾位置
input_folder = 'taiko/capture_2018_09_25_12_21_56'
output_folder = 'taiko_output'

convert_images_to_video(input_folder, output_folder)


# ### 2. 將影像轉回圖片

# In[15]:


# 指定資料夾位置
input_filename = 'taiko_output/capture_2018_09_25_12_21_56.flv'
timestamp_filename = 'taiko_output/timestamp_capture_2018_09_25_12_21_56.csv'
images_output_folder = 'taiko_output/recover_capture_2018_09_25_12_21_56'

convert_video_to_images(input_filename = input_filename, 
                                    timestamp_filename = timestamp_filename, 
                                    images_output_folder = images_output_folder, output_filename_extension='png', verbose=False)

