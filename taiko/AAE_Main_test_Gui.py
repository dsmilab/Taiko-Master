from .lstm_vae import *
from .DTW_song_dongKa_sampling import *
from .config import *

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import time
import posixpath

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras.constraints import non_neg
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import History
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, RepeatVector
import matplotlib.pyplot as plt

import glob as glob
import gc
history = History()

__all__ = ['process_aae']
# In[2]:


def process_aae(song_id):
    print('run main(%d)' % song_id)
    Samples_don, Samples_ka = inputData_train(song_id) # import 高手們該首歌的motifs
    tmp_sample_don, tmp_sample_ka = inputData_predict()  #import tmp data
    predict(Samples_don, tmp_sample_don)


# In[3]:


def inputData_train(song_id): #Dir為taiko_play_result.csv 路徑
    Dir_song_don = posixpath.join(MOTIF_DIR_PATH, 'don/song%d' % song_id)
    song_don = glob.glob(posixpath.join(Dir_song_don, '*.csv'))
    Dir_song_ka = posixpath.join(MOTIF_DIR_PATH, 'ka/song%d' % song_id)
    song_ka = glob.glob(posixpath.join(Dir_song_ka, '*.csv'))

    Don_samples = []
    Ka_samples = []

    for i in range(len(song_don)):
        df = pd.read_csv(song_don[i])
        array_don =  np.array(df)
        array_don = array_don.reshape(1,array_don.shape[0],array_don.shape[1])
        Don_samples.append(array_don)
    # for i in range(len(song_ka)):
    #     df = pd.read_csv(song_ka[i])
    #     array_ka = np.array(df)
    #     array_ka = array_ka.reshape(1,array_ka.shape[0],array_ka.shape[1])
    #     Ka_samples.append(array_ka)

    return Don_samples, Ka_samples #data type: list ,samples



# In[4]:


def inputData_predict(): #Dir為 tep 的motifs 的dir, eg. motifs/subject/song1/order1/dong
    Dir_don = posixpath.join(TMP_DIR_PATH, 'motif/don')
    Dir_ka = posixpath.join(TMP_DIR_PATH, 'motif/ka')

    tmp_sample_don = Main_Execure(Dir_don)
    tmp_sample_ka = Main_Execure(Dir_ka)

#     motifs_aray = np.array(df)
#     motifs_aray = motifs_aray.reshape(1,motifs_aray.shape[0],motifs_aray.shape[1])

    return tmp_sample_don, tmp_sample_ka  #data type: array (1,20,12)


# In[5]:


#AAE

def create_lstm_vae_train(input_dim,
timesteps,
batch_size,
intermediate_dim,
latent_dim,
epsilon_std=1.,
first_second=False

):

    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))
    if (first_second==False):
        # LSTM encoding
        h = LSTM(intermediate_dim)(x)

        # VAE Z layer
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)
    elif(first_second==True): #fix weights for encoder
        # LSTM encoding
        h = LSTM(intermediate_dim)(x)

        # VAE Z layer
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    if(first_second ==True):
        encoder.load_weights(posixpath.join(BASE_PATH, 'external/encoder.h5'))
        vae.load_weights(posixpath.join(BASE_PATH, 'external/vae.h5'))
    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator


# In[6]:


# enc predict & clustering
def predict(data_root, data_tmp):
    print(data_root)
    input_dim = data_tmp.shape[-1]
    timesteps = data_tmp.shape[1]
    vae, enc, gen = create_lstm_vae_train(input_dim,
            timesteps,
            batch_size=1,
            intermediate_dim=64,
            latent_dim=5,
            epsilon_std=1.,
            first_second =True
            )
    Latent_v = enc.predict(data_tmp)
    Latent_root = []
    for item in data_root:
        latentcode = enc.predict(item)
        Latent_root.append(latentcode)

    def cos_sim(vector_a, vector_b):
#     """
#     计算两个向量之间的余弦相似度
#     :param vector_a: 向量 a
#     :param vector_b: 向量 b
#     :return: sim
#     """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    Distance = []
    for vector in Latent_root:
        simi = cos_sim(Latent_v,vector)
        Distance.append(simi)

    minLocation = Distance.index(min(Distance))
    Latent_master = Latent_root[minLocation]

    print('in front of radar compare()')
    #計算Latent_v 與其他四個高手的每一個dong ka predict vector的相似度 選最小者
    Rader_compare(Latent_v,Latent_master)




# In[7]:


# training

def conservative_train_AAE(data1,data2=[]):

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    #traning
    x = data1
    y = data2



    if(y!=[]):
        consercative_or_not = True
    else:
        consercative_or_not = False

    if (consercative_or_not == True):
        xY = np.vstack((x,y))

        input_dim = xY.shape[-1]
        print (input_dim)
        timesteps = xY.shape[1]
        print (timesteps)
        batch_size = 1


        vae, enc, gen = create_lstm_vae_train(input_dim,
        timesteps,
        batch_size,
        intermediate_dim=64,
        latent_dim=5,
        epsilon_std=1.,
        first_second= True
                            )

        early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=2)
        vae.fit(xY, xY, epochs=10000,callbacks=[early_stopping])
        enc.save_weights(posixpath.join(BASE_PATH, 'external/encoder.h5'))
        vae.save_weights(posixpath.join(BASE_PATH, 'external/vae.h5'))
        preds = vae.predict(xY,batch_size=batch_size)

    else:
        input_dim = x.shape[-1] # 13
        print (input_dim)
        timesteps = x.shape[1] # 3
        print (timesteps)
        batch_size = 1


        vae, enc, gen = create_lstm_vae_train(input_dim,
        timesteps,
        batch_size,
        intermediate_dim=64,
        latent_dim=5,
        epsilon_std=1.)

        early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=2)
        vae.fit(x, x, epochs=10000,callbacks=[early_stopping])
        enc.save_weights(posixpath.join(BASE_PATH, 'external/encoder.h5'))
        vae.save_weights(posixpath.join(BASE_PATH, 'external/vae.h5'))
        preds = vae.predict(x,batch_size=batch_size)
    return vae, enc, gen

    gc.collect()


# In[8]:


#查詢data organize table (tmp)後一個一個執行 DTW_song_dongKa_sampling並且串成train set


# In[9]:


#train(母)串 train(子), then train
def training(train_set):
    vae, enc, gen  = conservative_train_AAE(data1 = train_set)
    print (' train data finish!')
    gc.collect()

    return vae, enc, gen


# In[10]:


#雷達圖
def Rader_single(sample,enc):
    import matplotlib.pyplot as plt
    Latent_code = enc.predict(sample)
    samplesrar = np.interp(Latent_code, (Latent_code.min(), Latent_code.max()), (50, 100)) #Rescale to 0-100
    # 中文和負號的正常顯示
    # plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    # plt.rcParams['axes.unicode_minus'] = False

    # 使用ggplot的繪圖風格
    # plt.style.use('ggplot')

    # 構造數據
    values = samplesrar[0]



    feature = ['feature01','feature02','feature03','feature04','feature05']

    N = len(values)
    # 設置雷達圖的角度，用於平分切開一個圓面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 為了使雷達圖一圈封閉起來，需要下面的步驟
    values=np.concatenate((values,[values[0]]))

    angles=np.concatenate((angles,[angles[0]]))

    # 繪圖
    fig= plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # 繪製折線圖
    ax.plot(angles, values, 'o-', linewidth=0.2, label = 'sample')
    # 填充顏色
    ax.fill(angles, values, alpha=0.25)
    # 繪製第二條折線圖


    # 添加每個特徵的標籤
    ax.set_thetagrids(angles * 180/np.pi, feature)
    # 設置雷達圖的範圍
    ax.set_ylim(0,100)
    # 添加標題
    plt.title('Radar_subject 02')

    # 添加網格線
    ax.grid(True)
    # 設置圖例
    plt.legend(loc = 'best')
    # 顯示圖形
    filename = posixpath.join(TMP_DIR_PATH, 'radar.png')
    plt.savefig(filename)
    # plt.show()


# In[11]:


def Rader_compare(Latent_code1,Latent_code2):
    samplesrar1 = np.interp(Latent_code1, (Latent_code1.min(), Latent_code1.max()), (50, 100)) #Rescale to 0-100
    samplesrar2 = np.interp(Latent_code2, (Latent_code2.min(), Latent_code2.max()), (50, 100)) #Rescale to 0-100
    # 中文和負號的正常顯示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 使用ggplot的繪圖風格
    plt.style.use('ggplot')

    # 構造數據
    values1 = samplesrar1[0]
    values2 = samplesrar2[0]


    feature = ['feature01','feature02','feature03','feature04','feature05']

    N = len(values1)
    # 設置雷達圖的角度，用於平分切開一個圓面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 為了使雷達圖一圈封閉起來，需要下面的步驟
    values1=np.concatenate((values1,[values1[0]]))
    values2=np.concatenate((values2,[values2[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    # 繪圖
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # 繪製折線圖
    ax.plot(angles, values1, 'o-', linewidth=0.2, label = 'you')
    # 填充顏色
    ax.fill(angles, values1, alpha=0.25)
    # 繪製第二條折線圖


    # 繪製折線圖
    ax.plot(angles, values2, 'o-', linewidth=0.2, label = 'veteran')
    # 填充顏色
    ax.fill(angles, values2, alpha=0.25)
    # 繪製第二條折線圖



    # 添加每個特徵的標籤
    ax.set_thetagrids(angles * 180/np.pi, feature)
    # 設置雷達圖的範圍
    ax.set_ylim(0,100)
    # 添加標題
    plt.title('Radar')

    # 添加網格線
    ax.grid(True)
    # 設置圖例
    plt.legend(loc = 'best')
    # 顯示圖形
    filename = posixpath.join(TMP_DIR_PATH, 'radar.png')
    plt.savefig(filename)
    # plt.show()
