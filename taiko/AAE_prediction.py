#!/usr/bin/env python
# coding: utf-8

# In[300]:


import numpy as np
import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility


import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from .config import *

import tensorflow as tf
from tensorflow.python.client import device_lib

__all__ = ['execute']


# In[208]:


def create_lstm_vae_first_train(input_dim,
    batch_size,
    intermediate_dim,
    latent_dim,
    epsilon_std=1.):

    input_dim = 24*12
    intermediate_dim = 16
    batch_size = 1
    epsilon_std = 1
    x = Input(shape=(input_dim,))

    # LSTM encoding
    h = Dense(intermediate_dim)(x)

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
    encoder = Model(x, z_mean)

    # decoded NN layer
    decoder_h = Dense(intermediate_dim)
    decoder_h0 = Dense(64)
    decoder_h1 = Dense(32)
    decoder_mean = Dense(input_dim)
    drop_ = Dropout(0.3, noise_shape=None, seed=None)

    #generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    added0 = keras.layers.concatenate([decoder_input,_h_decoded],axis = -1)
    _h_decoded = decoder_h0(added0)
    added1 = keras.layers.concatenate([decoder_input,_h_decoded],axis = -1)
    _h_decoded = decoder_h1(added1)

    added2 = keras.layers.concatenate([decoder_input,_h_decoded],axis = -1)
    _x_decoded_mean = decoder_mean(added2)
    generator = Model(decoder_input, _x_decoded_mean)

    # decoded layer
    h_decoded = decoder_h(z)
    added0 = keras.layers.concatenate([z,h_decoded],axis = -1)
    h_decoded = decoder_h0(added0)
    added1 = keras.layers.concatenate([z,h_decoded],axis = -1)
    h_decoded = decoder_h1(added1)
    added2 = keras.layers.concatenate([z,h_decoded],axis = -1)
    x_decoded_mean = decoder_mean(added2)

    def build_model_regression():

        #vae之後增加regression layers
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(64, activation="relu"))
#         model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(1))

        return model

    def build_model_disc():
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(32, activation="relu", input_shape=(20,)))
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        return model

    def build_model_aae_regression():

        regression = build_model_regression()
        discriminator = build_model_disc()

        model_vae = Model(x, x_decoded_mean)

        model_enc_disc = keras.models.Sequential()
        model_enc_disc.add(encoder)
        model_enc_disc.add(discriminator)

        model_vae_regression = keras.models.Sequential()
        model_vae_regression.add(model_vae)
        model_vae_regression.add(regression)

        return discriminator, model_vae, model_enc_disc, regression, model_vae_regression

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.01 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    discriminator, model_vae, model_enc_disc, regression, model_vae_regression = build_model_aae_regression()

#     model_vae.compile(optimizer='RMSprop', loss=vae_loss)
    model_vae.compile(optimizer='rmsprop', loss=vae_loss,metrics=['mse'])
    discriminator.compile(optimizer=keras.optimizers.Adam(lr=1e-7), loss="binary_crossentropy")
    model_enc_disc.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss="binary_crossentropy")
    regression.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0), loss=root_mean_squared_error,metrics=['mae','mean_absolute_percentage_error'] )
    model_vae_regression.compile(optimizer='rmsprop',  loss=root_mean_squared_error,metrics=['mae','mean_absolute_percentage_error'])

    return encoder, generator, discriminator, model_vae, model_enc_disc, regression, model_vae_regression


# In[209]:


def load_data(Dir):
    X_train = []
#     print ('Loading data...')
    filenames_don = glob.glob(Dir+'/motifs/don/*.csv')
#     print ('glob finish')

    for file in filenames_don:
        df = pd.read_csv(file)
        df = df.drop('Unnamed: 0',axis=1)
        df = df.drop('index',axis=1)
        X_train.append(df.values.flatten()) #攤平
#     print ('number of don: '+str(len(filenames_don)))

    samples_num = len(X_train)
    X_train = np.asarray(X_train)

#     print('X_train_shape: '+str(X_train.shape))
#     print('finish !')
    return X_train


# In[210]:


def AAE_analyze(Dir):
        X_train = load_data(Dir)
        input_dim = X_train.shape[-1] # 13
#         print ('input_dim: '+str(input_dim))
        batch_size = 1
        encoder, generator, discriminator, model_vae, model_enc_disc, regression, model_vae_regression = create_lstm_vae_first_train(input_dim,
                batch_size=batch_size,
                intermediate_dim=16,
                latent_dim=20,
                epsilon_std=1.)

        encoder.load_weights(ENCODER_MODEL_PATH)
        regression.load_weights(REGRESSION_MODEL_PATH)

        encoded_prediction = encoder.predict(X_train)[:,10]# node 11號
        encoded_mean = encoded_prediction.mean()

        score_prediction = regression.predict(X_train)
        score_mean = score_prediction.mean()
        return encoded_prediction, score_prediction, encoded_mean, score_mean


# In[489]:

def prepare_env():
    # 隨時間增加配置 的 GPU 記憶體
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # # 只使用 80% 的 GPU 記憶體
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # 設定 Keras 使用的 TensorFlow Session

    keras.backend.set_session(sess)
    K.set_learning_phase(1)
    print(device_lib.list_local_devices())

def execute():
    prepare_env()
    coordinate = []
    df = pd.read_csv('F:/MingChen/Taiko-Master-go/taiko/veteran.csv')

    encoded_mean = df['encoded_mean']
    score_variance = df['score_variance']

    encoded_mean_list = encoded_mean.tolist()
    score_variance_list = score_variance.tolist()

    temp_Dir = 'F:/MingChen/Taiko-Master-go/tmp/sensor_data'
    ep_temp, sp_temp, em_temp, sm_temp = AAE_analyze(temp_Dir)
    em_temp = em_temp*(-1)  #因為邏輯值大等於小力 所以變號
    variance_temp = np.std(sp_temp)

    encoded_mean_list.append((em_temp))
    encoded_mean_array = np.asarray(encoded_mean_list)
    encoded_mean_array = np.interp(encoded_mean_array, (encoded_mean_array.min(), encoded_mean_array.max()), (-1, +1)) #Scale to 0-1
    score_variance_list.append((variance_temp))
    score_variance_array = np.asarray(score_variance_list)
    score_variance_array = np.interp(score_variance_array, (score_variance_array.min(), score_variance_array.max()), (-1, +1)) #Scale to 0-1

    plt.figure(dpi=800)
    plt.xlabel('Strength_evaluation')
    plt.ylabel('Stablility(STD),Unit: score')
    plt.scatter(encoded_mean_array[0:-1],score_variance_array[0:-1],label='veteran')
    plt.scatter(encoded_mean_array[-1],score_variance_array[-1],c='r',label='you')

    people =['A','B','C','D','E','F','G','H','I','J','K']

    for i, txt in enumerate(people):
        plt.annotate(txt, (encoded_mean_array[i], score_variance_array[i]))
#         plt.annotate(round(score_variance_array[i],3), (encoded_mean_array[i], score_variance_array[i]+0.05))
        coordinate.append([encoded_mean_array[i], score_variance_array[i]])

    coordinate.append([encoded_mean_array[-1],score_variance_array[-1]])
    coordinate = np.array(coordinate)

    def find_nearest_vector(array, value):
        Dist = []
        for item in (array-value):
            Dist.append((np.linalg.norm(item)))
        idx = Dist.index(min(Dist))
        return idx

    idx = find_nearest_vector(coordinate[0:-1],coordinate[-1])

    plt.legend()
    plt.savefig(posixpath.join(TMP_DIR_PATH, 'result.jpg'))

    return idx+1, sm_temp #打擊者平均don分數


# In[490]:


#### idx, sm_temp = excecute()


# #老手的資料
# Dir_list = ['G:/full_combo_of_song1/aaaa',
#             'G:/full_combo_of_song1/carolyn',
#             'G:/full_combo_of_song1/celvin',
#             'G:/full_combo_of_song1/chris',
#             'G:/full_combo_of_song1/cuxi',
#             'G:/full_combo_of_song1/eve',
#             'G:/full_combo_of_song1/however',
#             'G:/full_combo_of_song1/john',
#             'G:/full_combo_of_song1/kdchang',
#             'G:/full_combo_of_song1/oliver',
#             'G:/full_combo_of_song1/sheep',
#            ]
# encoded_prediction = []
# score_prediction = []
# encoded_mean= []
# score_mean = []
# score_variance = []
#
# for item in Dir_list:
#     ep_temp, sp_temp, em_temp, sm_temp = AAE_analyze(item)
#     encoded_prediction.append(ep_temp)
#     score_prediction.append(sp_temp)
#     em_temp = em_temp*(-1)  #因為邏輯值大等於小力 所以變號
#     encoded_mean.append(em_temp)
#     score_mean.append(sm_temp)
#     score_variance.append(np.std(sp_temp))#高手們的穩定度:
#
# dict_column = {
#         "encoded_prediction": np.array(encoded_prediction),
#         "score_prediction": np.array(score_prediction),
#         'encoded_mean': encoded_mean,
#         'score_mean': score_mean,
#         'score_variance': score_variance
# }
#
# df_veteran= pd.DataFrame(dict_column)
# df_veteran.to_csv('veteran.csv')

# In[ ]:
