
# coding: utf-8

# In[1]:


import glob as glob
import pandas as pd
import numpy as np
import posixpath
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import time
import sys
import random
import math as m
import importlib
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['Main_Execure']


def Main_Execure(Dir): #Dir 為motifs目錄
    Dir_L = posixpath.join(Dir, 'L')
    Dir_R = posixpath.join(Dir, 'R')

    Train_data = Input_motifs(Dir_L,Dir_R)

    print (Train_data.shape)
    Train_data = np.array(Train_data)[:6]
    distances = np.zeros((np.shape(Train_data)[0],np.shape(Train_data)[0]))


    w = Train_data.shape[1]

    start_time = time.time()
    for ind,i in enumerate(Train_data):
        for c_ind,j in enumerate(Train_data):
            if c_ind > ind:
                continue
            cur_dist = 0.0
            #Find sum of distances along each dimension
            for z in range(np.shape(Train_data)[2]):
                cur_dist += DTWDistance(i[:,z],j[:,z],w)
            distances[ind,c_ind] = cur_dist

    for ind in range(len(Train_data)):
        for c_ind in range(len(Train_data)):
            if distances[ind, c_ind] == 0:
                distances[ind, c_ind] = distances[c_ind, ind]

    print("distances--- %s seconds ---" % (time.time() - start_time))
    clusters, curr_medoids = cluster(distances, 3)



    #挑出分類群中最多的那群資料
    array_whichMany = []
    for i in range(len(np.unique(clusters))):
        index_i = np.unique(clusters)[i]
        index_i = (np.where(clusters==index_i)[0])
        locals()['samples_%s'%i] = Train_data[index_i]

        array_whichMany.append(len(locals()['samples_%s'%i]))

    which_group = array_whichMany.index(max(array_whichMany))
    samples = locals()['samples_%s'%(which_group)]

    timeseries_sample = []
    for i in range(samples.shape[1]):
        axe_mean = []

        for j in range(samples.shape[2]):
            axis_total =(samples[:,i,j])
            mean = axis_total.mean()  #sample shape= (20,7) 的 第i個time的第j個軸的平均值
            axe_mean.append(mean) #從軸0~7依序的平均值

        timeseries_sample.append(axe_mean) #20個timeseries都完成

    timeseries_sample = np.array(timeseries_sample)
    timeseries_sample = timeseries_sample.reshape(1,timeseries_sample.shape[0],timeseries_sample.shape[1])  ##最後的合成訊號
    df = pd.DataFrame(timeseries_sample[0])
    df.to_csv(Dir+'motifs.csv',index=False,header=['L_imu_ax',
                          'L_imu_ay',
                           'L_imu_az',
                           'L_imu_gx',
                           'L_imu_gy',
                           'L_imu_gz',
                           'R_imu_ax',
                           'R_imu_ay',
                           'R_imu_az',
                           'R_imu_gx',
                           'R_imu_gy',
                           'R_imu_gz',
                         ])

    print( 'sample finished')
    return timeseries_sample


# In[3]:


def preprocess(df):

        df = np.asarray(df, dtype=np.float32)
        if len(df.shape) == 1:
            raise ValueError('Data must be a 2-D array')

    #     if np.any(sum(np.isnan(df)) != 0):
    #         print('Data contains null values. Will be replaced with 0')
    #         df = np.nan_to_num()

        #standardize data
        df = StandardScaler().fit_transform(df)
        #normalize data
        df = MinMaxScaler().fit_transform(df)
        return df


# In[4]:


def Input_motifs(DirL,DirR):
    path1 = posixpath.join(DirL, '*.csv')
    path2 = posixpath.join(DirR, '*.csv')

    filenames1 = glob.glob(path1)
    filenames2 = glob.glob(path2)

    new_filename = []
    for i in range(len(filenames1)):
        df_left = pd.read_csv(filenames1[i])
        df_right =  pd.read_csv(filenames2[i])
        new_df = pd.concat([df_left,df_right],axis=1)
        new_df = new_df.drop(new_df.columns[7],axis=1)
        new_df.columns = ['L_imu_ax',
                          'L_imu_ay',
                           'L_imu_az',
                           'L_imu_gx',
                           'L_imu_gy',
                           'L_imu_gz',
                           'R_imu_ax',
                           'R_imu_ay',
                           'R_imu_az',
                           'R_imu_gx',
                           'R_imu_gy',
                           'R_imu_gz',
                         ]
        new_filename.append(new_df)


    samples_Scaling=[]

    for df in new_filename:
        sample_No = preprocess(df)
        samples_Scaling.append(sample_No)

    samples_Scaling = np.array(samples_Scaling)
    print( 'input finished')
    return samples_Scaling


# In[5]:



def cluster(distances, k):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k)
    new_medoids = np.array([-1]*k)

    # To be repeated until mediods stop updating
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    print( 'cluster finished')
    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

def DTWDistance(s1,s2,w):
    x = np.array(s1)
    y = np.array(s2)
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance

    # DTW={}
    # w = max(w, abs(len(s1)-len(s2)))
    #
    # for i in range(-1,len(s1)):
    #     for j in range(-1,len(s2)):
    #         DTW[(i, j)] = float('inf')
    # DTW[(-1, -1)] = 0
    #
    # for i in range(len(s1)):
    #     for j in range(max(0, i-w), min(len(s2), i+w)):
    #         dist= (s1[i]-s2[j])**2
    #         DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    #
    # return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    '''
    Calculates LB_Keough lower bound to dynamic time warping. Linear
    complexity compared to quadratic complexity of dtw.
    '''
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)


# Main_Execure('D:/Ming/Taiko-Master-develop/taiko/motif/aaaaa/song4/order4/don')

# Main_Execure('D:/Ming/Taiko-Master-develop/taiko/motif/aaaaa/song4/order4/ka')

# Main_Execure('D:/Ming/motif/aaaaa/song1/order4/don')
