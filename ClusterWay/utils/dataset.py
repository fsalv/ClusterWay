# Copyright 2022 Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import glob

from utils.tools import resizeAddCorrection


# def get_dist_map(end, config, r=10):
#     dim = config['MASK_DIM']//config['K']
#     dist = 1./(1e-5 + np.linalg.norm(np.array(list(np.ndindex(dim,dim))) - end[::-1], axis=1))**2
#     return tf.nn.tanh(r*dist.reshape(dim, dim))


def dataset_generator(TRAIN_DATA_PATH, config):
    def generator():
        data_n = config['DATA_N']
        MASK_DIM = config['MASK_DIM']   
        df = pd.read_csv(os.path.join(TRAIN_DATA_PATH, 'waypoints.csv'))
        indices = list(range(data_n))
        for index in indices:
            X = cv2.imread(os.path.join(TRAIN_DATA_PATH, f'img{index}.png'), cv2.IMREAD_GRAYSCALE)
            X = cv2.bitwise_not(X)
            points = df.loc[df['N_img'] == f'img{index}'].to_numpy()[:,1:].astype('uint32')
            points_x = points[:,0]
            points_y = points[:,1]
            clusters = points[:,2] + 1
            y = np.zeros((MASK_DIM, MASK_DIM, 1))
            y[points_y, points_x, 0] = config['WAYP_VALUE']
            y_cluster = np.zeros((MASK_DIM,MASK_DIM))
            indices_one = np.where(clusters==1)[0]
            indices_two = np.where(clusters==2)[0]
            y_cluster[points_y[indices_one], points_x[indices_one]] = config['CLUST_ONE_VALUE']
            y_cluster[points_y[indices_two], points_x[indices_two]] = config['CLUST_TWO_VALUE']
            y_complete, y_cluster_complete = resizeAddCorrection(y[None], 
                                                 y_cluster[None],
                                                 config['WAYP_VALUE'], config['K'],
                                                 extended=False)
            yield X, {'mask': y_complete[0], 'features': y_cluster_complete[0]}
    return generator


def get_dataset_train(TRAIN_DATA_PATH, config, return_df=False):
    dataset_train = tf.data.Dataset.from_generator(dataset_generator(TRAIN_DATA_PATH, config),
                    output_signature=(
                        tf.TensorSpec(shape=[config['MASK_DIM'],config['MASK_DIM']], dtype=tf.float32),
                        {'mask': tf.TensorSpec(shape=[config['MASK_DIM']//config['K'],
                                               config['MASK_DIM']//config['K'], 3], dtype=tf.float32),
                         'features': tf.TensorSpec(shape=[config['MASK_DIM']//config['K'],
                                               config['MASK_DIM']//config['K']], dtype=tf.float32)}))
    
    dataset_train = dataset_train.cache().shuffle(1000)
    dataset_train = dataset_train.batch(batch_size = config['BATCH_SIZE'], drop_remainder=True)
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
    if return_df:
        df = pd.read_csv(os.path.join(TRAIN_DATA_PATH, 'waypoints.csv'))
        return dataset_train, df
    return dataset_train


def load_dataset_val(VAL_DATA_PATH, config):
    data_n = config['DATA_N_VAL']
    MASK_DIM = config['MASK_DIM']  
    X = np.empty((data_n, MASK_DIM, MASK_DIM), dtype='float32')
    df = pd.read_csv(os.path.join(VAL_DATA_PATH, 'waypoints.csv'))
    y = np.empty((data_n, MASK_DIM, MASK_DIM, 1), dtype='float32')
    y_cluster = np.empty((data_n, MASK_DIM, MASK_DIM))
    for index in tqdm(range(data_n)):
        mask = cv2.imread(os.path.join(VAL_DATA_PATH, f'img{index}.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.bitwise_not(mask)
        points = df.loc[df['N_img'] == f'img{index}'].to_numpy()[:,1:].astype('uint32')
        points_x = points[:,0]
        points_y = points[:,1]
        clusters = points[:,2] + 1
        mask_points = np.zeros((MASK_DIM, MASK_DIM, 1))
        mask_points[points_y, points_x, 0] = config['WAYP_VALUE']
        mask_cluster = np.zeros((MASK_DIM,MASK_DIM))
        indices_one = np.where(clusters==1)[0]
        indices_two = np.where(clusters==2)[0]
        mask_cluster[points_y[indices_one], points_x[indices_one]] = config['CLUST_ONE_VALUE']
        mask_cluster[points_y[indices_two], points_x[indices_two]] = config['CLUST_TWO_VALUE']
        X[index] = mask
        y[index] = mask_points
        y_cluster[index] = mask_cluster
    y, y_cluster = resizeAddCorrection(y, y_cluster, config['WAYP_VALUE'],
                                              config['K'], extended=False)
    return X, y, y_cluster


def get_dataset_val(VAL_DATA_PATH, config, return_df=False):
    X_val, y_val_mask, y_val_features = load_dataset_val(VAL_DATA_PATH, config)
    dataset_val = tf.data.Dataset.from_tensor_slices((X_val.astype('float32'),
                                                      y_val_mask.astype('float32'),
                                                      y_val_features.astype('float32')))
    dataset_val = dataset_val.batch(batch_size = config['VAL_BATCH_SIZE'], drop_remainder=False)
    dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE) 
    if return_df:
        df = pd.read_csv(os.path.join(VAL_DATA_PATH, 'waypoints.csv'))
        return dataset_val, df
    return dataset_val



def load_dataset_test(TEST_DATA_PATH, config):
    img_list = glob.glob(os.path.join(TEST_DATA_PATH, '*.png'))
    img_list = sorted(img_list,key=lambda s: int(s.split('/')[-1][3:-4])) #sort by name
    data_n = len(img_list)
    MASK_DIM = config['MASK_DIM']
    df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'waypoints.csv'))
    X = np.empty((data_n, MASK_DIM, MASK_DIM), dtype='float32')
    y = np.empty((data_n, MASK_DIM, MASK_DIM, 1), dtype='float32')
    y_cluster = np.zeros((data_n, MASK_DIM, MASK_DIM), dtype='float32')
    for index,img in tqdm(enumerate(img_list), total=data_n):
        name = img.split('/')[-1][:-4]
        mask = cv2.bitwise_not(cv2.imread(img, cv2.IMREAD_GRAYSCALE)) # open grayscale and invert 255
        points = df.loc[df['N_img'] == name].to_numpy()[:,1:].astype('uint32')
        points_x = points[:,0]
        points_y = points[:,1]
        clusters = points[:,2] + 1
        ends = np.concatenate((points[:2], points[-2:]), axis=0)[:,:-1]
        mask_points = np.zeros((MASK_DIM,MASK_DIM, 1))
        mask_points[points_y, points_x] = config['WAYP_VALUE']
        mask_cluster = np.zeros((MASK_DIM,MASK_DIM))
        indices_one = np.where(clusters==1)[0]
        indices_two = np.where(clusters==2)[0]
        X[index,:,:] = mask
        y[index,:,:] = mask_points
        y_cluster[index] = mask_cluster
    y, y_cluster = resizeAddCorrection(y, y_cluster, config['WAYP_VALUE'], config['K'], extended=False)
    return X, y, y_cluster, df
