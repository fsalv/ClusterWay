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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # remove debugging tesorflow
PATH_DIR = os.path.abspath('.')

import numpy as np
import cv2
import tensorflow as tf
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans

from utils.tools import load_config, deepWayLoss, clusterLoss, get_scheduler, APCalculator
from utils.models import build_deepway, build_clusterway
from utils.dataset import load_dataset_test
from utils.logger import Logger
from utils.train import Trainer
from utils.geometry import line_polar, line_polar_to_cart
from utils.postprocessing import order_cluster, estimate_row_angle, cluster_wp, get_principal_clusters


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch cluster baseline computation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--name", type=str, default='deep_way', help="Model name")
    parser.add_argument("--curved", action='store_true', help="Model type")
    parser.add_argument("--i", type=int, default=0,  help="Iteration")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--config_file", type=str, default=os.path.join(PATH_DIR, 'config.json'),
                        help="Config file")
    
    parser.add_argument("--TEST_DATA_PATH", type=str,
                        default=os.path.join(PATH_DIR, '../Datasets/straight/test'),
                        help="Test data path")
    parser.add_argument("--TEST_CURVED_DATA_PATH", type=str,
                        default=os.path.join(PATH_DIR, '../Datasets/curved/test'),
                        help="Test curved data path")
    parser.add_argument("--TEST_SATELLITE_DATA_PATH", type=str,
                        default=os.path.join(PATH_DIR, '../Datasets/satellite'),
                        help="Test satellite data path")
    parser.add_argument("--TEST_SATELLITE_CURVED_DATA_PATH", type=str,
                        default=os.path.join(PATH_DIR, '../Datasets/satellite_curved'),
                        help="Test satelllite curved data path")
    
    parser.add_argument("--PATH_WEIGHTS", type=str, default=os.path.join(PATH_DIR, 'bin'),
                        help="Weights data path")
    parser.add_argument("--LOG_DIR", type=str, default=os.path.join(PATH_DIR, 'logs'),
                        help="Logger and Tensorboard log folder")
    parser.add_argument("--clear_file", action='store_true', help="Clear log file")
    parser.add_argument("--seed", type=int, default=None, help="Seed for clustering reproducibility")
    return parser.parse_args()


def dbscan_clustering(wp, img, verbose=False, seed=None):
    if seed is not None: # np seed
        rng = np.random.RandomState(seed)
        cv2.setRNGSeed(seed)
    else:
        rng = np.random
    # cv2 seed
    
    mask = np.bitwise_not(img.astype('bool')).astype('float32')
    row_angle,row_vector,row_normal = estimate_row_angle(mask, rng=rng)
    labels,lab_sorted = cluster_wp(wp, eps=40, verbose=verbose)

    wp_cl = []
    for label in lab_sorted:
        wp_cli = order_cluster(wp[labels==label],row_normal)
        wp_cl.append(wp_cli)
    for p in wp[labels==-1]:   # add non-clustered points
        wp_cl.append(np.array([p]))

    cluster_a, cluster_b = get_principal_clusters(mask, wp_cl, row_angle,row_normal)
    
    cl_pred = []
    for j,p in enumerate(wp):
        c = np.argmin((np.linalg.norm(cluster_a - p, axis=1).min(), np.linalg.norm(cluster_b - p, axis=1).min()))
        cl_pred.append(c)
    cl_pred = np.array(cl_pred)
    return cl_pred


def kmeans_clustering(wp, seed=None):
    classifier = KMeans(2, random_state=seed)
    cl_pred = classifier.fit_predict(wp)
    return cl_pred


def clustering_baselines(X_test, df_waypoints_test, AP, seed=None):
    cl_acc_kmeans = []
    cl_acc_dbscan = []
    err_kmeans = []
    err_dbscan = []
    no_cl = 0

    for index in tqdm(range(len(X_test))):
        img = X_test[index]
        gt = df_waypoints_test.loc[df_waypoints_test['N_img'] == f"img{index + AP.const}"].to_numpy()[:,1:4].astype('int32')
        cl_true = gt[:,-1]
        gt = gt[:,:-1]
        wp = AP.y_pred_interpreted[index][0]

        if not len(wp) or len(wp)==1: # 1 or 0 predictions, no clustering needed
            print(f"[WARNING] Image {index} has {len(wp)} predictions with conf {config['conf_cl_test']}")
            no_cl +=1
            continue

        # KMeans
        wp_class_kmeans = kmeans_clustering(wp, seed=seed)
        acc, err = AP.compute_cluster_accuracy(wp, wp_class_kmeans, gt, cl_true, True)
        cl_acc_kmeans.append(acc)
        err_kmeans.append(err)

        # DBSCAN
        wp_class_old = dbscan_clustering(wp, img, seed=seed)
        acc, err = AP.compute_cluster_accuracy(wp, wp_class_old, gt, cl_true, True)
        cl_acc_dbscan.append(acc)
        err_dbscan.append(err)
        
        return np.mean(cl_acc_kmeans), np.mean(cl_acc_dbscan), np.mean(err_kmeans), np.mean(err_dbscan), no_cl



def cluster(args, config):
    name_model = args.name + f'_{args.i}'
    
    tf.keras.backend.clear_session()
    if 'deep_way' in name_model:
        deepway_net = build_deepway(name_model, config['FILTERS'],
                                config['KERNEL_SIZE'],
                                config['R'], config['MASK_DIM'])
    else:
        raise ValueError(f'Wrong model {name_model}.') # clustering baseline is computed for classic deep_way only
    
    
    # load weights
    loss={'mask': deepWayLoss('none')}
    Trainer(deepway_net, config, loss=loss, optimizer=tf.keras.optimizers.Adam(0.),
            checkpoint_dir=args.PATH_WEIGHTS, logger=logger)
    
    # test phase
    for test_folder in (args.TEST_DATA_PATH, args.TEST_CURVED_DATA_PATH,
                        args.TEST_SATELLITE_DATA_PATH, args.TEST_SATELLITE_CURVED_DATA_PATH):
        X, _, _, df = load_dataset_test(test_folder, config)
        AP = APCalculator(X, df, deepway_net, test_folder==args.TEST_SATELLITE_CURVED_DATA_PATH)
        AP.interpret(conf_min=config['conf_min_test'], K=config['K'], wp_prox_sup_thresh=config['wp_sup_thresh'])
        t = f"Clustering baselines metric on {os.path.relpath(test_folder, PATH_DIR)} dataset:"

        cl_acc_kmeans, cl_acc_old, err_kmeans, err_old, no_cl = clustering_baselines(X, df, AP, args.seed)
        
        t += f"\n\tKmeans\t cl_acc: {cl_acc_kmeans}\n\t\t errors: {err_kmeans}\n\t\t no_cl: {no_cl}"
        t += f"\n\tDBSCAN\t cl_acc: {cl_acc_old}\n\t\t errors: {err_old}\n\t\t no_cl: {no_cl}\n"       
        logger.log(t)        
        
        
def main():
    global logger
    args = get_args()
    logger = Logger(os.path.join(args.LOG_DIR, 'log_baseline_cluster.txt'), clear_file = args.clear_file,
                    add_time=True, print_to_stdout=True)
    
    args.name += '_curved' if args.curved else ''
    
    # select a GPU and set memory growth 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
    
    config = load_config(args.config_file)
    
    text = '\n\n'
    text += '-'*50 + '\n'
    text += '-'*50 + '\n'
    text += f'Starting baseline clustering \n'
    text += f'Model {args.name}, iteration {args.i}\n'
    text += '-'*50 + '\n'
    if args.seed is not None:
        text += f'Random seed set to {args.seed}'
    logger.log(text)
    try:
        cluster(args, config)
        logger.log('-'*50)
    except Exception as e:
        raise
            
            
if __name__ == '__main__':
    main()