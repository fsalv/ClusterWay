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
import tensorflow as tf
import argparse

from utils.tools import load_config, deepWayLoss, clusterLoss, APCalculator
from utils.models import build_deepway, build_clusterway
from utils.dataset import load_dataset_test
from utils.logger import Logger
from utils.train import Trainer


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch testing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--name", type=str, default='cluster_way', help="Model name")
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
    parser.add_argument("--seed", type=int, default=42, help="Seed for clustering reproducibility")
    return parser.parse_args()



def test(args, config):
    name_model = args.name + f'_{args.i}'
    
    # create model
    tf.keras.backend.clear_session()
    if 'deep_way' in name_model:
        deepway_net = build_deepway(name_model, config['FILTERS'],
                                config['KERNEL_SIZE'],
                                config['N'], config['MASK_DIM'])
    elif 'cluster_way' in name_model:
        j = name_model.find('cluster_way')
        name_classic = name_model[:j] + 'deep_way' + name_model[j+11:]
        
        model_classic = build_deepway(name_classic, config['FILTERS'],
                                config['KERNEL_SIZE'],
                                config['R'], config['MASK_DIM'], True)
     
        deepway_net = build_clusterway(name_model, model_classic, config['FILTERS'],
                                config['KERNEL_SIZE'], out_feats=config['OUT_FEATS'])
    else:
        raise ValueError(f'Wrong model {name_model}.')
        
    # load weights
    loss={'mask': deepWayLoss('none')}
    if 'cluster_way' in name_model:
        loss['features'] = clusterLoss('none')
    Trainer(deepway_net, config, loss=loss, optimizer=tf.keras.optimizers.Adam(0.),
            checkpoint_dir=args.PATH_WEIGHTS, logger=logger)

    # test phase
    clustering = 'cluster_way' in name_model # clustering head
    for test_folder in (args.TEST_DATA_PATH, args.TEST_CURVED_DATA_PATH,
                        args.TEST_SATELLITE_DATA_PATH, args.TEST_SATELLITE_CURVED_DATA_PATH):
        X, _, _, df = load_dataset_test(test_folder, config)
        AP = APCalculator(X, df, deepway_net, test_folder==args.TEST_SATELLITE_CURVED_DATA_PATH)
        AP.interpret(conf_min=config['conf_min_test'], K=config['K'], wp_prox_sup_thresh=config['wp_sup_thresh'])
        t = f"Test metric on {os.path.relpath(test_folder, PATH_DIR)} dataset:"
        if clustering:
            cl_acc, errors, no_cl = AP.cluster_accuracy(conf=config['conf_cl_test'], K=config['K'],
                                                        wp_prox_sup_thresh=config['wp_sup_thresh'],
                                                        return_errors=True, seed=args.seed)
            t += f"\n\t cl_acc: {cl_acc}\n\t errors: {errors}\n\t no_cl: {no_cl}"       
        for DIST_RANGE in (2,3,4,6,8):
            ap, _, _ =  AP.compute(DIST_RANGE, conf_min=config['conf_min_test'],
                                   K=config['K'], wp_prox_sup_thresh=config['wp_sup_thresh'])
            t += f"\n\t ap@{DIST_RANGE}: {ap}"
        t += "\n"
        logger.log(t)


        
def main():
    global logger
    args = get_args()
    logger = Logger(os.path.join(args.LOG_DIR, 'log_test.txt'), clear_file = args.clear_file,
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
    text += f'Starting testing \n'
    text += f'Model {args.name}, iteration {args.i}\n'
    text += '-'*50 + '\n'
    if args.seed is not None:
        text += f'Random seed set to {args.seed}'
    logger.log(text)
    try:
        test(args, config)
        logger.log('-'*50)
    except Exception as e:
        raise
            
            
if __name__ == '__main__':
    main()