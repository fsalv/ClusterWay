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

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json, cv2

######################## Configuration ########################    


def load_config(config_path='config.json'):
    """
    Load config file
    """
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    
    return config


######################## Rotate Image ########################    

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
    
######################## Ground truth computation ########################    


def resizeAddCorrection(y, y_cluster, WAYP_VALUE=1, K=800, extended=False, correction=True,
                        normalize=True, IRD_MEAN=17, IRD_STD=10, y_end=None):
    """
    Take a waypoint mask and transform it in a scaled (K) version with three channels.
    First channels rescaled waypoints, second one x correction coordinates and last
    one y correction coordinates. Transform also features and end matrices.
    """
    if normalize:
        if extended: # position, angles, ird 
            y[...,-2] = (y[...,-2]-np.pi/2)/(np.pi/2)     # angle norm
            y[...,-1] = (y[...,-1]-IRD_MEAN)/IRD_STD      # ird norm
        norm = K // 2
    else:
        norm = 1
    dim = 1 + correction*2 + extended*2
    y_complete = np.zeros((y.shape[0], y.shape[1]//K, y.shape[2]//K, dim))
    y_complete_cluster = np.zeros((y.shape[0], y.shape[1]//K, y.shape[2]//K))
    if y_end is not None:
        y_complete_end = np.zeros(y_complete_cluster.shape)
    for index in range(y.shape[0]):
        row, col, _ = np.where(y[index] == WAYP_VALUE)
        row_res, col_res = row // K, col //K
        if correction:
            y_complete[index, row_res, col_res, 0] = WAYP_VALUE
            y_complete[index, row_res, col_res, 1] = (((row % K) - ((K // 2) + 1)) / norm)
            y_complete[index, row_res, col_res, 2] = (((col % K) - ((K // 2) + 1)) / norm)
        else:
            y_complete[index, row_res, col_res, 0] = WAYP_VALUE
        if extended:
            y_complete[index, row_res, col_res, -2:] = y[index, row, col, -2:]
        y_complete_cluster[index, row_res, col_res] = y_cluster[index, row, col]
        if y_end is not None:
            y_complete_end[index, row_res, col_res] = y_end[index, row, col]
    if y_end is not None:
        return y_complete, y_complete_cluster, y_complete_end
    else:
        return y_complete, y_complete_cluster
    
    
######################## Training Tools ########################    


def get_scheduler(scheduler, lr, tot_steps):
    if scheduler=='step':
        sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([tot_steps//2, tot_steps*3//4], [lr, lr/10, lr/100])
    elif scheduler=='cosine':
        sched = tf.keras.optimizers.schedules.CosineDecayRestarts(lr, tot_steps//4, t_mul=1, m_mul=0.7, alpha=0.01)
    elif scheduler=='constant':
        sched = lr
    else:
        raise "Scheduler error"
    return sched
    

class clusterLoss(tf.keras.losses.Loss):
    """
    Constrastive loss for features-matching clustering ispired by ArXiv:2002.05709
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='cluster_loss'):
        super().__init__(reduction=reduction, name=name)
        self.ZERO_LOSS = 0.3132617 #to shift loss, so that targets zero (visualization only)
    
    def call(self, y_true, y_pred):
        cluster_true = y_true
        features = y_pred
        
        batch_size = tf.shape(features)[0]

        #cluster one per-image mask and features
        one = tf.constant(1, dtype=tf.float32)
        where = tf.equal(cluster_true, one)
        indices_one = tf.where(where)
        feats_one_size = tf.shape(indices_one)[0]
        feats_one = tf.gather_nd(features, indices_one)
        range_repeated_one = tf.repeat(tf.range(batch_size, dtype=tf.int64)[:,None], feats_one_size, axis=1)
        per_image_mask_one = tf.cast(tf.equal(indices_one[:,0], range_repeated_one), tf.float32)

        #cluster two per-image mask and features
        two = tf.constant(2, dtype=tf.float32)
        where = tf.equal(cluster_true, two)
        indices_two = tf.where(where)
        feats_two_size = tf.shape(indices_two)[0]
        feats_two = tf.gather_nd(features, indices_two)
        range_repeated_two = tf.repeat(tf.range(batch_size, dtype=tf.int64)[:,None], feats_two_size, axis=1)
        per_image_mask_two = tf.cast(tf.equal(indices_two[:,0], range_repeated_two), tf.float32)

        #cumulative feats and masks
        feats = tf.concat((feats_one,feats_two), axis=0)
        per_image_mask = tf.concat((per_image_mask_one, per_image_mask_two), axis=1)
        per_image_mask_one_extended = tf.concat((per_image_mask_one, tf.zeros((batch_size, feats_two_size))), axis=1)
        per_image_mask_two_extended = tf.concat((tf.zeros((batch_size, feats_one_size)), per_image_mask_two), axis=1)

        #positive and negative masks
        diag_mask = 1 - tf.eye(feats_one_size + feats_two_size) #to avoid k=i and j=i
        mask_cluster_one = tf.reduce_sum(per_image_mask_one_extended[:,None]*per_image_mask_one_extended[...,None], axis=0)
        mask_cluster_two = tf.reduce_sum(per_image_mask_two_extended[:,None]*per_image_mask_two_extended[...,None], axis=0)
        mask_cluster = tf.reduce_sum((mask_cluster_one,mask_cluster_two), axis=0)
        mask_pos = mask_cluster * diag_mask
        mask_image = tf.reduce_sum(per_image_mask[:,None]*per_image_mask[...,None], axis=0)
        mask_neg = mask_image-mask_cluster

        #cosine similarities and pos/neg losses
        sim = cos_sim(feats,feats)
        sig_sim = tf.sigmoid(sim)
        neg_loss_pos = tf.math.log(sig_sim) * mask_pos
        neg_loss_neg = tf.math.log(1-sig_sim) * mask_neg
        mask_tot = mask_pos + mask_neg
        neg_loss_tot = (neg_loss_pos + neg_loss_neg) * mask_tot
        neg_loss = tf.reduce_sum(neg_loss_tot, axis=-1) / tf.reduce_sum(mask_tot, axis=-1)
        return -neg_loss - self.ZERO_LOSS


def cos_sim(u,v):
    """
    Cosine similarity
    """
    u = tf.linalg.l2_normalize(u, axis=-1)
    v = tf.linalg.l2_normalize(v, axis=-1)
    return tf.matmul(u, v, transpose_b=True)


class deepWayLoss(tf.keras.losses.Loss):
    """
    MSE weighted for true and false waypoints
    """    
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, weights=[0.3, 0.7], name='deep_path_loss'):
        super().__init__(reduction=reduction, name=name)
        self.weights = weights
    
    def call(self, y_true, y_pred): 
        # find zero values in y_true
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.equal(y_true[...,0], zero)
        indices_zero = tf.where(where)

        y_true_zero = tf.gather_nd(y_true, indices_zero)
        y_pred_zero = tf.gather_nd(y_pred, indices_zero)

        # find one values in y_true
        one = tf.constant(1, dtype=tf.float32)
        where = tf.equal(y_true[...,0], one)
        indices_one = tf.where(where)

        y_true_one = tf.gather_nd(y_true, indices_one)
        y_pred_one = tf.gather_nd(y_pred, indices_one)
        
        l_one = tf.keras.metrics.mean_squared_error(y_true_zero, y_pred_zero)*self.weights[0]
        l_zero = tf.keras.metrics.mean_squared_error(y_true_one, y_pred_one)*self.weights[1]

        return tf.concat((l_one, l_zero), axis=0)

    
######################## Output interpretation ########################    


def waypointProxSup(wp, conf, d=8):
    """
    Eliminate points too near to each other using the Euclidean distance
    """
    order = np.argsort(conf)[::-1]
    wp = wp[order]
    conf = conf[order]
    distances = np.linalg.norm(wp - np.repeat(wp[:,None], len(wp), axis=1), 2, axis=-1)
    where = distances<d
    selectable = np.ones(len(wp), dtype='bool')
    selected = np.zeros(len(wp), dtype='bool')

    for i in range(len(wp)):
        if selectable[order[i]]:
            selected[order[i]] = True
            indices = np.where(where[i])[0]
            selectable[order[indices]] = False
    return selected



def interpret(y_pred, conf_thresh, dist_thresh=3, normalize=True, features=True, waypoint_prox_sup=True, K=8):
    """ BATCH Interpret predictions rescaling to the original dimension usinng correction coordinates. 
    If waypoint_prox_sup=True it applies a proximity suppression for the points in the scaled dimension.
    """
    if normalize:
        norm = K // 2
    else:
        norm = 1
 
    if features:
        assert isinstance(y_pred, dict)
        features = y_pred['features']
    else:
        features = None
    if isinstance(y_pred, dict):   
        y_pred = y_pred['mask']

    wp = []
    for i in range(y_pred.shape[0]):
        row, col = np.where(y_pred[i,...,0]>conf_thresh)
        coors = y_pred[i,row,col,1:] #corrections
        conf = y_pred[i,row,col,0]    #confidences
        wp_x = ((row * K) + coors[:,0] * norm + ((K // 2) + 1)).astype('int')
        wp_y = ((col * K) + coors[:,1] * norm + ((K // 2) + 1)).astype('int')
        wp_i = np.stack((wp_x, wp_y), axis=1)
        selected = np.ones(len(wp_i), dtype='bool')
        if waypoint_prox_sup:
            selected = waypointProxSup(wp_i, conf, dist_thresh)
        out_i = [wp_i[selected,::-1], conf[selected]]
        if features is not None:
            feats = features[i,row,col][selected]
            out_i.append(feats)
        wp.append(out_i)
    return wp


# def upRes(y_pred, normalize=True, MASK_DIM = 800, K = 8, WAYP_VALUE = 255):
#     """BATCH Rescale prediction taking into account correction coordinates"""
#     if normalize:
#         norm = (K // 2)
#     else:
#         norm = 1
#     y_up = np.zeros((y_pred.shape[0], MASK_DIM, MASK_DIM))
#     for i in range(y_pred.shape[0]):
#         row, col = np.where(y_pred[i,:,:,0]==WAYP_VALUE)
#         for r, c in zip(row, col):
#             coors = y_pred[i,r,c,1:]
#             r_o = int((r * K) + coors[0] * norm + ((K // 2) + 1))
#             c_o = int((c * K) + coors[1] * norm + ((K // 2) + 1))
#             y_up[i,r_o,c_o] = WAYP_VALUE
#     return y_up



######################## Results ########################    


class APCalculator():
    def __init__(self, X_test, df_waypoints_test, model, satellite_curved=False):
        self.predict(X_test, model)
        self.df_waypoints_test = df_waypoints_test
        self.interpret_params = {'conf_min': None, 'waypoint_prox_sup': None, 'K': None, 'wp_prox_sup_thresh': None}
        self.interpreted = False
        self.clustered = False
        self.const = 100 if satellite_curved else 0
    
    
    def predict(self, X, model):
        self.features = False
        y_pred = []
        for index in tqdm(range(len(X))):
            y_pred.append(model.predict(X[index:index+1]))
        
        self.y_pred_test = np.concatenate([y['mask'] for y in y_pred])
        if 'features' in y_pred[0]:
            self.y_pred_test = {'mask': self.y_pred_test}
            self.y_pred_test['features'] = np.concatenate([y['features'] for y in y_pred])
            self.features = True

            
    def interpret(self, conf_min=0.1, waypoint_prox_sup=True, K=8, wp_prox_sup_thresh=8):
        self.interpret_params = {'conf_min': conf_min, 'waypoint_prox_sup': waypoint_prox_sup,
                                 'K': K, 'wp_prox_sup_thresh': wp_prox_sup_thresh}
        self.y_pred_interpreted = interpret(self.y_pred_test, conf_thresh=conf_min, 
                                            K=K, features=self.features, dist_thresh=wp_prox_sup_thresh, 
                                            waypoint_prox_sup=waypoint_prox_sup)
        self.interpreted = True
         
    
    def cluster(self, conf=0.1, K=8, waypoint_prox_sup=True, wp_prox_sup_thresh=8, seed=None):
        if not self.interpreted or conf<self.interpret_params['conf_min']:
            self.interpret(conf, waypoint_prox_sup, K, wp_prox_sup_thresh)       
        clusterer = KMeans(2, random_state=seed)
        wp_classes = []
        wp_conf = []
        for index in range(len(self.y_pred_interpreted)):
            y_pred = self.y_pred_interpreted[index][0]
            y_conf = self.y_pred_interpreted[index][1]
            y_features = self.y_pred_interpreted[index][2]
            y_pred = y_pred[y_conf>=conf]
            y_features = y_features[y_conf>=conf]
            y_conf = y_conf[y_conf>=conf]
            if not len(y_pred) or len(y_pred)==1: # 1 or 0 predictions, no clustering
                wp_classes.append([])
                wp_conf.append([])
                continue
            y_features /= np.linalg.norm(y_features, axis=-1, keepdims=True)
            wp_classes.append(clusterer.fit_predict(y_features))
            wp_conf.append(y_conf)
        self.wp_classes = {'wp_classes' : wp_classes, 'wp_conf': wp_conf}
        self.clustered = True
        
    
    def cluster_accuracy(self, conf=0.1, K=8, waypoint_prox_sup=True, wp_prox_sup_thresh=8, return_errors=False, seed=None):
        if not self.features:
            raise ValueError('The model does not provide clustering features.')

        if not self.clustered or (conf<self.interpret_params['conf_min'] if self.interpreted else True):
            self.cluster(conf, K, waypoint_prox_sup, wp_prox_sup_thresh,seed=seed)

        cl_acc = []
        errors = []
        no_cl = 0

        for index in range(len(self.y_pred_interpreted)):
            y_pred = self.y_pred_interpreted[index][0]
            y_conf = self.y_pred_interpreted[index][1]
            y_pred = y_pred[y_conf>=conf]
            if not len(y_pred) or len(y_pred)==1: # 1 or 0 predictions, no clustering
                print(f'[WARNING] Image {index} has {len(y_pred)} predictions with conf {conf}')
                no_cl +=1
                continue
            cluster_pred = self.wp_classes['wp_classes'][index]
            cluster_conf = self.wp_classes['wp_conf'][index]
            cluster_pred = cluster_pred[cluster_conf>=conf]
            wp = self.df_waypoints_test.loc[self.df_waypoints_test['N_img']==f'img{index + self.const}'].to_numpy()[...,1:4].astype('uint32')
            cluster_true = wp[...,-1]
            wp = wp[...,:-1]
            
            acc = self.compute_cluster_accuracy(y_pred, cluster_pred, wp, cluster_true, return_errors)
            if return_errors:
                acc, err = acc[0], acc[1]
                errors.append(err)
            cl_acc.append(acc)
        if return_errors:
            return np.mean(cl_acc), np.mean(errors), no_cl
        return np.mean(cl_acc), no_cl
    
    
    def compute_cluster_accuracy(self, wp_pred, cluster_pred, wp_true, cluster_true, return_errors=False):
        dist = np.linalg.norm(wp_pred - np.repeat(wp_true[:,None], len(wp_pred), axis=1), 2, axis=-1)
        cluster_target = cluster_true[np.argmin(dist, axis=0)]
        acc = np.max((np.mean(cluster_target == cluster_pred),
                      np.mean(cluster_target == 1-cluster_pred)))
        metric = 2*acc-1 #scale to 0-1
        if return_errors:
            return metric, round((1-acc)*len(wp_pred)) 
        return metric
    
    
    def compute(self, dist_range=8, conf_min=0.1, K=8, waypoint_prox_sup=True, wp_prox_sup_thresh=8):
        interpret_params = {'conf_min': conf_min, 'waypoint_prox_sup': waypoint_prox_sup,
                            'K': K, 'wp_prox_sup_thresh': wp_prox_sup_thresh}
        self.dist_range = dist_range
        if not self.interpreted or not interpret_params == self.interpret_params:
            self.interpret(conf_min, waypoint_prox_sup, K, wp_prox_sup_thresh)
        
        true_p = []

        for index in range(len(self.y_pred_interpreted)):                
            y_pred = self.y_pred_interpreted[index][0]
            y_conf = self.y_pred_interpreted[index][1]
            if not len(y_pred): # no predictions, no true positive nor negative
                continue
            wp = self.df_waypoints_test.loc[self.df_waypoints_test['N_img']==f'img{index+self.const}'].to_numpy()[...,1:3].astype('uint32')

            pred_available = np.ones(len(y_pred), 'bool')

            dist = np.linalg.norm(y_pred - np.repeat(wp[:,None], len(y_pred), axis=1), 2, axis=-1)
            try:
                order = np.argsort(np.min(dist, axis=1))
            except:
                print(index)
                raise
            for i_gt in range(len(wp)):
                idx = np.argmin(dist[order[i_gt]])
                if dist[order[i_gt]][idx] <= dist_range and pred_available[idx]:
                    pred_available[idx] = False
            true_p.extend(np.bitwise_not(pred_available))

        true_p = np.array(true_p)

        conf = np.concatenate([w[1] for w in self.y_pred_interpreted])
        order = np.argsort(conf)[::-1]
        conf = conf[order]
        true_p = true_p[order]
        true_p_cum = np.cumsum(true_p)
        precision = true_p_cum/np.arange(1, len(true_p)+1)
        recall = true_p_cum/len(self.df_waypoints_test)
        precision = np.concatenate((precision, [0]))
        recall = np.concatenate((recall, [recall[-1]]))
        self.ap = self.compute_ap(precision, recall)
        self.precision = precision
        self.recall = recall
        return self.ap, self.precision, self.recall
    
    
    def compute_ap(self, precision, recall):
        ap = np.sum(np.diff(recall)* np.array(precision)[1:])
        return ap
    
    
    def plot(self, recall=None, precision=None, ap=None, dist_range=None):
        if recall is None or precision is None:
            ap = self.ap
            recall = self.recall
            precision = self.precision
            dist_range = self.dist_range
        else:
            assert recall is not None and precision is not None
            if ap is None:
                ap = self.compute_ap(precision, recall)
                
        plt.plot(recall, precision, c='r')
        plt.title(f"AP" + f"@{dist_range}: {ap:.4f}" if dist_range is not None else "")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim((-0.05,1.05))
        plt.ylim((-0.05,1.05))
        plt.grid()
        plt.show()
        