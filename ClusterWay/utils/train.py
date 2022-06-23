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

import tensorflow as tf
import numpy as np
from glob import glob
import os
from utils.tools import APCalculator

class Trainer:
    def __init__(self, model, config, loss, optimizer, loss_weights=None, checkpoint_dir='bin',
                 log_dir='logs', do_not_restore=False, force_restore=False, logger=None):
        
        self.print = logger.log if logger is not None else print
        self.config = config
        self.name = model.name
        
        if not isinstance(loss['mask'], tf.keras.losses.Loss) or not isinstance(loss['features'], tf.keras.losses.Loss) if 'features' in loss else False or not isinstance(loss['end'], tf.keras.losses.Loss) if 'end' in loss else False:
            raise TypeError("Losses should derive from 'tf.keras.losses.Loss'.")
            
        self.mask_loss = loss['mask']
        # ensure loss reduction strategy is 'none' to guarantee correct losses and gradients computation
        self.mask_loss.reduction = 'none'
        
        if 'features' in loss:
            self.features_loss = loss['features']
            self.features_loss.reduction = 'none'
        if 'end' in loss:
            self.end_loss = loss['end']
            self.end_loss.reduction = 'none'        
        
        self.loss_weights = {'mask': loss_weights['mask'] if loss_weights is not None else 1.,
                             'features': loss_weights['features'] if loss_weights is not None else 1.,
                             'end': loss_weights['end'] if loss_weights is not None else 1.}
        
        self.log_dir = os.path.join(log_dir, self.name)
        self.ckp_dir = os.path.join(checkpoint_dir, self.name)
        
        input_signature = [tf.TensorSpec(shape=(None, config['MASK_DIM'],
                                                config['MASK_DIM']), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, config['MASK_DIM']//config['K'],
                                                config['MASK_DIM']//config['K'], 3), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, config['MASK_DIM']//config['K'],
                                                config['MASK_DIM']//config['K']), dtype=tf.float32)]
        
        self.atomic_train_step = tf.function(input_signature=input_signature)(self.atomic_train_step)
        self.test_step = tf.function(input_signature=input_signature)(self.test_step)
        
        self.loss_accumulator = {'train': {'loss': tf.keras.metrics.Mean(name='loss'),
                                           'mask_loss': tf.keras.metrics.Mean(name='mask_loss'),
                                           'features_loss': tf.keras.metrics.Mean(name='features_loss'),
                                           'end_loss': tf.keras.metrics.Mean(name='end_loss')},
                         'val': {'val_loss': tf.keras.metrics.Mean(name='val_loss'),
                                 'val_mask_loss': tf.keras.metrics.Mean(name='val_mask_loss'),
                                 'val_end_loss': tf.keras.metrics.Mean(name='val_end_loss'),
                                 'val_features_loss': tf.keras.metrics.Mean(name='val_features_loss')}}
        
        self.test_metric_accumulator = {'AP@2': 0., 'AP@3': 0., 'AP@4': 0., 'AP@8': 0.}
        
        ckp_loss = {'loss': tf.Variable(np.inf), 'mask_loss': tf.Variable(np.inf),
                    'features_loss': tf.Variable(np.inf), 'end_loss': tf.Variable(np.inf),
                    'val_loss': tf.Variable(np.inf), 'val_mask_loss': tf.Variable(np.inf),
                    'val_features_loss': tf.Variable(np.inf), 'val_end_loss': tf.Variable(np.inf)}
        ckp_test_metric = {'AP@2': tf.Variable(0.), 'AP@3': tf.Variable(0.), 'AP@4': tf.Variable(0.), 'AP@8': tf.Variable(0.)}
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), loss=ckp_loss, test_metric=ckp_test_metric,
                                              optimizer=optimizer, model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             max_to_keep=1, directory=self.ckp_dir)
        if not do_not_restore:
            self.restore(force_restore)
        elif force_restore:
            self.print(f"[WARNING] Both 'do_not_restore' and 'force_restore' set. Checkpoint is not restored.")
        
    
    def restore(self, force=False):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            self.print(f'Model {self.model.name} restored from checkpoint at step {self.checkpoint.step.numpy()}.')
        elif force:
            raise FileNotFoundError(f"Cannot find checkpoint for model '{self.name}' but 'force_restore' is set.")
            

            
    @property
    def model(self):
        return self.checkpoint.model
    
    
    def stop_training(self):
        self.stop = True
        self.print(f"Stopping training...")
    
    
    def fit(self, train_data, epochs=100, evaluate_every="epoch", validation_data=None, 
            tensorboard=True, initial_epoch=0, save_best_only=True, track="loss",
            accumulating_gradients=False, test_data=None, test_every=10):
        
        self.print(f"Starting training...")
        self.print(f"Setting 'accumulating_gradients' to {accumulating_gradients}.")
        
        track = track.lower()
        if track not in ["loss", "mask_loss", "features_loss", "end_loss"]:
            raise ValueError(f"Cannot track {track}.")
        
        if tensorboard:
            writer_train = tf.summary.create_file_writer(os.path.join(self.log_dir, f'train'))
        for l in self.loss_accumulator['train']:
            self.loss_accumulator['train'][l].reset_states()
        stateful_metrics = list(self.loss_accumulator['train'].keys())
        
        tracked_loss = {'name': track, 'value': tf.constant(np.inf)}
        
        if validation_data is not None:
            if tensorboard:
                writer_val = tf.summary.create_file_writer(os.path.join(self.log_dir, f'val'))
            for l in self.loss_accumulator['val']:
                self.loss_accumulator['val'][l].reset_states()
            self.test_metric_accumulutor = {'AP@2': 0., 'AP@3': 0., 'AP@4': 0., 'AP@8': 0.}
            stateful_metrics.extend(list(self.loss_accumulator['val'].keys()))
        
        global_step = tf.cast(self.checkpoint.step, tf.int64)
        steps_per_epoch = self.config['DATA_N']//self.config['BATCH_SIZE']
        
        if validation_data is not None:
            if evaluate_every == "step":
                evaluate_every = 1
            elif evaluate_every == "epoch":
                evaluate_every = steps_per_epoch
            else:
                if not isinstance(evaluate_every, int):
                    raise ValueError(f'Wrong "evaluate_every": {evaluate_every}. Acceptable values are "step", "epoch" or int.')
                else:
                    evaluate_every = min(evaluate_every, steps_per_epoch)
                    self.print(f"Validating validation dataset every {evaluate_every} steps.")
        
        self.stop = False
        for epoch in range(initial_epoch, epochs):
            self.print("\nEpoch {}/{}".format(epoch + 1, epochs))
            pb_i = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=stateful_metrics)
            
            for step, (x, y) in enumerate(train_data):
                if self.stop:
                    break
                if step == 0: # a new epoch is starting -> reset metrics
                    for l in self.loss_accumulator['train']:
                        self.loss_accumulator['train'][l].reset_states()

                global_step += 1
                self.train_step(x, y['mask'], y['features'], accumulating_gradients)
                self.checkpoint.step.assign_add(1)
                
                values = []
                for l in self.loss_accumulator['train']:
                    if 'end' in l:
                        if hasattr(self, 'end_loss'):
                            values.append((l, self.loss_accumulator['train'][l].result()))
                    else:
                        values.append((l, self.loss_accumulator['train'][l].result()))
                if tensorboard:
                    with writer_train.as_default():
                        for l in self.loss_accumulator['train']:
                            if 'end' in l:
                                if hasattr(self, 'end_loss'):
                                    tf.summary.scalar(l, self.loss_accumulator['train'][l].result(),
                                              step=global_step)
                            else:
                                tf.summary.scalar(l, self.loss_accumulator['train'][l].result(),
                                              step=global_step)
                    writer_train.flush()
                
                if validation_data is not None:
                    if step != 0 and ((step + 1) % evaluate_every) == 0:
                        for l in self.loss_accumulator['val']:
                            self.loss_accumulator['val'][l].reset_states()
                        
                        for x, y_mask, y_features in validation_data:
                            self.test_step(x, y_mask, y_features)                      
                        
                        for l in self.loss_accumulator['val']:
                            if 'end' in l:
                                if hasattr(self, 'end_loss'):
                                    values.append((l, self.loss_accumulator['val'][l].result()))
                            else:
                                values.append((l, self.loss_accumulator['val'][l].result()))
                        
                        if step == 0 and (epoch+1 % test_every) == 0: #beginning of a new epoch multiple of test every
                            if test_data is not None:
                                try:
                                    AP = APCalculator(test_data[0], test_data[1], self.checkpoint.model, test_data[-1])
                                    self.test_metric_accumulator['AP@2'] = AP.compute(dist_range = 2, K = test_data[2])[0]
                                    self.test_metric_accumulator['AP@3'] = AP.compute(dist_range = 3)[0]
                                    self.test_metric_accumulator['AP@4'] = AP.compute(dist_range = 4)[0]
                                    self.test_metric_accumulator['AP@8'] = AP.compute(dist_range = 8)[0]
                                except Exception as e:
                                    self.print('Error in test_metric', e)
                            
                                for m in self.test_metric_accumulutor:
                                    values.append((m, self.test_metric_accumulutor[m]))
                            
                        if tensorboard:
                            with writer_val.as_default():
                                for l in self.loss_accumulator['val']:
                                    if 'end' in l:
                                        if hasattr(self, 'end_loss'):
                                            tf.summary.scalar(l, self.loss_accumulator['val'][l].result(),
                                                      step=global_step)
                                    else:
                                        tf.summary.scalar(l, self.loss_accumulator['val'][l].result(),
                                                      step=global_step)
                                if test_data is not None:
                                    for m in self.test_metric_accumulutor:
                                        tf.summary.scalar(m, self.test_metric_accumulutor[m],
                                                      step=global_step)
                                writer_val.flush()
                                
                        tracked_loss['value'] = self.loss_accumulator['val']['val_' + tracked_loss['name']].result()
                else: # if validation is not available, track training
                    tracked_loss['value'] = self.loss_accumulator['train'][tracked_loss['name']].result()
                
                
                pb_i.add(1, values=values) #update bar
                                
                if step == steps_per_epoch - 1: #end of the epoch
                    if save_best_only:
                        if (tracked_loss['value'] >= self.checkpoint.loss[tracked_loss['name']]): # no improvement, skip saving
                            continue
                    
                    self.print('Saving model...')
                    for l in self.loss_accumulator['train']:
                        self.checkpoint.loss[l] = self.loss_accumulator['train'][l].result()
                    for l in self.loss_accumulator['val']:
                        self.checkpoint.loss[l] = self.loss_accumulator['val'][l].result()
                    for m in self.test_metric_accumulator:
                        self.checkpoint.test_metric[m].assign(self.test_metric_accumulutor[m])
                    self.checkpoint_manager.save()
    
    
    def check_gradient(self, gradient):
        self.print(gradient)
        is_nan = [tf.math.reduce_any(tf.math.is_nan(grad)) if grad is not None else False for grad in gradient]
        self.print(is_nan)
        if tf.math.reduce_any(is_nan):
            self.print('Gradient is NaN!')
            self.stop_training()
            raise
    
    
    def atomic_train_step(self, x, y_mask, y_feats):
        l1 = l2 = l3 = None
        gradient1 = gradient2 = gradient3 = None
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x, training=True)
            l1 = self.mask_loss(y_mask, y_pred['mask']) * self.loss_weights['mask']
            if hasattr(self, 'features_loss'):
                l2 = self.features_loss(y_feats, y_pred['features']) * self.loss_weights['features']
            if hasattr(self, 'end_loss'):
                l3 = self.end_loss(y_end, y_pred['end']) * self.loss_weights['end']
            
        gradient1 = tape.gradient(l1, self.model.trainable_variables)
        #self.check_gradient(gradient1)
        if hasattr(self, 'features_loss'):
            gradient2 = tape.gradient(l2, self.model.trainable_variables)
            #self.check_gradient(gradient2)
        if hasattr(self, 'end_loss'):
            gradient3 = tape.gradient(l3, self.model.trainable_variables)
            #self.check_gradient(gradient3)
        del tape
        return gradient1, gradient2, gradient3, l1, l2, l3
    
    
    def train_step(self, x, y_mask, y_feats, accumulating_gradients_batch=False, y_end=None):  
        B = len(x) #batch size
        if accumulating_gradients_batch:
            accumulating_gradients_batch = int(accumulating_gradients_batch)
            b = np.clip(self.config['BATCH_SIZE_ACC'], 1, B)                
        else:
            b = B
        N = np.ceil(B/b).astype('int')
        
        tot_samples1 = 0.
        tot_samples2 = 0.
        tot_samples3 = 0.
        accum_gradient1 = [tf.zeros_like(var) for var in self.model.trainable_variables]
        accum_gradient2 = [tf.zeros_like(var) for var in self.model.trainable_variables]
        accum_gradient3 = [tf.zeros_like(var) for var in self.model.trainable_variables]
        
        for i in range(N):
            x_mini = x[i*b:(i+1)*b]
            y_mask_mini = y_mask[i*b:(i+1)*b]
            y_feats_mini = y_feats[i*b:(i+1)*b]
            gradient1, gradient2, gradient3, l1, l2, l3 = self.atomic_train_step(x_mini,y_mask_mini,
                                                                             y_feats_mini)
            
            self.loss_accumulator['train']['mask_loss'](l1)
            if l2 is not None:
                self.loss_accumulator['train']['features_loss'](l2)
            if l3 is not None:
                self.loss_accumulator['train']['end_loss'](l3)
        
            tot_samples1 += tf.cast(len(l1), tf.float32)
            if l2 is not None:
                tot_samples2 += tf.cast(len(l2), tf.float32)
            if l3 is not None:
                tot_samples3 += tf.cast(len(l3), tf.float32)   
            accum_gradient1 = [(accum_grad+grad) if grad is not None else accum_grad for accum_grad, grad in zip(accum_gradient1, gradient1)]
            if gradient2 is not None:
                accum_gradient2 = [(accum_grad+grad) if grad is not None else accum_grad for accum_grad, grad in zip(accum_gradient2, gradient2)]
            if gradient3 is not None:
                accum_gradient3 = [(accum_grad+grad) if grad is not None else accum_grad for accum_grad, grad in zip(accum_gradient3, gradient3)]

        accum_gradient1 = [(accum_grad)/tot_samples1 for accum_grad in accum_gradient1]
        if tot_samples2:
            accum_gradient2 = [(accum_grad)/tot_samples2 for accum_grad in accum_gradient2]
        if tot_samples3:
            accum_gradient3 = [(accum_grad)/tot_samples3 for accum_grad in accum_gradient3]
        gradients = [g1+g2+g3 for g1,g2,g3 in zip(accum_gradient1, accum_gradient2, accum_gradient3)]
        
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss_accumulator['train']['loss'].reset_states()
        tot_l = self.loss_accumulator['train']['mask_loss'].result() + \
                self.loss_accumulator['train']['features_loss'].result() + \
                self.loss_accumulator['train']['end_loss'].result()
        self.loss_accumulator['train']['loss'](tot_l)
  

    def test_step(self, x, y_mask, y_feats):        
        y_pred = self.model(x, training=False)
            
        loss1 = self.mask_loss(y_mask, y_pred['mask']) * self.loss_weights['mask']
        self.loss_accumulator['val']['val_mask_loss'](loss1)
        
        if hasattr(self, 'features_loss'):
            loss2 = self.features_loss(y_feats, y_pred['features']) * self.loss_weights['features']
            self.loss_accumulator['val']['val_features_loss'](loss2)
        
        self.loss_accumulator['val']['val_loss'].reset_states()
        tot_l = self.loss_accumulator['val']['val_mask_loss'].result() + \
                self.loss_accumulator['val']['val_features_loss'].result() + \
                self.loss_accumulator['val']['val_end_loss'].result()
        self.loss_accumulator['val']['val_loss'](tot_l)
        
