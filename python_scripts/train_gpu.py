# Everything is the same as with_hash, though without the snapshots.
# Only difference is that I use the deterministic ops flag here.

import sys
import numpy as np
import pickle
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dropout, GlobalAveragePooling2D,
                                    MaxPooling2D, Activation, Dense, Layer)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras import backend as K 
sys.path.append('../../../python_scripts/')
import correlations

# Set seed values
seed_value= 0
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed_value)
np.random.seed(seed_value)

# Function to make models
def init_model(architecture: str, seed: int):
    
    if (architecture == 'all_cnn_c'):
        model = Sequential()
        # model.add(Dropout(0.2, input_shape=x_train.shape[1:])) #input shape from keras cifar10 example
        model.add(Conv2D(96, (3, 3), input_shape=(32, 32, 3), padding='same', bias_regularizer=l2(1e-5),
                         kernel_regularizer=l2(1e-5), kernel_initializer=he_normal(seed), 
                         bias_initializer='zeros', activation='relu'))
        model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        model.add(Conv2D(96, (3, 3), strides=2, padding='same', bias_regularizer=l2(1e-5),
                         kernel_regularizer=l2(1e-5), bias_initializer='zeros', activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        model.add(Conv2D(192, (3, 3), strides=2, padding='same',kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Conv2D(192, (3, 3), padding='valid', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        model.add(Conv2D(192, (1, 1), padding='valid', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5), 
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        model.add(Conv2D(10, (1, 1), padding='valid', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5), 
                         kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, clipnorm=500),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    else:
        raise

    return model

# Scheduler callback
def scheduler(epoch, lr):
    if (epoch == 200 or epoch == 250 or epoch == 300):
        return lr * 0.1
    return lr

lr_callback = LearningRateScheduler(scheduler)

# Trajectory callback
class Trajectory_Callback(Callback):
    '''
    Pre: Must define instance_num, layer_arr, x_predict
    '''
    def on_epoch_end(self, epoch, logs=None):
        if epoch in [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9,
                     49, 99, 149, 199, 249, 299, 349]:
            print('\n\nSnapshot instance', str(instance_num), 'at epoch', str(int(epoch)+1))
            acts = correlations.get_acts(self.model, layer_arr, x_predict)
            np.save('../outputs/representations/acts/Version_5/i'+str(instance_num)+'e'+str(epoch)+'.npy', acts)
            print('\n')

# Trajectory callback
class Weights_Callback(Callback):
    '''
    Pre: Must define instance_num, layer_arr, x_predict
    '''
    epoch = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    def on_train_batch_begin(self, batch, logs=None):
        if (self.epoch % 50 == 0 and batch % 390 == 0):
            self.model.save_weights('../outputs/weights/primary/Version_5/i'+str(instance_num)+'e'+str(self.epoch)+'b'+str(batch)+'.h5')            

# Setup to train on both GPUs concurrently
# This is kind of hacky, should really have just set up to user multi-GPU training but oh well
GPU_ID = sys.argv[1]
if GPU_ID == '0':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # Load predict datasets and whitened datasets
    trainData, testData = correlations.make_train_data()
    x_predict, y_predict = correlations.make_predict_data(testData)
    for i in range(5):
        print('Training All_CNN_C with seed', i)
        K.clear_session()
        tf.random.set_seed(seed_value)
        all_cnn_c = init_model('all_cnn_c', seed=i)

        instance_num = i
        layer_arr = [7]
        history = all_cnn_c.fit(
            trainData,
            epochs=350,
            validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE)\
                            .batch(128),
            callbacks=[lr_callback, Trajectory_Callback(), Weights_Callback()],
            shuffle=False)

        all_cnn_c.save('../outputs/models/primary/Version_5/instance_'+str(i)+'.h5')

elif GPU_ID == '1':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    # Load predict datasets and whitened datasets
    trainData, testData = correlations.make_train_data()
    x_predict, y_predict = correlations.make_predict_data(testData)
    for i in range(5, 10):
        print('Training All_CNN_C with seed', i)
        K.clear_session()
        tf.random.set_seed(seed_value)
        all_cnn_c = init_model('all_cnn_c', seed=i)

        instance_num = i
        layer_arr = [7]

        history = all_cnn_c.fit(
            trainData,
            epochs=350,
            validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE)\
                            .batch(128),
            callbacks=[lr_callback, Trajectory_Callback(), Weights_Callback()],
            shuffle=False)

        all_cnn_c.save('../outputs/models/primary/Version_5/instance_'+str(i)+'.h5')
