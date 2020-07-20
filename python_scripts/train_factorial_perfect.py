'''
CURRENTLY IN THE 'IMPERFECT' STATE
'''

import sys
import numpy as np
import pickle
import os
import random
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dropout, GlobalAveragePooling2D,
                                    MaxPooling2D, Activation, Dense, Layer)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras import backend as K 
import analysis, datasets

# Function to make models
def init_all_cnn_c(seed: int):
    
    model = Sequential()
    # model.add(Dropout(0.2, input_shape=x_train.shape[1:])) #input shape from keras cifar10 example
    model.add(Conv2D(96, (3, 3), input_shape=(32, 32, 3), padding='same',
                        kernel_regularizer=l2(1e-5), kernel_initializer=he_normal(seed), 
                        bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(96, (3, 3), strides=2, padding='same', 
                        kernel_regularizer=l2(1e-5), bias_initializer='zeros', activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(192, (3, 3), strides=2, padding='same',kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Conv2D(192, (3, 3), padding='valid', kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='valid', kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(10, (1, 1), padding='valid', kernel_regularizer=l2(1e-5),
                        kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, clipnorm=500),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model

# Make a dummy function
def init_dummy(seed: int):

    model = Sequential()
    model.add(Conv2D(10, (3, 3), input_shape=(32, 32, 3), padding='same',
                        kernel_regularizer=l2(1e-5), kernel_initializer=he_normal(seed), 
                        bias_initializer='zeros', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, clipnorm=500),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model

# Scheduler callback
def scheduler(epoch, lr):
    if (epoch == 200 or epoch == 250 or epoch == 300):
        return lr * 0.1
    return lr

LR_Callback = LearningRateScheduler(scheduler)

# Trajectory callback
class Trajectory_Callback(Callback):
    '''
    Pre: Must define i, x_predict
    '''
    def on_epoch_end(self, epoch, logs=None):
        layer_arr = [7]
        global w
        global s
        print(w)
        if epoch in [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9,
                     49, 99, 149, 199, 249, 299, 349]:
            print('\n\nSnapshot weight', str(w), 'shuffle', str(s), 'at epoch', str(int(epoch)+1))
            acts = analysis.get_acts(self.model, layer_arr, x_predict, cocktail_blank=False)
            np.save('../outputs/representations/acts/factorial_perfect/w'+str(w)+'s'+str(s)+'e'+str(epoch)+'.npy', acts)
            print('\n')


# Cut off training if local minimum hit  
class Early_Abort_Callback(Callback):
    '''
    Pre: abort is set to False at the beginning of each training instance
    '''
    def on_epoch_end(self, epoch, logs=None):
        global abort
        if (epoch > 10 and logs.get('accuracy') <= 0.11 or
                epoch == 70 and logs.get('accuracy') < 1.0):
            abort = True
            print('Acc:', logs.get('accuracy'))
            self.model.stop_training = True

w = 0
s = 0
abort = True                        

def train(seed_type, weights, shuffles):
    seed_value = 0
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['PYTHONHASHSEED']=str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    if seed_type == 'weight':
        s = int(shuffles)
        models = []
        while(abort):
            for _w in weights:
                w = int(_w)
                print('** Shuffle Seed:', s)
                print('** Weight Seed:', w)
                K.clear_session()
                tf.random.set_seed(seed_value)

                trainData, testData = datasets.make_train_data(shuffle_seed=s)
                x_predict, y_predict = datasets.make_predict_data(testData)

                # Note: it's possible that this only walks the shuffle seed, not the weights
                print('Training throwaway...')
                throwaway = init_dummy(seed=w).fit(
                    trainData,
                    epochs=1,
                    validation_data=(testData
                                     .prefetch(tf.data.experimental.AUTOTUNE)
                                     .batch(128)))

                print('Now training the real one...')
                model = init_all_cnn_c(seed=w)

                # Set flag to true if converges to local min
                abort = False
                history = model.fit(
                    trainData,
                    epochs=350,
                    validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE)\
                                    .batch(128),
                    callbacks=[LR_Callback, Trajectory_Callback(), Early_Abort_Callback()])

                print('After fit abort:', abort)
                if not abort:
                    models.append(model)
                else:
                    # Start over if even one failure
                    models = []
                    s += 1
                    break
        # If success, save models
        index = 0
        for model in models:
            w = weights[index]
            model.save('../outputs/models/factorial_perfect/w'+w+'s'+str(s)+'.h5')
            index += 1
    else:
        w = int(weights)
        models = []
        while(abort):
            for _s in shuffles:
                s = int(_s)
                print('** Weight Seed:', w)
                print('** Shuffle Seed:', s)
                K.clear_session()
                tf.random.set_seed(seed_value)

                trainData, testData = datasets.make_train_data(shuffle_seed=s)
                x_predict, y_predict = datasets.make_predict_data(testData)

                # Note: it's possible that this only walks the shuffle seed, not the weights
                print('Training throwaway...')
                throwaway = init_dummy(seed=w).fit(
                    trainData,
                    epochs=1,
                    validation_data=(testData
                                     .prefetch(tf.data.experimental.AUTOTUNE)
                                     .batch(128)))

                print('Now training the real one...')
                model = init_all_cnn_c(seed=w)

                # Set flag to true if converges to local min
                abort = False
                history = model.fit(
                    trainData,
                    epochs=350,
                    validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE)\
                                    .batch(128),
                    callbacks=[LR_Callback, Trajectory_Callback(), Early_Abort_Callback()])

                print('After fit abort:', abort)
                if not abort:
                    models.append(model)
                else:
                    # Start over if even one failure
                    models = []
                    w += 1
                    break
        # If success, save models
        index = 0
        for model in models:
            s = shuffles[index]
            model.save('../outputs/models/factorial_perfect/w'+w+'s'+str(s)+'.h5')
            index += 1
        
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-t','--seed_type',required=True)
    ap.add_argument('-w','--weights',required=True)
    ap.add_argument('-s','--shuffles',required=True)
    args=vars(ap.parse_args())
    
    seed_type = args['seed_type']
    weights = args['weights']
    shuffles = args['shuffles']
    
    assert seed_type in ['weight', 'shuffle']
    
    train(seed_type, weights, shuffles)


