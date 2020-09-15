import sys
import numpy as np
import pickle
import os
import random
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import (Conv2D, Dropout, GlobalAveragePooling2D,
                                    MaxPooling2D, Activation, Dense, Layer)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras import backend as K 
import analysis, datasets

# Make GPU training deterministic
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
        if epoch in [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9,
                     49, 99, 149, 199, 249, 299, 349]:
            print('\n\nSnapshot weight', str(w), 'shuffle', str(s), 'at epoch', str(int(epoch)+1))
            acts = analysis.get_acts(self.model, layer_arr, x_predict, cocktail_blank=False)
            np.save('../outputs/representations/acts/ten_by_ten_3/w'+str(w)+'s'+str(s)+'e'+str(epoch)+'.npy', acts)
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
s = 0
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--starting_weight', required=True)
args = vars(ap.parse_args())
w = int(args['starting_weight']) 
count = 0
while count < 5:
    print('** Weight seed:', w)
    K.clear_session()
    # Set seed values
    # Setting seed_value to w should only affect the generation of new_weight_seed
    # There should not be any notable correlation between it and the shuffle seed,
    # as the shuffle seed overrides any system seeds
    seed_value = w
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    trainData, testData = datasets.make_train_data(shuffle_seed=s, augment=True)
    x_predict, y_predict = datasets.make_predict_data(testData)
    for r in range(1000):
        random.randint(-10000, 10000)
    new_weight_seed = random.randint(-10000, 10000)
    print('new_weight_seed=', new_weight_seed)
    print('Throwing away shuffle values...')

    print('Now training the real one...')
    model = init_all_cnn_c(seed=new_weight_seed)

    # Set flag to true if converges to local min
    abort = False
    history = model.fit(
        trainData,
        epochs=150,
        validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE)\
                        .batch(128),
        callbacks=[LR_Callback, Early_Abort_Callback()])
    # Move onto the next shuffle candidate
    
    if not abort:
        save_model('../outputs/models/augment/w'+str(w)+'s0.pb', model)
        count += 1
    w += 1    

