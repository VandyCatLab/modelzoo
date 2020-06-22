'''
SAFE
'''

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def make_train_data():
    # Set seed values
    seed_value= 0
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['PYTHONHASHSEED']=str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    print('Making train data...')
    # Load CIFAR10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Get mean, SD of training set
    mean = np.mean(x_train)
    sd = np.std(x_train)
    print('GCN...')
    # Apply global contrast normalization
    x_train = (x_train-mean)/sd
    x_test = (x_test-mean)/sd
    print('ZCA...')
    # Do ZCA whitening
    x_flat = x_train.reshape(x_train.shape[0], -1)

    vec, val, vecT = np.linalg.svd(np.cov(x_flat, rowvar=False))
    prinComps = np.dot(vec, np.dot(np.diag(1.0/np.sqrt(val+0.00001)), vec.T))

    x_train = np.dot(x_flat, prinComps).reshape(x_train.shape)
    testFlat = x_test.reshape(x_test.shape[0], -1)
    x_test = np.dot(testFlat, prinComps).reshape(x_test.shape)

    # Convert to one hot vector
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    trainData = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    trainData = trainData.prefetch(tf.data.experimental.AUTOTUNE)\
        .shuffle(x_train.shape[0])\
        .batch(128)
    
    print('Done!')
    return trainData, testData

def make_predict_data(dataset):
    print('Making test data...')
    counts = [0] * 10
    x_predict = np.empty((1000, 32, 32, 3))
    y_predict = np.empty((1000, 10))
    for data in dataset:
        index = np.argmax(data[1])
        cur_count = counts[index]
        if cur_count != 100:
            x_predict[100 * index + cur_count] = data[0].numpy()
            y_predict[100 * index + cur_count] = data[1][0].numpy()
            counts[index] += 1
        # Finish once all 10 categories are full
        if all(count == 100 for count in counts):
            break
    
    print('Done!')
    return x_predict, y_predict