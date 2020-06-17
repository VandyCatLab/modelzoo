import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten
from scipy.stats import pearsonr, spearmanr

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
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

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
#     y_train = tf.one_hot(y_train, 10)
#     y_test = tf.one_hot(y_test, 10)
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

        
def make_RDMs(model, layer_arr, x_predict):
    '''
    Create a list of layer_num RDMs, one RDM per layer
    Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
    Post: Returns list of RDMs over x_predict for relevant layers in this particular model instance
    '''
    inp = model.input
    num_images = len(x_predict)
    num_layers = len(layer_arr)

    RDMs = np.empty((num_layers, num_images, num_images))
    # Loop through layers
    layer_count = 0
    for layer_id in layer_arr:
        print('Layer', str(layer_id))
        out = model.layers[layer_id].output
        # Flatten representation if needed
        if len(out.shape) != 2:
            out = Flatten()(out)
        temp_model = Model(inputs=inp, outputs=out)
        # Predict on x_predict, transpose for spearman
        print('Getting representation...')
        representations = temp_model.predict(x_predict).T
        print(representations.shape)
        print('Getting RDM...')
        RDMs[layer_count] = spearmanr(representations, representations)[0][:num_images, :num_images]
        layer_count += 1
    
    return RDMs

def get_acts(model, layer_arr, x_predict):
    '''
    Same as above but for CCA
    Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
    Post: Returns list of activations over x_predict for relevant layers in this particular model instance
    '''
    inp = model.input
    acts_list = []
    
    for layer in layer_arr:
        print('Layer', str(layer))
        out = model.layers[layer].output
        # Flatten representation if needed
        if len(out.shape) != 2:
            out = Flatten()(out)
        temp_model = Model(inputs=inp, outputs=out)
        # Predict on x_predict, transpose for spearman
        print('Getting activations...')
        acts = temp_model.predict(x_predict)
        acts_list.append(acts)
    
    return acts_list
     