import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.datasets import cifar10

# Get training information
(x_trainRaw, y_trainRaw), (x_testRaw, y_testRaw) = cifar10.load_data()
mean = np.mean(x_trainRaw)
sd = np.std(x_trainRaw)


def make_train_data(shuffle_seed=None, set_seed=False, augment=False):
    """
    Apply ZCA Whitening and Global Contrast Normalization to CIFAR10 dataset
    """
    # Set seed values only if calling program hasn't already, otherwise it will override
    if set_seed:
        seed_value = 0
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    print("Making train data...")
    print("GCN...")
    # Apply global contrast normalization
    x_train = (x_trainRaw - mean) / sd
    x_test = (x_testRaw - mean) / sd
    print("ZCA...")
    # Do ZCA whitening
    x_flat = x_train.reshape(x_train.shape[0], -1)

    vec, val, _ = np.linalg.svd(np.cov(x_flat, rowvar=False))
    prinComps = np.dot(vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T))

    x_train = np.dot(x_flat, prinComps).reshape(x_train.shape)
    testFlat = x_test.reshape(x_test.shape[0], -1)
    x_test = np.dot(testFlat, prinComps).reshape(x_test.shape)

    # Convert to one hot vector
    y_train = tf.keras.utils.to_categorical(y_trainRaw, 10)
    y_test = tf.keras.utils.to_categorical(y_testRaw, 10)

    trainData = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # TODO: once we know that augmenting works, do the 10x10
    if augment:
        trainData = (
            trainData.prefetch(tf.data.experimental.AUTOTUNE)
            .shuffle(x_train.shape[0], seed=shuffle_seed)
            .map(augmentData)
            .batch(128)
        )
    else:
        trainData = (
            trainData.prefetch(tf.data.experimental.AUTOTUNE)
            .shuffle(x_train.shape[0], seed=shuffle_seed)
            .batch(128)
        )

    print("Done!")
    return trainData, testData


def augmentData(image, label):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    x = tf.random.uniform((), minval=-5, maxval=5, dtype=tf.dtypes.int64)
    y = tf.random.uniform((), minval=-5, maxval=5, dtype=tf.dtypes.int64)
    image = tfa.image.translate(images=image, translations=[x, y])

    return image, label


def preprocess(imgset):
    """
    Intended for use with baseline only: take a regular image set and apply ZCA whitening and GCN
    Note that a lot of code is copied over from make_train_data because we need info from x_train
    to do consistent preprocessing on the transformed images
    """
    print("Making train data...")

    print("GCN...")
    # Apply global contrast normalization
    x_train = (x_trainRaw - mean) / sd
    imgset = (imgset - mean) / sd
    print("ZCA...")
    # Do ZCA whitening
    x_flat = x_train.reshape(x_train.shape[0], -1)

    vec, val, _ = np.linalg.svd(np.cov(x_flat, rowvar=False))
    prinComps = np.dot(vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T))

    testFlat = imgset.reshape(imgset.shape[0], -1)
    imgset = np.dot(testFlat, prinComps).reshape(imgset.shape)

    return imgset


def make_predict_data(dataset, dtype=None):
    """
    Curate prediction set with 1000 images, 100 images for
    all 10 categories of CIFAR10
    """
    print("Making test data...")
    counts = [0] * 10
    x_predict = np.empty((1000, 32, 32, 3))
    if dtype is not None:
        x_predict = x_predict.astype(dtype)
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

    print("Done!")
    return x_predict, y_predict
