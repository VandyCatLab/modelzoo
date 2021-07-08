"""
SAFE
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import cifar10
import analysis, datasets


def transform_baseline(
    transform, full_model, layer_num, correlate_func, preprocess_func
):
    """
    Given a transform type, a model and a correlation type, return
    a list of correlations of RDM_x with RDM_0, x being some transform depth and
    0 being the RDM for the untransformed imageset

    imageset (must be square): np array shape (num_imgs, dim, dim, channels)

    Transform information:
        - reflect: flip across y axis
        - color: generate 200 recolored (not including original) versions of varying severity
        - zoom: zoom in towards center by clipping off 1 pixel from each side
        - shift: pixel-by-pixel translation, fill with gray, each "depth" is mean of up-down-left-right shifting
    """

    print("### Transform:", transform, "with", correlate_func.__name__)
    print("Generating dataset")
    _, testData = datasets.make_train_data(shuffle_seed=0)
    imgset, _ = datasets.make_predict_data(testData)
    # print('orig imgset:', imgset)

    # Dataset information
    num_imgs = imgset.shape[0]
    dim = imgset.shape[1]
    correlations = []

    # Set model to output reps at selected layer
    inp = full_model.input
    layer = full_model.layers[layer_num]
    out = layer.output
    # Flatten if necessary and using RSA
    if len(out.shape) != 2 and correlate_func == analysis.do_rsa:
        out = Flatten()(out)

    model = Model(inputs=inp, outputs=out)

    # Get reps for originals
    rep_orig = model.predict(imgset)
    rep_orig = preprocess_func(rep_orig)

    # Reflect and shift don't require remaking the dataset
    if transform == "reflect":
        print(" - Working on version 1 of 1")
        transformed_imgset = np.flip(imgset, axis=2)
        rep = model.predict(transformed_imgset, verbose=0)
        rep = preprocess_func(rep)
        print(" - Now correlating...")
        correlations.append(correlate_func(rep_orig, rep))

    elif transform == "shift":
        versions = dim

        for v in range(versions):
            # Generate transformed imageset
            # print(' - Working on version', v, 'of', versions)
            transImg = tfa.image.translate(imgset, [v, 0])  # Right
            transImg = tf.concat(
                (transImg, tfa.image.translate(imgset, [-v, 0])), axis=0
            )  # Left
            transImg = tf.concat(
                (transImg, tfa.image.translate(imgset, [0, v])), axis=0
            )  # Down
            transImg = tf.concat(
                (transImg, tfa.image.translate(imgset, [0, -v])), axis=0
            )  # Up

            # plt.imshow(up_imgset[i])
            # plt.show()
            # plt.imshow(imgset[i])
            # plt.show()

            # Get average of all 4 directions
            # print(' - Now correlating...')
            reps = model.predict(transImg, verbose=0)
            # Split back out
            reps = tf.split(reps, 4)
            reps = [preprocess_func(np.array(rep)) for rep in reps]

            cors = [
                correlate_func(rep_orig, rep)
                if not np.ptp(rep) == 0
                else np.nan
                for rep in reps
            ]

            # print('corr_sum:', corr_sum)
            correlations.append(tmpCor)

    # Color and zoom will need new datasets every version because of ZCA
    elif transform == "color":
        versions = 51
        alphas = np.linspace(-10, 10, versions)

        print("Do PCA on raw training set to get eigenvalues and -vectors")
        # x_train = datasets.x_trainRaw.reshape(-1, 3)
        x_train = datasets.x_trainRaw / 255.0
        x_train = x_train.reshape(-1, 3)
        x_trainCentre = x_train - np.mean(x_train, axis=0)
        cov = np.cov(x_trainCentre.T)
        values, vectors = np.linalg.eigh(cov)

        # Get new test set
        x_test = datasets.preprocess(datasets.x_testRaw)
        y_test = tf.keras.utils.to_categorical(datasets.y_testRaw, 10)
        testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        imgset, _ = datasets.make_predict_data(testData)

        # Get reps for originals
        rep_orig = model.predict(imgset)
        rep_orig = preprocess_func(rep_orig)

        for v in range(versions):
            # Generate transformed imageset
            print(" - Working on version", v, "of", versions)

            # Start with a raw test set again
            transImg = datasets.x_testRaw / 255.0

            # Add multiple of shift
            alpha = alphas[v]
            change = np.dot(vectors, values * alpha)
            transImg[:, :, :, 0] += change[0]
            transImg[:, :, :, 1] += change[1]
            transImg[:, :, :, 2] += change[2]
            transImg = np.clip(transImg, a_min=0, a_max=1)

            # Unscale
            transImg *= 255

            transImg = datasets.preprocess(transImg)
            transData = tf.data.Dataset.from_tensor_slices((transImg, y_test))
            transImgset, _ = datasets.make_predict_data(transData)

            print(" - Now correlating...")
            rep = model.predict(transImgset, verbose=0)
            rep = preprocess_func(rep)
            correlations.append(correlate_func(rep_orig, rep))
            print("correlation:", correlations[v])

    elif transform == "zoom":
        versions = dim // 2
        # Create dataset
        x_test = datasets.preprocess(datasets.x_testRaw)
        y_test = tf.keras.utils.to_categorical(datasets.y_testRaw, 10)
        testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        imgset, _ = datasets.make_predict_data(testData)

        # Get reps for originals
        rep_orig = model.predict(imgset)
        rep_orig = preprocess_func(rep_orig)

        for v in range(versions):
            # Generate transformed imageset
            print(" - Working on version", v, "of", versions)
            transformed_imgset = imgset[:, v : dim - v, v : dim - v, :]
            transformed_imgset = tf.image.resize(
                transformed_imgset, (dim, dim)
            )

            # print(transformed_imgset)
            # plt.imshow(transformed_imgset[0, :, :, :].astype(int))
            # plt.show()
            # print('transformed_imgset:', transformed_imgset)
            print(" - Now correlating...")
            rep = model.predict(transformed_imgset, verbose=0)
            rep = preprocess_func(rep)
            correlations.append(correlate_func(rep_orig, rep))
            print("correlation:", correlations[v])

    print("Done!\n")
    return correlations


"""
Sanity check/Visualization functions
"""


def visualize_transform(transform, depth, img_arr):
    if transform == "reflect":
        # Depth doesn't matter
        transformed = np.flip(img_arr, axis=1)
        plt.imshow(transformed)

    elif transform == "color":
        # Depth = alpha value * 1000 (ints -100 : 100)
        alpha = depth / 1000
        img_reshaped = img_arr.reshape(-1, 3)
        cov = np.cov(img_reshaped.T)
        values, vectors = np.linalg.eig(cov)
        change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
        new_img = np.round(img_arr + change)
        transformed = np.clip(new_img, a_min=0, a_max=255, out=None)
        plt.imshow(transformed)

    elif transform == "zoom":
        dim = img_arr.shape[0]
        v = depth
        img = Image.fromarray(img_arr)
        new_img = img.crop((v, v, dim - v, dim - v))
        new_img = new_img.resize((dim, dim), resample=Image.BICUBIC)
        transformed = img_to_array(new_img)
        plt.imshow(transformed)

    elif transform == "shift":
        dim = img_arr.shape[0]
        v = depth
        empty = np.zeros((dim, dim, 3))
        up_transformed = np.concatenate(
            [img_arr[v:dim, :, :], empty[0:v, :, :]]
        )
        down_transformed = np.concatenate(
            [empty[0:v, :, :], img_arr[0 : dim - v, :, :]]
        )
        left_transformed = np.concatenate(
            [img_arr[:, v:dim, :], empty[:, 0:v, :]], axis=1
        )
        right_transformed = np.concatenate(
            [empty[:, 0:v, :], img_arr[:, 0 : dim - v, :]], axis=1
        )

        _, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(up_transformed)
        axarr[0, 1].imshow(down_transformed)
        axarr[1, 0].imshow(left_transformed)
        axarr[1, 1].imshow(right_transformed)

