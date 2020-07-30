'''
SAFE
'''

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import cifar10
import analysis, datasets



def transform_baseline(transform, full_model, layer_num, correlate_func, preprocess_func):
    '''
    Given a transform type, a model and a correlation type, return
    a list of correlations of RDM_x with RDM_0, x being some transform depth and
    0 being the RDM for the untransformed imageset

    imageset (must be square): np array shape (num_imgs, dim, dim, channels)

    Transform information:
        - reflect: flip across y axis
        - color: generate 200 recolored (not including original) versions of varying severity
        - zoom: zoom in towards center by clipping off 1 pixel from each side
        - shift: pixel-by-pixel translation, fill with gray, each "depth" is mean of up-down-left-right shifting
    '''

    print('### Transform:', transform, 'with', correlate_func.__name__)
    print('Generating dataset')
    _, testData  = datasets.make_train_data(shuffle_seed=0)
    imgset, _ = datasets.make_predict_data(testData)
    num_imgs = imgset.shape[0]
    dim = imgset.shape[1]
    correlations = []

    # Set model to output reps at layer 7
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
    if transform == 'reflect':
        print(' - Working on version 1 of 1')
        transformed_imgset = np.flip(imgset, axis=2)
        rep = model.predict(transformed_imgset, verbose=0)
        rep = preprocess_func(rep)
        print(' - Now correlating...')
        correlations.append(correlate_func(rep_orig, rep))

    elif transform == 'shift':
        versions = dim
        up_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        down_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        left_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        right_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        # create the gray values we use to fill in space
        empty = np.zeros((dim, dim, 3))
        print('Generating dataset')
        _, testData  = datasets.make_train_data(shuffle_seed=0)
        imgset, _ = datasets.make_predict_data(testData)
        for v in range(versions):
            # Generate transformed imageset
            print(' - Working on version', v, 'of', versions)
            for i in range(num_imgs):
                img = imgset[i, :, :, :]
                up_imgset[i]    = np.concatenate([img[v:dim, :, :], empty[0:v, :, :]])
                down_imgset[i]  = np.concatenate([empty[0:v, :, :], img[0:dim - v, :, :]])
                left_imgset[i]  = np.concatenate([img[:, v:dim, :], empty[:, 0:v, :]], axis=1)
                right_imgset[i] = np.concatenate([empty[:, 0:v, :], img[:, 0:dim - v, :]], axis=1)
             
            plt.imshow(up_imgset[i])
            plt.show()
            plt.imshow(imgset[i])
            plt.show()

            # Get average of all 4 directions
            print(' - Now correlating...')
            corr_sum = 0.
            rep = preprocess_func(model.predict(up_imgset, verbose=0))
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            rep = preprocess_func(model.predict(down_imgset, verbose=0))
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            rep = preprocess_func(model.predict(left_imgset, verbose=0))
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            rep = preprocess_func(model.predict(right_imgset, verbose=0))
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            correlations.append(corr_sum / 4)
    
    # Color and zoom will need new datasets every version because of ZCA
    elif transform == 'color':
        versions = 51
        alphas = np.linspace(-0.1, 0.1, versions)
        
        print("Getting non-ZCA/GCN'd data...")
        _, (x_test, y_test) = cifar10.load_data()
        testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        imgset, _ = datasets.make_predict_data(testData)
        imgset = imgset.astype(np.uint8)
                                               
        for v in range(versions):
            # Generate transformed imageset
            print(' - Working on version', v, 'of', versions)
            transformed_imgset = np.empty((num_imgs, dim, dim, 3), dtype=np.uint8)
            alpha = alphas[v]
            for i in range(num_imgs):
                img = imgset[i, :, :, :]
                img_reshaped = img.reshape(-1, 3)
                cov = np.cov(img_reshaped.T)
                values, vectors = np.linalg.eig(cov)

                change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
                transformed_imgset[i, :, :, :] = np.clip(img + change, a_min=0, a_max=255, out=None)

            transformed_imgset = datasets.preprocess(transformed_imgset)
            print(' - Now correlating...')
            rep = model.predict(transformed_imgset, verbose=0)
            rep = preprocess_func(rep)
            correlations.append(correlate_func(rep_orig, rep))
            print('correlation:', correlations[v])
            
    elif transform == 'zoom':
        versions = dim // 2
        
        print("Getting non-ZCA/GCN'd data...")
        _, (x_test, y_test) = cifar10.load_data()
        testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        imgset, _ = datasets.make_predict_data(testData)
        imgset = imgset.astype(np.uint8)
        print('imgset shape:', imgset.shape)
        
        for v in range(versions):
            # Generate transformed imageset
            print(' - Working on version', v, 'of', versions)
            transformed_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
            for i in range(num_imgs):
                img = Image.fromarray(imgset[i, :, :, :])
                new_img = img.crop((v, v, dim - v, dim - v))
                new_img = new_img.resize((dim, dim), resample=Image.BICUBIC)
                new_img = img_to_array(new_img).astype(np.uint8)
                transformed_imgset[i, :, :, :] = new_img
            
            transformed_imgset = datasets.preprocess(transformed_imgset)
            print(' - Now correlating...')
            rep = model.predict(transformed_imgset, verbose=0)
            rep = preprocess_func(rep)
            correlations.append(correlate_func(rep_orig, rep))
            print('correlation:', correlations[v])


    print('Done!\n')
    return correlations

"""
Sanity check/Visualization functions
"""
def visualize_transform(transform, depth, img_arr):
    if transform == 'reflect':
        # Depth doesn't matter
        transformed = np.flip(img_arr, axis=1)
        plt.imshow(transformed)
        
    elif transform == 'color':
        # Depth = alpha value * 1000 (ints -100 : 100)
        alpha = depth / 1000
        img_reshaped = img_arr.reshape(-1, 3)
        cov = np.cov(img_reshaped.T)
        values, vectors = np.linalg.eig(cov)
        change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
        new_img = np.round(img_arr + change)
        transformed = np.clip(new_img, a_min=0, a_max=255, out=None).astype('uint8')
        plt.imshow(transformed)
            
    elif transform == 'zoom':
        dim = img_arr.shape[0]
        v = depth
        img = Image.fromarray(img_arr)
        new_img = img.crop((v, v, dim - v, dim - v))
        new_img = new_img.resize((dim, dim), resample=Image.BICUBIC)
        transformed = img_to_array(new_img).astype(np.uint8)
        plt.imshow(transformed)
         
    elif transform == 'shift':
        dim = img_arr.shape[0]
        v = depth
        empty = np.zeros((dim, dim, 3)).astype(np.uint8)
        up_transformed    = np.concatenate([img_arr[v:dim, :, :], empty[0:v, :, :]])
        down_transformed  = np.concatenate([empty[0:v, :, :], img_arr[0:dim - v, :, :]])
        left_transformed  = np.concatenate([img_arr[:, v:dim, :], empty[:, 0:v, :]], axis=1)
        right_transformed = np.concatenate([empty[:, 0:v, :], img_arr[:, 0:dim - v, :]], axis=1)
        
        _, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(up_transformed)
        axarr[0,1].imshow(down_transformed)
        axarr[1,0].imshow(left_transformed)
        axarr[1,1].imshow(right_transformed)
        



