'''
SAFE
'''

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import analysis


def transform_baseline(imgset, transform, full_model, layer_num, correlate_func):
    '''
    Given an original imageset, a transform type, a model and a correlation type, return
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
    num_imgs = imgset.shape[0]
    dim = imgset.shape[1]
    correlations = []

    # Set model to output reps at layer 7
    inp = full_model.input
    layer = full_model.layers[layer_num]
    out = layer.output
    # Flatten if necessary and using RSA
    if len(out.shape) != 2 and correlate_func == do_rsa:
        out = Flatten()(out) 
    
    model = Model(inputs=inp, outputs=out)

    # Get reps for originals
    rep_orig = model.predict(imgset)

    if transform == 'reflect':
        print(' - Working on version 1 of 1')
        transformed_imgset = np.flip(imgset, axis=2)
        rep = model.predict(transformed_imgset, verbose=0)
        print(' - Now correlating...')
        correlations.append(correlate_func(rep_orig, rep))

    elif transform == 'color':
        versions = 201
        transformed_imgset = np.empty((num_imgs, dim, dim, 3), dtype=np.uint8)
        alphas = np.linspace(-0.1, 0.1, versions)

        for v in range(versions):
            # Generate transformed imageset
            print(' - Working on version', v, 'of', versions)
            alpha = alphas[v]
            for i in range(num_imgs):
                img = imgset[i, :, :, :]
                img_reshaped = img.reshape(-1, 3)
                cov = np.cov(img_reshaped.T)
                values, vectors = np.linalg.eig(cov)

                change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
                transformed_imgset[i, :, :, :] = np.clip(img + change, a_min=0, a_max=255, out=None)
            
            print(' - Now correlating...')
            rep = model.predict(transformed_imgset, verbose=0)
            correlations.append(correlate_func(rep_orig, rep))
            print('correlation:', correlations[v])
            
    elif transform == 'zoom':
        versions = dim // 2
        transformed_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        for v in range(versions):
            # Generate transformed imageset
            print(' - Working on version', v, 'of', versions)
            for i in range(num_imgs):
                img = Image.fromarray(imgset[i, :, :, :])
                new_img = img.crop((v, v, dim - v, dim - v))
                new_img = new_img.resize((dim, dim), resample=Image.BICUBIC)
                new_img = img_to_array(new_img).astype(np.uint8)
                transformed_imgset[i, :, :, :] = new_img
                
            print(' - Now correlating...')
            rep = model.predict(transformed_imgset, verbose=0)
            correlations.append(correlate_func(rep_orig, rep))
            print('correlation:', correlations[v])

    elif transform == 'shift':
        versions = dim
        up_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        down_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        left_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        right_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        # create the gray values we use to fill in space
        empty = np.zeros((dim, dim, 3)).astype(np.uint8)
        empty.fill(128)
        for v in range(versions):
            # Generate transformed imageset
            print(' - Working on version', v, 'of', versions)
            for i in range(num_imgs):
                img = imgset[i, :, :, :]
                up_imgset[i]    = np.concatenate([img[v:dim, :, :], empty[0:v, :, :]])
                down_imgset[i]  = np.concatenate([empty[0:v, :, :], img[0:dim - v, :, :]])
                left_imgset[i]  = np.concatenate([img[:, v:dim, :], empty[:, 0:v, :]], axis=1)
                right_imgset[i] = np.concatenate([empty[:, 0:v, :], img[:, 0:dim - v, :]], axis=1)

            # Get average of all 4 directions
            print(' - Now correlating...')
            corr_sum = 0.
            rep = model.predict(up_imgset, verbose=0)
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            rep = model.predict(down_imgset, verbose=0)
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            rep = model.predict(left_imgset, verbose=0)
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            rep = model.predict(right_imgset, verbose=0)
            corr_sum += correlate_func(rep_orig, rep)
            print('corr_sum:', corr_sum)
            correlations.append(corr_sum / 4)

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
        



