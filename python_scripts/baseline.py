import os
import sys
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.applications.xception import Xception, decode_predictions
import cca_core

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

    print('Transform:', transform, 'with', correlate_func.__name__)
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

    # Get reps for originals
    rep_orig = model.predict(imageset)
    # Set orig RDM if using RSA
    if correlate_func == do_rsa:
        rep_orig = get_rdm(rep_orig)

    if transform == 'reflect':
        print('Working on version 1 of 1')
        transformed_imgset = np.reflect(imgset)
        rep = model.predict(transformed_imgset, verbose=1)
        print('Now correlating...')
        correlations.append(correlate_func(rep_orig, rep))

    elif transform == 'color':Flatten
        versions = 200
        transformed_imgset = np.empty((num_imgs, dim, dim, 3), dtype=np.uint8)
        alphas = np.linspace(-0.1, 0.1, versions)

        for v in range(versions):
            # Generate transformed imageset
            print('Working on version', v, 'of', versions)
            alpha = alphas[v]
            for i in range(num_imgs):
                img = imgset[i, :, :, :]
                img_reshaped = img.reshape(-1, 3)
                cov = np.cov(img_reshaped.T)
                values, vectors = np.linalg.eig(cov)

                change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
                transformed_imgset[i, :, :, :] = np.clip(img + change, a_min=0, a_max=255, out=None)
            
            print('Now correlating...')
            rep = model.predict(transformed_imgset, verbose=1)
            correlations.append(correlate_func(rep_orig, rep))
            
    elif transform == 'zoom':
        versions = dim / 2
        transformed_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        for v in range(versions):
            # Generate transformed imageset
            print('Working on version', v, 'of', versions)
            for i in range(num_imgs):
                img = Image.fromarray(imgset[i, :, :, :])
                new_img = img.crop((v, v, dim - v, dim - v))
                new_img = newImg.resize((dim, dim), resample=Image.BICUBIC)
                new_img = img_to_array(new_img).astype(np.uint8)
                transformed_imgset[i, :, :, :] = new_img
                
            print('Now correlating...')
            rep = model.predict(transformed_imgset, verbose=1)
            correlations.append(correlate_func(rep_orig, rep))

    elif transform == 'shift':
        versions = dim
        up_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        down_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        left_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        right_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        # create the gray values we use to fill in space
        empty = np.zeros((dim, dim, 3)).astype(np.uint8)
        empty.fill(128)
        for v in range(versions):
            # Generate transformed imageset
            print('Working on version', v, 'of', versions)
            for i in range(img_count):
                img = imgmset[i, :, :, :, 0]
                up_imgset[v]    = np.concatenate([img[v:pixels, :, :], empty[0:v, :, :]])
                down_imgset[v]  = np.concatenate([empty[0:v, :, :], img[0:pixels - v, :, :]])
                left_imgset[v]  = np.concatenate([img[:, v:pixels, :], empty[:, 0:v, :]], axis=1)
                right_imgset[v] = np.concatenate([empty[:, 0:v, :], img[:, 0:pixels - v, :]], axis=1)

            # Get average of all 4 directions
            print('Now correlating...')
            corr_sum = 0.
            rep = model.predict(up_imgset, verbose=1)
            corr_sum += correlate_func(rep_orig, rep)
            rep = model.predict(down_imgset, verbose=1)
            corr_sum += correlate_func(rep_orig, rep)
            rep = model.predict(left_imgset, verbose=1)
            corr_sum += correlate_func(rep_orig, rep)
            rep = model.predict(right_imgset, verbose=1)
            corr_sum += correlate_func(rep_orig, rep)
            correlations.append(corr_sum / 4)

    return correlations

'''
Correlations functions
'''
def do_rsa(rdm1, rep2):
    rdm2 = get_rdm(rep2)
    assert rdm1.shape == rdm2.shape
    num_imgs = rdm1.shape[0]
    rdm1_flat = rdm1[np.triu_indices(n=num_imgs, k=1)]
    rdm2_flat = rdm2[np.triu_indices(n=num_imgs, k=1)]
    return pearsonr(rdm1_flat, rdm2_flat)[0]

def do_svcca(acts1, acts2):
    # Not saving thresholds
    # Transpose to get shape [neurons, datapoints]
    threshold1 = find_threshold(acts1.T)
    threshold2 = find_threshold(acts2.T)
    # Mean subtract activations
    cacts1 = acts1.T - np.mean(acts1.T, axis=1, keepdims=True)
    cacts2 = acts2.T - np.mean(acts2.T, axis=1, keepdims=True)
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:threshold1]*np.eye(threshold1), V1[:threshold1])
    svacts2 = np.dot(s2[:threshold2]*np.eye(threshold2), V2[:threshold2])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    return svcca_results['cca_coef1']

def do_pwcca(acts1, acts2):
    # Just a wrapper to make passing easier
    return pwcca.compute_pwcca(acts1, acts2)[0]

'''
Helper functions
'''
def find_threshold(acts):
    start = 0
    end = acts.shape[0]
    return_dict = {}
    ans = -1
    while start <= end:
        mid = (start + end) // 2
        # Move to right side if target is 
        # greater. 
        s = np.linalg.svd(acts - np.mean(acts, axis=1, keepdims=True), full_matrices=False)[1]
        # Note: normally comparing floating points is a bad bad but the precision we need is low enough
        if (np.sum(s[:mid])/np.sum(s) <= 0.99): 
            start = mid + 1;
        # Move left side. 
        else: 
            ans = mid; 
            end = mid - 1;
    
    print('Found', ans, '/', end, 'neurons accounts for', np.sum(s[:ans])/np.sum(s), 'of variance')

    return ans

def get_rdm(rep):
    num_imgs = rep.shape[0]
    print('num_images =', num_imgs)
    return spearmanr(rep.T, rep.T)[0][0:num_imgs, 0:num_imgs]



