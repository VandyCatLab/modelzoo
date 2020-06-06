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


def load_images(img_dir, pixels, img_count=25000):
    imgs = os.listdir(img_dir)
    imgs.sort()
    img_arr = np.empty((img_count, pixels, pixels, 3, 1), dtype=np.uint8)
    
    # Load images
    for i, img_path in enumerate(imgs):
        if i >= img_count:
            break
        print('Loading image', i, 'of', img_count)
        img = load_img(img_dir+'/'+img_path, target_size=(pixels, pixels), interpolation='bicubic')
        img_arr[i, :, :, :, 0] = img_to_array(img).astype(np.uint8)
    
    return img_arr

def transform_images(img_dir, pixels, transform, img_count=0):

    imgs = os.listdir(img_dir)
    imgs.sort()
    if img_count == 0:
        img_count = len(imgs)

    if transform == 'reflect':
        versions = 2
        transformed_imgs = np.empty((img_count, pixels, pixels, 3, versions), dtype=np.uint8)

        for i, imgPath in enumerate(imgs):
            if i >= img_count:
                break

            print(' - Transforming image:', i, 'of', img_count)
            imgName = img_dir + '/' + imgPath
            img = load_img(imgName, target_size=(pixels, pixels), interpolation='bicubic')
            img = img_to_array(img).astype(np.uint8)
            img_reflect = np.flip(img, axis=1)
            transformed_imgs[i, :, :, :, 0] = img
            transformed_imgs[i, :, :, :, 1] = img_reflect
            
    elif transform == 'color':
        versions = 201
        transformed_imgs = np.empty((img_count, pixels, pixels, 3, versions), dtype=np.uint8)
        alphas = np.concatenate(([0], np.linspace(-0.1, 0.1, 200)), axis=0)

        for i, imgPath in enumerate(imgs):
            if i >= img_count:
                break

            print(' - Transforming image:', i, 'of', img_count)
            imgName = img_dir + '/' + imgPath
            img = load_img(imgName, target_size=(pixels, pixels), interpolation='bicubic')

            img_as_array = img_to_array(img).astype(np.uint8)
            img_reshaped = img_as_array.reshape(-1, 3)
            cov = np.cov(img_reshaped.T)
            values, vectors = np.linalg.eig(cov)

            # Create a bunch of recolored images and store them in img_arr
            print('Processing images...')
            for v in range(0, versions):
                print(v, 'of', versions)
                alpha = alphas[v]
                change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
                transformed = np.clip(img + change, a_min=0, a_max=255, out=None)
                transformed_imgs[i, :, :, :, v] = transformed

    elif transform == 'zoom':
        print('##### WORKING ON ZOOM #####')
        versions = int(pixels / 2)
        mag = 1 # magnification factor
        transformed_imgs = np.zeros((img_count, pixels, pixels, 3, versions), dtype=np.uint8)
        for i, imgPath in enumerate(imgs):
            if i >= img_count:
                break
            
            print(' - Transforming image:', i, 'of', img_count)
            imgName = img_dir + '/' + imgPath
            img = load_img(imgName, target_size=(pixels, pixels), interpolation='bicubic')

            for v in range(versions):
                x = mag * v
                curImg = img.crop((x, x, img.width - x, img.height - x))
                curImg = curImg.resize((pixels, pixels), resample=Image.BICUBIC)
                curImg = img_to_array(curImg).astype(np.uint8)
                transformed_imgs[i, :, :, :, v] = curImg
                
    elif transform == 'shift':
        print('##### WORKING ON SHIFT #####')
        versions = pixels * 2
        scoot = 2  # Make the scoot factor such that entire image is pushed off and fits versions
        transformed_imgs = np.zeros((img_count, pixels, pixels, 3, versions), dtype=np.uint8)
        # create the gray values we use to fill in space
        empty = np.zeros((pixels, pixels, 3)).astype(np.uint8)
        empty.fill(128)

        for i, imgPath in enumerate(imgs):
            if i >= img_count:
                break
            print(' - Transforming image:', i, 'of', img_count)
            imgName = img_dir + '/' + imgPath
            img = load_img(imgName, target_size=(pixels, pixels), interpolation='bicubic')

            img_as_array = img_to_array(img).astype(np.uint8)
            v = 0  # v corresponds to slot in transformed_imgs
            for v1 in range(int(versions / 4)):  # v1 corresponds to the amount of shift to make in each direction
                displace     = int(v1 * scoot)
                upScooted    = np.concatenate([img_as_array[displace:pixels, :, :], empty[0:displace, :, :]])
                downScooted  = np.concatenate([empty[0:displace, :, :], img_as_array[0:pixels - displace, :, :]])
                leftScooted  = np.concatenate([img_as_array[:, displace:pixels, :], empty[:, 0:displace, :]], axis=1)
                rightScooted = np.concatenate([empty[:, 0:displace, :], img_as_array[:, 0:pixels - displace, :]], axis=1)

                transformed_imgs[i, :, :, :, v]     = upScooted
                transformed_imgs[i, :, :, :, v + 1] = downScooted
                transformed_imgs[i, :, :, :, v + 2] = leftScooted
                transformed_imgs[i, :, :, :, v + 3] = rightScooted

                v += 4
    else: raise Exception('Invalid transform given')
    
    return transformed_imgs

def get_representations(img_arr, network, layer_arr):
    # Wrapper function for model prediction
    # Input shapes: img_arr [imgNum, pixels, pixels, channels, versions]
    # Import network
    if network == 'VGG16':
        fullMdl = VGG16()

    elif network == 'Xception':
        fullMdl = Xception()

    else: raise Exception('Invalid network given')

    print ('########')
    print('Getting representations from', network)
    
    versions = len(img_arr[0, 0, 0, 0])
    inp = fullMdl.input

    # List of representations for each layer
    representations = []

    for layer_id in layer_arr:
        print('Working on layer ' + str(layer_id))
        layer = fullMdl.layers[layer_id]
        print(layer)
        print('===')

        out = layer.output
        # if len(out.shape) != 2: out = Flatten(out)
        
        tmpMdl = Model(inputs=inp, outputs=out)

        # subgroup of representations for each layer
        rep = np.empty((img_arr.shape[0], out.shape[1], versions))

        for v in range(versions):
            #print(' - Processing version:', v, 'of', versions)
            # Don't wanna overload memory. Split into sections
            i = 0
            while (i * 5000 < len(img_arr)):
                rep[i*5000:(i+1)*5000, :, v] = tmpMdl.predict(img_arr[i*5000:(i+1)*5000, :, :, :, v], verbose=1)
                i += 1

        representations.append(rep)

    return representations

def get_variance_thresholds(pred_arr):
    
    versions = pred_arr[0].shape[2]
    total_thresholds = []

    # Looking for minimum number of neurons that account for 99% variance
    target = 0.99
    for layer in pred_arr:
        #TODO: include the layer number at some point in print statement
        print('###Getting thresholds###')
        thresholds = np.empty((versions), dtype=np.uint8)
        
        '''
        First, find the threshold for the un-transformed image and work down
        '''

        start = 0
        end = layer.shape[1] - 1
        # Transpose to format (neurons, datapoints)
        acts = layer[:, :, 0].T

        res = search_threshold(acts, start, end)

        # Sanity check
        if (res['ans'] == -1):
            raise Exception('Something is wrong with the threshold')
        
        print(' - Found', str(res['ans']), 'neurons produce', str(res['var']), 'of total variance for version #0')
        index = res['ans']
        thresholds[0] = index

        '''
        With 'start' as a starting position, sequential search for the threshold for each version
        '''

        for version in range(1, versions):
            # Get activation set and svd
            acts = layer[:, :, version].T
            s = np.linalg.svd(acts - np.mean(acts, axis=1, keepdims=True), full_matrices=False)[1]
            # Sanity check
            if np.sum(s[:index])/np.sum(s) < 0.99:
                print('***You are either running color or decreasing thresholds is wrong\n', str(index), 'neurons produced', str(np.sum(s[:index])/np.sum(s)))
                res = search_threshold(acts, index, end)
                index = res['ans']
            # Search for minimum over 0.99
            else:
                while (np.sum(s[:index])/np.sum(s) >= 0.99):
                    index -= 1
                index += 1

            thresholds[version] = index
            # Sanity check again
            print(' - Found', str(index), 'neurons produce', str(np.sum(s[:index])/np.sum(s)), 'of total variance for version #', str(version))

        total_thresholds.append(thresholds)

    return total_thresholds

# Helper Function for binary search            
def search_threshold(acts, start, end):
    
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
    
    return_dict['ans'] = ans
    return_dict['var'] = np.sum(s[:mid])/np.sum(s)

    return return_dict

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

    num_imgs = imgset.shape[0]
    dim = imgset.shape[1]
    correlations = []

    # Set model to output reps at layer 7
    inp = full_model.input
    layer = full_model.layers[layer_num]
    out = layer.output

    # Flatten if necessary and using RSA
    if len(out.shape) != 2 and correlate_func == RSA:
            out = Flatten()(out) 

    # Get reps for originals
    rep_orig = model.predict(imageset)
    # Set orig RDM if using RSA
    if transform == RSA:
        rep_orig = get_RDM(rep_orig)

    if transform == 'reflect':
        transformed_imgset = np.reflect(imgset)
        rep = model.predict(transformed_imgset, verbose=1)
        correlations.append(correlate_func(rep_orig, rep))

    elif transform == 'color':
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
        print('##### WORKING ON ZOOM #####')
        versions = dim / 2
        transformed_imgset = np.zeros((num_imgs, dim, dim, 3), dtype=np.uint8)
        for v in range(versions):
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
        print('##### WORKING ON SHIFT #####')
        versions = dim
        up_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        down_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        left_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        right_imgset = np.zeros((img_count, dim, dim, 3), dtype=np.uint8)
        # create the gray values we use to fill in space
        empty = np.zeros((dim, dim, 3)).astype(np.uint8)
        empty.fill(128)
        for v in range(versions):
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
def RSA(RDM1, rep2):
    RDM2 = get_RDM(rep2)
    assert RDM1.shape == RDM2.shape
    num_imgs = RDM1.shape[0]
    RDM1_flat = RDM1[np.triu_indices(n=num_imgs, k=1)]
    RDM2_flat = RDM2[np.triu_indices(n=num_imgs, k=1)]
    return pearsonr(RDM1_flat, RDM2_flat)[0]

def SVCCA(acts1, acts2):
    # Not saving thresholds

    threshold1 = find_threshold(acts1)
    threshold2 = find_threshold(acts2)
    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:threshold1]*np.eye(threshold1), V1[:threshold1])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:threshold2]*np.eye(threshold2), V2[:threshold2])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    return svcca_results['cca_coef1']

def find_threshold(acts):
    ...

    

def PWCCA(rep1, rep2):
    ... 

def get_RDM(rep):
    num_imgs = rep.shape[0]
    print('num_images =', num_imgs)
    return spearmanr(rep.T, rep.T)[0][0:num_imgs, 0:num_imgs]



