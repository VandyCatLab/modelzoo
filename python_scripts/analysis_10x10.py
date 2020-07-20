'''
CORRELATE TRAJECTORY UNTESTED
'''

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K
import sys, os
sys.path.append('../imported_code/svcca')
import cca_core, pwcca

def correlate(method: str,
              path_to_instances,
              weight_seeds, 
              shuffle_seeds,
              x_predict,
              tracker,
              consistency='exemplar',
              cocktail_blank=False):

    preprocess_func, corr_func = get_funcs(method)

    # Create matrix to store representations in
    all_acts = [[], [], [], [], [], [], [], [], [], []]
    for w in weight_seeds:
        for s in shuffle_seeds:
            # Not a real numpy file, just a naming mistake
            instance = load_model(os.path.join(path_to_instances, 'w'+str(w)+'s'+str(s)+'.npy'))
            acts = get_acts(instance, [7], x_predict, cocktail_blank)[0]
            all_acts[w].append(preprocess_func(acts, consistency))

    # Get the mean representation to compare everything against
    indices = np.argwhere(tracker == 2)
    avg = np.zeros(all_acts[0][0].shape)
    total = 0
    for index in indices:
        w = index[0]
        s = index[1]
        acts = all_acts[w][s]
        avg += acts
        total += 0
    
    avg /= total

    # Get the correlations
    correlations = np.zeros((10, 10))
    for w in range(correlations.shape[0]):
        for s in range(correlations.shape[1]):
            if tracker[w, s] == 2:
                correlations[w, s] = corr_func(all_acts[w][s], avg)
    
    print('Done!')
    return correlations

def get_funcs(method):
    assert method in ['RSA', 'SVCCA', 'PWCCA'], 'Invalid correlation method'
    if method == 'RSA':
        return preprocess_rsa, do_rsa
    elif method == 'SVCCA':
        return preprocess_svcca, do_svcca
    elif method == 'PWCCA':
        return preprocess_pwcca, do_pwcca

    
'''
Preprocessing functions
'''
def preprocess_rsa(acts, consistency):
    # Note: Hardcoded on 10 categories
    categories = 10
    if len(acts.shape) > 2:
        imgs, h, w, channels = acts.shape
        acts = np.reshape(acts, newshape=(imgs, h*w*channels))
    if consistency == 'centroid':
        centroid_acts = np.empty((categories, acts.shape[1]))
        # imgs/categories should be a clean divide
        imgs_per_cat = int(imgs/categories)
        for i in range(0, imgs, imgs_per_cat):
            centroid_acts[int(i/imgs_per_cat)] = np.mean(acts[i:i+imgs_per_cat], axis=0)
        acts = centroid_acts
    rdm = get_rdm(acts)
    return rdm

# TODO: merge with interpolate
def preprocess_svcca(acts, interpolate=False):
    if len(acts.shape) > 2:
        acts = np.mean(acts, axis=(1,2))
    # Transpose to get shape [neurons, datapoints]
    threshold = get_threshold(acts.T)
    # Mean subtract activations
    cacts = acts.T - np.mean(acts.T, axis=1, keepdims=True)
    # Perform SVD
    _, s, V = np.linalg.svd(cacts, full_matrices=False)

    svacts = np.dot(s[:threshold]*np.eye(threshold), V[:threshold])
    return svacts

# TODO: merge with interpolate
def preprocess_pwcca(acts, interpolate=False):
    if len(acts.shape) > 2:
        acts = np.mean(acts, axis=(1,2))
    
    return acts


'''
Correlation analysis functions
'''
def do_rsa_from_acts(acts1, acts2):
    '''
    Pre: acts must be shape (neurons, datapoints)
    '''
    rdm1 = get_rdm(acts1)
    rdm2 = get_rdm(acts2)
    return do_rsa(rdm1, rdm2)

def do_rsa(rdm1, rdm2):
    '''
    Pre: RDMs must be same shape
    '''
    assert rdm1.shape == rdm2.shape
    num_imgs = rdm1.shape[0]
    # Only use upper-triangular values
    rdm1_flat = rdm1[np.triu_indices(n=num_imgs, k=1)]
    rdm2_flat = rdm2[np.triu_indices(n=num_imgs, k=1)]
    # Return squared pearson coefficient
    return pearsonr(rdm1_flat, rdm2_flat)[0] ** 2    

def do_svcca(acts1, acts2):
    '''
    Pre: acts must be shape (neurons, datapoints) and preprocessed with SVD
    '''
    svcca_results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-10, verbose=False)
    return np.mean(svcca_results['cca_coef1'])

def do_pwcca(acts1, acts2):
    '''
    Pre: acts must be shape (neurons, datapoints)
    '''
    # acts1.shape cannot be bigger than acts2.shape for pwcca
    if acts1.shape <= acts2.shape:
        return np.mean(pwcca.compute_pwcca(acts1.T, acts2.T, epsilon=1e-10)[0])
    return np.mean(pwcca.compute_pwcca(acts2.T, acts1.T, epsilon=1e-10)[0])

'''
Helper functions
'''
def get_acts(model, layer_arr, x_predict, cocktail_blank):
    '''
    Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
    Post: Returns list of activations over x_predict for relevant layers in this particular model instance
    '''
    inp = model.input
    acts_list = []
    
    for layer in layer_arr:
        print('Layer', str(layer))
        out = model.layers[layer].output
        temp_model = Model(inputs=inp, outputs=out)
        # Predict on x_predict, transpose for spearman
        print('Getting activations...')
        acts = temp_model.predict(x_predict)
        if cocktail_blank:
            # subtracting the mean activation pattern across all images from each network unit
            acts -= np.mean(acts, axis=0)
        acts_list.append(acts)
    
    return acts_list

def get_threshold(acts):
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
    
    print('Found', ans, '/', acts.shape[0], 'neurons accounts for', np.sum(s[:ans])/np.sum(s), 'of variance')

    return ans

def get_rdm(acts):
    '''
    Pre: acts must be flattened
    '''
    print('shape:', acts.shape)
    num_imgs = acts.shape[0]
    print('num_images =', num_imgs)
    return spearmanr(acts.T, acts.T)[0][0:num_imgs, 0:num_imgs]

