'''
UNTESTED!!
'''

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K
import sys
sys.path.append('../imported_code/svcca')
import cca_core, pwcca

def correlate(correlation_function, path_to_instances, identifiers, x_predict):
    '''
    Pre: ***HARDCODED*** 10 instances at specified path with 9 layers each, 1000 images
    Post: returns 90x90 correlation matrix using RSA, SVCCA or PWCCA
    '''
    # Check ahead of time for typos
    assert correlation_function in [do_rsa, do_svcca, do_pwcca]
    print('**** Load and Preprocess Acts ****')
    # Load up all acts into layer * instance grid
    all_acts = [[], [], [], [], [], [], [], [], []]
    for i in identifiers:
        print('*** Working on model', str(i), '***')
        K.clear_session()
        model = load_model(path_to_instances + str(i) + '.h5')
        print('** First, get acts')
        acts_list = get_acts(model, range(9), x_predict)
        print('** Second, reorder to fit graph scheme')
        # Loop through layers
        layer_num = 0
        for acts in acts_list:
            # Flatten by averaging if conv layer
            if len(acts.shape) > 2:
                if correlation_function == do_rsa:
                    imgs, h, w, channels = acts.shape
                    acts = np.reshape(acts, newshape=(imgs, h*w*channels))
                else:
                    acts = np.mean(acts, axis=(1,2))             
            all_acts[layer_num].append(acts.T)
            layer_num += 1
    
    # Now do correlations
    print('**** Done gathering RDMs, now correlations ****')
    num_networks = 10
    correlations = np.zeros((90, 90))
    # Run SVCCA
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1] - i):
            print('Correlation', str(i), ',', str(j))
            # Decode into the org scheme we want (i.e. layer * instance)
            layer_i = i // num_networks
            network_i = i % num_networks
            layer_j = j // num_networks
            network_j = j % num_networks
            acts1 = all_acts[layer_i][network_i]
            acts2 = all_acts[layer_j][network_j]
            
            correlations[i, j] = correlation_function(acts1, acts2)
    
    # OK this part might be a bit sketch. Basically, grab all the non-diag values
    # into a second matrix, then transpose to flip, then add it to the orig matrix.
    # This makes it so we don't need to compute every value twice
    num_imgs = 90
    correlations_lower = correlations[np.triu_indices(n=num_imgs, k=1)].T
    correlations += correlations_lower

    print('Done!')
    return correlations

'''
Correlation analysis functions
'''
def do_rsa(acts1, acts2):
    rdm1 = get_rdm(acts1)
    rdm2 = get_rdm(acts2)
    assert rdm1.shape == rdm2.shape
    num_imgs = rdm1.shape[0]
    # Only use upper-triangular values
    rdm1_flat = rdm1[np.triu_indices(n=num_imgs, k=1)]
    rdm2_flat = rdm2[np.triu_indices(n=num_imgs, k=1)]
    return pearsonr(rdm1_flat, rdm2_flat)[0]    

def do_svcca(acts1, acts2):
    # Transpose to get shape [neurons, datapoints]
    threshold1 = get_threshold(acts1.T)
    threshold2 = get_threshold(acts2.T)
    # Mean subtract activations
    cacts1 = acts1.T - np.mean(acts1.T, axis=1, keepdims=True)
    cacts2 = acts2.T - np.mean(acts2.T, axis=1, keepdims=True)
    # Perform SVD
    _, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    _, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

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
def get_acts(model, layer_arr, x_predict):
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
    
    print('Found', ans, '/', end, 'neurons accounts for', np.sum(s[:ans])/np.sum(s), 'of variance')

    return ans

def get_rdm(acts):
    print('shape:', rep.shape)
    num_imgs = rep.shape[0]
    print('num_images =', num_imgs)
    return spearmanr(rep.T, rep.T)[0][0:num_imgs, 0:num_imgs]