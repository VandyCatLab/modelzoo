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

'''
High level, high specificity functions
'''
def RSA_specific(path_to_instances, identifiers, x_predict):
    '''
    Pre: ***HARDCODED*** 10 instances at specified path with 9 layers each, 1000 images
    Post: returns 90x90 correlation matrix using RSA
    '''
    # Create container for RDMs
    RDM_list = np.empty((90, 1000, 1000)) # shape = #layers * #instances, #images, #images

    # Loop through network instances to get RDMs
    print('**** Gather RDMs ****')
    instance_num = 0
    for i in identifiers:
        print('*** Working on model', str(i), '***')
        K.clear_session()
        model = load_model(path_to_instances + str(i) + '.h5')
        print('** First, get acts')
        acts_list = get_acts(model, range(9), x_predict)
        print('** Second, get RDMs')
        # Loop through layers
        layer_num = 0
        for acts in acts_list:
            # Flatten if necessary
            # TODO: Haven't tested this method of reshaping yet, look here first if things break
            if len(acts.shape) > 2:
                imgs, h, w, channels = acts1.shape
                acts = np.reshape(acts, newshape=(imgs, h*w*channels))
            RDM_list[10 * layer_num + instance_num] = get_rdm(acts.T)
            layer_num += 1
        instance_num += 1

    # Create correlation matrix with 2nd-level RSA
    print('**** Done gathering RDMs, now correlations ****')
    correlations = np.empty((90, 90))
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            print('Correlation', str(i), ',', str(j))
            correlations[i, j] = do_rsa_second_level_only(RDM_list[i], RDM_list[j])
    
    print('Done!')
    return correlations

def SVCCA_specific(path_to_instances, identifiers, x_predict):
    '''
    Pre: ***HARDCODED*** 10 instances at specified path with 9 layers each, 1000 images
    Post: returns 90x90 correlation matrix using SVCCA
    '''
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
            acts = np.mean(acts, axis=(1,2)) if acts.shape > 2 else acts
            all_acts[layer_num].append(acts.T)
            layer_num += 1
    
    print('**** Done gathering RDMs, now correlations ****')
    num_networks = 10
    correlations = np.empty((90, 90))
    # Run SVCCA
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            print('Correlation', str(i), ',', str(j))
            
            layer_i = i // num_networks
            network_i = i % num_networks
            layer_j = j // num_networks
            network_j = j % num_networks
            acts1 = all_acts[layer_i][network_i]
            acts2 = all_acts[layer_j][network_j]
            
            # Do svcca
            correlations[i, j] = do_svcca(acts1, acts2)
    
    print('Done!')
    return correlations

def PWCCA_specific(path_to_instances, identifiers, x_predict):
    '''
    Pre: ***HARDCODED*** 10 instances at specified path with 9 layers each, 1000 images
    Post: returns 90x90 correlation matrix using SVCCA
    '''


'''
Correlation analysis functions
'''
def do_rsa(acts1, acts2):
    '''
    Wrapper for do_rsa_second_level_only (for easy passing to baseline)
    '''
    rdm1 = get_rdm(acts1)
    rdm2 = get_rdm(acts2)
    return do_rsa_second_level_only(rdm1, rdm2)
    
def do_rsa_second_level_only(rdm1, rdm2):
    assert rdm1.shape == rdm2.shape
    num_imgs = rdm1.shape[0]
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



    