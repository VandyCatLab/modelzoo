'''
UNTESTED!!
'''

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy import interpolate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K
import sys
sys.path.append('../imported_code/svcca')
import cca_core, pwcca

def correlate(method, path_to_instances, identifiers, x_predict):
    '''
    Pre: ***HARDCODED*** 10 instances at specified path with 9 layers each, 1000 images
    Post: returns 90x90 correlation matrix using RSA, SVCCA or PWCCA
    '''
    # Get necessary functions
    preprocess_func, corr_func = get_funcs(method)
    
    print('**** Load and Preprocess Acts ****')
    # Load up all acts into layer * instance grid
    all_acts = [[], [], [], [], [], [], [], [], []]
    for i in identifiers:
        print('*** Working on model', str(i), '***')
        K.clear_session()
        model = load_model(path_to_instances + str(i) + '.h5')
        acts_list = get_acts(model, range(9), x_predict)
        # Loop through layers
        layer_num = 0
        for acts in acts_list:
            print('* Preprocessing...')
            acts = preprocess_func(acts)
            all_acts[layer_num].append(acts)
            layer_num += 1
    
        
    # Now do correlations
    print('**** Done gathering RDMs, now correlations ****')
    num_networks = 10
    correlations = np.zeros((90, 90))
    # Run SVCCA
    for i in range(correlations.shape[0]):
        # Only perform on lower triangle, avoid repeats
        # NOTE: Lower triangle b/c yields better separation with PWCCA
        for j in range(i + 1):
            print('Correlation', str(i), ',', str(j))
            # Decode into the org scheme we want (i.e. layer * instance)
            layer_i = i // num_networks
            network_i = i % num_networks
            layer_j = j // num_networks
            network_j = j % num_networks
            acts1 = all_acts[layer_i][network_i]
            acts2 = all_acts[layer_j][network_j]
            
            correlations[i, j] = corr_func(acts1, acts2)
    
    # Fill in other side of graph with reflection
    correlations += correlations.T
    for i in range(correlations.shape[0]):
        correlations[i, i] /= 2

    print('Done!')
    return correlations

def get_funcs(method):
    assert method in ['RSA', 'SVCCA', 'PWCCA'], 'Invalid correlation method'
    if method == 'RSA':
        return preprocess_rsa, do_rsa
    elif method == 'SVCCA':
        return preprocess_cca, do_svcca
    elif method == 'PWCCA':
        return preprocess_cca, do_pwcca

'''
Preprocessing functions
'''
def preprocess_rsa(acts):
    if len(acts.shape) > 2:
        imgs, h, w, channels = acts.shape
        acts = np.reshape(acts, newshape=(imgs, h*w*channels))
        
    rdm = get_rdm(acts.T)
    return rdm

def preprocess_cca(acts, use_interpolate=True):
    if len(acts.shape) > 2 and not use_interpolate:
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
    return pearsonr(rdm1_flat, rdm2_flat)[0]    

def do_svcca(acts1, acts2, use_interpolate=True):

    if (acts1.shape > 2 or acts2.shape > 2):
        if use_interpolate:
            if acts1.shape != acts2.shape:
                acts1, acts2 = interpolate_acts(acts1, acts2)
            #Flatten
            num_datapoints, h, w, channels = acts1.shape
            acts1 = acts1.reshape((num_datapoints*h*w, channels))
            num_datapoints, h, w, channels = acts2.shape
            acts2 = acts2.reshape((num_datapoints*h*w, channels))
            print(acts1.shape, acts2.shape)
        else:
            raise('Must preprocess acts if not using interpolate')

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
    return np.mean(svcca_results['cca_coef1'])

def do_pwcca(acts1, acts2, use_interpolate=True):
    if (acts1.shape > 2 or acts2.shape > 2):
        if use_interpolate:
            if acts1.shape != acts2.shape:
                acts1, acts2 = interpolate_acts(acts1, acts2)
            #Flatten
            num_datapoints, h, w, channels = acts1.shape
            acts1 = acts1.reshape((num_datapoints*h*w, channels))
            num_datapoints, h, w, channels = acts2.shape
            acts2 = acts2.reshape((num_datapoints*h*w, channels))
            print(acts1.shape, acts2.shape)
        else:
            raise('Must preprocess acts if not using interpolate')

    if acts1.shape <= acts2.shape:
        return np.mean(pwcca.compute_pwcca(acts1.T, acts2.T, epsilon=1e-10)[0])
    return np.mean(pwcca.compute_pwcca(acts2.T, acts1.T, epsilon=1e-10)[0])

def interpolate_acts(acts1, acts2):
    '''
    Largely stolen from svcca tutorial
    '''
    if acts1.shape[1] < acts.shape[2]:
        smaller = acts1
        larger = acts2
    else:
        smaller = acts2
        larger = acts1
    
    num_d, h, w, _ = larger.shape
    num_c = smaller.shape[-1]
    smaller_interp = np.zeros((num_d, h, w, num_c))

    for d in range(num_d):
        for c in range(num_c):
            # form interpolation function
            idxs1 = np.linspace(0, smaller.shape[1],
                                smaller.shape[1],
                                endpoint=False)
            idxs2 = np.linspace(0, smaller.shape[2],
                                smaller.shape[2],
                                endpoint=False)
            arr = smaller[d,:,:,c]
            f_interp = interpolate.interp2d(idxs1, idxs2, arr)
            
            # creater larger arr
            large_idxs1 = np.linspace(0, smaller.shape[1],
                                larger.shape[1],
                                endpoint=False)
            large_idxs2 = np.linspace(0, smaller.shape[2],
                                larger.shape[2],
                                endpoint=False)
            
            smaller_interp[d, :, :, c] = f_interp(large_idxs1, large_idxs2)

    print("new shape", smaller_interp.shape)

    return smaller_interp, larger


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
    
    print('Found', ans, '/', acts.shape[0], 'neurons accounts for', np.sum(s[:ans])/np.sum(s), 'of variance')

    return ans

def get_rdm(acts):
    print('shape:', acts.shape)
    num_imgs = acts.shape[0]
    print('num_images =', num_imgs)
    return spearmanr(acts.T, acts.T)[0][0:num_imgs, 0:num_imgs]

