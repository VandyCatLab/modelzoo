'''
UNTESTED!!
'''

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
sys.path.append('../imported_code/svcca')
import cca_core, pwcca

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
        # Flatten representation if needed
        if len(out.shape) != 2:
            out = Flatten()(out)
        temp_model = Model(inputs=inp, outputs=out)
        # Predict on x_predict, transpose for spearman
        print('Getting activations...')
        acts = temp_model.predict(x_predict)
        acts_list.append(acts)
    
    return acts_list

'''
Correlation analysis functions
'''
def do_rsa(acts1, acts2):
    rdm1 = get_rdm(acts1)
    rdm2 = get_rdm(acts2)
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
Helper functions for correlation analysis
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
    print('shape:', rep.shape)
    num_imgs = rep.shape[0]
    print('num_images =', num_imgs)
    return spearmanr(rep.T, rep.T)[0][0:num_imgs, 0:num_imgs]



    