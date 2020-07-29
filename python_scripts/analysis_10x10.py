'''
CORRELATE TRAJECTORY UNTESTED
'''

import numpy as np
import sys, os, pickle
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K
sys.path.append('../imported_code/svcca')
import cca_core, pwcca



def correlate(method: str,
              primary: str,
              path_to_instances: str,
              save_dir: str,
              x_predict,
              consistency='exemplar',
              cocktail_blank=False):
    '''
    Description:
        Correlates a 10x10 set of models as ordered by either weightsXshuffle or shuffleXweight.
        *** Note: instances must be in format 'w[weight_seed]s[shuffle_seed].h5'
        *** Note: intended only for All-CNN-C or comparable 9-layer architectures
        
    Arguments:
        method: correlation method, *choose from 'RSA', 'SVCCA' or 'PWCCA'
        primary: sorting heirarchy, determines how graph is blocked, *choose from 'weights' or 'shuffle'
        path_to_instances: absolute or relative path to instances as string
        x_predict: data used to extract representations from instances, as np array or dataset
        consistency: how categories are treated, choose from centroid or exemplar, *default exemplar
        cocktail_blank: whether or not to use cocktail blank normalization, *default True
    '''
    
    # Check that arguments are valid because you're a dumb dumb
    assert method in ['RSA', 'SVCCA', 'PWCCA']
    assert primary in ['weights', 'shuffle']
    assert consistency in ['exemplar', 'centroid']
    
    # Get necessary functions and model instances
    preprocess_func, corr_func = get_funcs(method)
    instances = os.listdir(path_to_instances)
    
    # Apply method-specific preprocessing to all acts, save to disk to avoid overloading memory 
    print('**** Load and Preprocess Acts ****')
    for instance in instances:
        
        # Skip any non-model files that may have snuck in
#         if '.h5' not in instance:
#             continue
        if '.h5' not in instance:
            continue
        
        # Get acts for this instance
        print(' *** Working on', instance, '***')
        full_path = os.path.join(path_to_instances, instance)
        K.clear_session()
        model = load_model(full_path)
        preprocessed_acts = []
        raw_acts = get_acts(model, range(9), x_predict, cocktail_blank)
        
        # Preprocess
        layer_num = 0
        for acts in raw_acts:
            print('  ** Preprocessing... **')
#             try:
#                 acts = preprocess_func(acts, consistency)
#             except:
#                 print('~~~~~ YO this model is a dud ~~~~~~')
#                 acts = [0]
#             finally:
#                 preprocessed_acts.append(acts)
            acts = preprocess_func(acts, consistency)
            preprocessed_acts.append(acts)
            
            layer_num += 1
            
        
        # Save to disk
        filename = instance[:-3]
        save_path = os.path.join(save_dir, filename+'.pickle')
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessed_acts, f)
    
    ################### Now do correlations ###################
    
    print('**** Done gathering representations, now correlations ****')
    num_networks = 100
    correlations = np.zeros((900, 900))
    
    # To simplify logic, instead of explicitly leaving out lower triangle comparisons, just pass them by when they come up

    # Loop through vertical increments (base)
    counter = 0
    for i in range(100):
        # Decode 'i' using the 'primary' setting
        outer = i // 10
        inner = i % 10
        
        w = outer if primary == 'weight' else inner
        s = outer if primary == 'shuffle' else inner
        
        base_acts_path = os.path.join(save_dir, 'w'+str(w)+'s'+str(s)+'.pickle')
        
        print('Base Path:', base_acts_path)
        
        with open(base_acts_path, 'rb') as f:
            base_acts = pickle.load(f)
        
        # Loop through horizontal increments (comparer)
        for j in range(num_networks):
            # Decode 'i' using the 'primary' setting
            outer1 = j // 10
            inner1 = j % 10

            w1 = outer1 if primary == 'weight' else inner1
            s1 = outer1 if primary == 'shuffle' else inner1
            
            compare_acts_path = os.path.join(save_dir, 'w'+str(w1)+'s'+str(s1)+'.pickle')
            with open(compare_acts_path, 'rb') as f:
                compare_acts = pickle.load(f)
                
            print('Compare Path:', compare_acts_path)
            
            # At this point we have a 9x9 layer-wise
            # i is in 100, j is in 100
            
            for base_layer in range(len(base_acts)):
                
                placement_row = base_layer * num_networks + i
                base_act = base_acts[base_layer]
                
                for compare_layer in range(len(compare_acts)):
                    
                    placement_col = compare_layer * num_networks + j
                    compare_act = compare_acts[compare_layer]
                    
                    if placement_row <= placement_col:
                        # Check for duds
#                         if base_act == [0] or compare_act == [0]:
#                             correlations[placement_row, placement_col] = 0
#                         else:
#                             correlations[placement_row, placement_col] = corr_func(base_act, compare_act)
                        try:
                            correlations[placement_row, placement_col] = corr_func(base_act, compare_act)
                            print(counter, '*** Successful ***')
                        except:
                            print(counter, '~~~ DUD ~~~')
                            correlations[placement_row, placement_col] = -1
                        counter += 1
                            
    # Fill in other side of graph with reflection
    correlations += correlations.T
    for i in range(correlations.shape[0]):
        correlations[i, i] /= 2

    print('Done!')
    return correlations


def test_correlate(method: str,
              primary: str,
              path_to_instances: str,
              save_path: str,
              x_predict,
              consistency='exemplar',
              cocktail_blank=False):
    '''
    Description:
        Correlates a 10x10 set of models as ordered by either weightsXshuffle or shuffleXweight.
        *** Note: instances must be in format 'w[weight_seed]s[shuffle_seed].h5'
        *** Note: intended only for All-CNN-C or comparable 9-layer architectures
        
    Arguments:
        method: correlation method, *choose from 'RSA', 'SVCCA' or 'PWCCA'
        primary: sorting heirarchy, determines how graph is blocked, *choose from 'weights' or 'shuffle'
        path_to_instances: absolute or relative path to instances as string
        x_predict: data used to extract representations from instances, as np array or dataset
        consistency: how categories are treated, choose from centroid or exemplar, *default exemplar
        cocktail_blank: whether or not to use cocktail blank normalization, *default True
    '''
    
#     # Check that arguments are valid because you're a dumb dumb
#     assert method in ['RSA', 'SVCCA', 'PWCCA']
#     assert primary in ['weights', 'shuffle']
#     assert consistency in ['exemplar', 'centroid']
    
#     # Get necessary functions and model instances
#     preprocess_func, corr_func = get_funcs(method)
#     instances = os.listdir(path_to_instances)
    
#     # Apply method-specific preprocessing to all acts, save to disk to avoid overloading memory 
#     print('**** Load and Preprocess Acts ****')
#     for instance in instances:
        
#         # Skip any non-model files that may have snuck in
#         if '.h5' not in instance:
#             continue
        
#         # Get acts for this instance
#         print(' *** Working on', instance, '***')
#         full_path = os.path.join(path_to_instances, instance)
#         K.clear_session()
#         model = load_model(full_path)
#         preprocessed_acts = []
#         raw_acts = get_acts(model, range(9), x_predict, cocktail_blank)
        
#         # Preprocess
#         layer_num = 0
#         for acts in acts_list:
#             print('  ** Preprocessing... **')
#             acts = preprocess_func(acts, consistency)
#             preprocessed_acts.append(acts)
#             layer_num += 1
        
#         # Save to disk
#         filename = instance[:-3]
#         save_path = os.path.join(save_path, filename, '.pickle')
#         with open(save_path, 'wb') as f:
#             pickle.dump(preprocessed_acts, f)
    
    ################### Now do correlations ###################
    
    print('**** Done gathering representations, now correlations ****')
    num_networks = 100
    correlations = np.zeros((900, 900))
    
    # To simplify logic, instead of explicitly leaving out lower triangle comparisons, just pass them by when they come up
    
    # Loop through vertical increments (base)
    for i in range(100):
        # Decode 'i' using the 'primary' setting
        outer = i // 10
        inner = i % 10
        
        w = outer if primary == 'weight' else inner
        s = outer if primary == 'shuffle' else inner
        
#         base_acts_path = os.path.join(save_path, 'w'+str(w)+'s'+str(s)+'.pickle')
#         with open(base_acts_path, 'rb') as f:
#             base_acts = pickle.load(f)
        base_acts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Loop through horizontal increments (comparer)
        for j in range(num_networks):
            # Decode 'i' using the 'primary' setting
            outer1 = j // 10
            inner1 = j % 10

            w1 = outer1 if primary == 'weight' else inner1
            s1 = outer1 if primary == 'shuffle' else inner1
            
#             compare_acts_path = os.path.join(save_path, 'w'+str(w1)+'s'+str(s1)+'.pickle')
#             with open(compare_acts_path, 'rb') as f:
#                 compare_acts = pickle.load(f)
            compare_acts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # At this point we have a 9x9 layer-wise
            # i is in 100, j is in 100
            
            for base_layer in range(len(base_acts)):
                
                placement_row = base_layer * num_networks + i
                base_act = base_acts[base_layer]
                
                for compare_layer in range(len(compare_acts)):
                    
                    placement_col = compare_layer * num_networks + j
                    compare_act = compare_acts[compare_layer]
                    
                    # TODO: Just testing to see if all places are hit
                    if placement_row <= placement_col:
                        correlations[placement_row, placement_col] += 1# = corr_func(base_act, compare_act)

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

