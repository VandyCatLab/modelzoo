import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten
from scipy.stats import pearsonr, spearmanr

def make_test_data():
    '''
    Obtain data as described in paper, 1000 images, 100 per category
    Post: Returns curated dataset
    '''
    print('Making test data...')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_predict = np.empty((1000, 32, 32, 3))
    y_predict = np.empty((1000, 1))

    # Fill up 10 different categories with 100 images
    full = False
    # 10 different counters for the 10 categories
    counts = [0] * 10
    i = 0
    while not full:
        x = x_test[i]
        y = y_test[i] # also serves as index
        # Check if the category is full first, otherwise skip
        cur_count = counts[y[0]]
        if cur_count != 100:
            x_predict[100 * y[0] + cur_count] = x
            y_predict[100 * y[0] + cur_count] = y
            counts[y[0]] += 1
        # Quit when all categories are full
        full = all(count == 100 for count in counts)
        i += 1
    print('Done!')
    return x_predict, y_predict

        
def make_RDMs(model, layer_arr, x_predict):
    '''
    Create a list of layer_num RDMs, one RDM per layer
    Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
    Post: Returns list of RDMs over x_predict for relevant layers in this particular model instance
    '''
    inp = model.input
    num_images = len(x_predict)
    num_layers = len(layer_arr)

    RDMs = np.empty((num_layers, num_images, num_images))
    # Loop through layers
    layer_count = 0
    for layer_id in layer_arr:
        print('Layer', str(layer_id + 1))
        out = model.layers[layer_id].output
        # Flatten representation if needed
        if len(out.shape) != 2:
            out = Flatten()(out)
        temp_model = Model(inputs=inp, outputs=out)
        # Predict on x_predict, transpose for spearman
        print('Getting representation...')
        representations = temp_model.predict(x_predict).T
        print(representations.shape)
        print('Getting RDM...')
        RDMs[layer_count] = spearmanr(representations, representations)[0][:num_images, :num_images]
        layer_count += 1
    
    return RDMs

def get_acts(model, layer_arr, x_predict):
    '''
    Same as above but for CCA
    Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
    Post: Returns list of activations over x_predict for relevant layers in this particular model instance
    '''
    inp = model.input
    acts_list = []
    
    for layer in layer_arr:
        print('Layer', str(layer + 1))
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
     