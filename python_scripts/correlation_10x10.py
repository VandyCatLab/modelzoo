import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from scipy.stats import pearsonr, spearmanr
import sys, os
import analysis, datasets, analysis_10x10# , baseline

'''
Load Data
'''
import datasets
_, testData = datasets.make_train_data(None)
x_predict, _ = datasets.make_predict_data(testData)

wm = [0, 1, 105, 106, 107, 108, 109, 110, 111, 112, 113]
correlations = analysis_10x10.correlate('RSA', 'weights', '../outputs/models/ten-by-ten2', '../notebooks/Correlations_10x10_dropout_full', x_predict, weight_mappings=wm)

np.save('../outputs/correlations/ten-by-ten-dropout-full-kendall.npy', correlations)


