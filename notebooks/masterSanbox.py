# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../python_scripts/')
import os
import datasets, train, baseline, analysis
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Complete notebook to present data for IDVOR
# Compiled and fixed by Jason, base code by Sam
# %% [markdown]
# ## Baseline Variability
# We should quantify baseline variability before how measuring how much variability comes from randomization seed and dataset shuffling
# 
# First, we'll train an instance of All-CNN-C on CIFAR10. 

# %%
# Make GPU training deterministic
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(0)

# Set seeds
weightSeed = 43
shuffleSeed = 99

trainData, testData = datasets.make_train_data(shuffle_seed=shuffleSeed, augment=True)
x_predict, y_predict = datasets.make_predict_data(testData)

# %% [markdown]
# We preprocess the images the same way that the original authors do, global contrast normalization and whitened with ZCA. During training, we augment the data the same way: horizontal flipping and random translation (5 pixels in any direction). 

# %%
modelPath = 'masterOutput/baseline/w'+str(weightSeed)+'s'+str(shuffleSeed)+'.pb'

if os.path.exists(modelPath): # The model already exists, load it.
    print('Loading model')
    model = tf.keras.models.load_model(modelPath)
else: # Train new model
    print('Training new model')
    model = train.init_all_cnn_c(seed=weightSeed)

    # Set flag to true if converges to local min
    abort = False
    history = model.fit(
        x=trainData, 
        epochs=350, 
        validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE).batch(128), 
        callbacks=[train.LR_Callback, train.Early_Abort_Callback()])

    if not abort:
        tf.keras.models.save_model(model, modelPath)
    else:
        raise ValueException('Model hit local minimum!')


# %%
model.summary()

# %% [markdown]
# All-CNN-C is a relatively small and efficient network that handles CIFAR10 relatively well. It's a good test case for our purposes in this project and we are following Kriegeskorte on this one. Something that we improve over previous work is that we explicitly train with augmentation. Notably the augmentation only applies translation and flip. This might matter later. Another note here is that perhaps we should be repeating this analysis with multiple models as we do know models differ between instances.
# 
# We take the output from layer index 12 to do our RSAs, this is the output of the global average pooling layer. Our test set is the 1000 test images of CIFAR10. The transformations we use to measure baseline variability are flip, translation, zoom, and color shift. We'll look at each in turn.
# 
# Flip is relatively simple. Since there's only one variation, it's a simple point estimate.

# %%
modelOutputIndex = 12


# %%
reflectRSAPath = 'masterOutput/baseline/reflectRSA.npy'
if os.path.exists(reflectRSAPath):
    print('Loading reflect RSA results')
    reflectRSA = np.load(reflectRSAPath)
else:
    print('Doing reflect RSA')
    reflectRSA = baseline.transform_baseline('reflect', 
                                             model, modelOutputIndex,
                                             analysis.do_rsa,
                                             analysis.preprocess_rsa)
    np.save(reflectRSAPath, reflectRSA)

print(f'Reflect RSA value: {reflectRSA[0]}')