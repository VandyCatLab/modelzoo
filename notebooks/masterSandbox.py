import sys
sys.path.append('../python_scripts/')
import os
import datasets, train, baseline, analysis
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Make GPU training deterministic
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(0)

# Set seeds
weightSeed = 43
shuffleSeed = 99

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

modelOutputIndex = 12

colorRSAPath = 'masterOutput/baseline/colorRSAPath.npy'
if os.path.exists(colorRSAPath):
    print('Loading color RSA results')
    colorRSA = np.load(colorRSAPath)
else:
    print('Doing color RSA')
    colorRSA = baseline.transform_baseline('color', 
                                           model, modelOutputIndex,
                                           analysis.do_rsa,
                                           analysis.preprocess_rsa)
    np.save(colorRSAPath, colorRSA)


colorRSA
