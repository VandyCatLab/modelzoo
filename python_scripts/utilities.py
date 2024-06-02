import glob
from operator import index
import pandas as pd
import os
import itertools
import numpy as np
import scipy.stats as stats
import json
import glob
import tensorflow as tf
import gc




def hub_rep_completion(repDir, hubInfo):
    """
    Return a list of missing Tensorflow Hub representations in repDir given
    hubInfo.
    """
    # Get list of rep files
    repFiles = glob.glob(os.path.join(repDir, "*.npy"))
    repFiles = [name.replace(repDir + "/", "") for name in repFiles]
    with open(hubInfo, "r") as f:
        hubModels = json.loads(f.read())

    # Fix up model names
    hubModels = list(hubModels.keys())
    hubModels = [name.replace("/", "-") for name in hubModels]

    missing = []
    for name in hubModels:
        if not name + "-Reps.npy" in repFiles:
            print(f"Missing representations from model: {name}")
            missing += [name]

    return missing


def limit_hub_rep_feature_size(repDir, featureLimit):
    """
    Return a list of hub model representations in repDir that have less than or
    equal to featureLimit features.
    """
    # Get a list of rep files
    repFiles = glob.glob(os.path.join(repDir, "*.npy"))

    # Loop through rep files
    keptReps = []
    for file in repFiles:
        # Load rep
        rep = np.load(file)
        # Check if it has less than or equal to featureLimit features
        if rep.shape[-1] <= featureLimit:
            print(f"{file} has {rep.shape[-1]} features, keeping!")
            keptReps += [file]
        else:
            print(f"{file} has {rep.shape[-1]} features, too many!")

    print(repDir)
    return keptReps


def hubSims_completion_check(simDir, hubInfo, featureLimit=None):
    """
    Return a list of missing Tensorflow Hub similarities in simDir given
    hubInfo. If feature limit is set, check that the number of features is
    valid.
    """
    # Load hub model info
    with open(hubInfo, "r") as f:
        hubModels = json.loads(f.read())

    # Create combination of hub models
    hubModels = list(hubModels.keys())

    # Loop through models and look for the corresponding simliarity file
    missing = []
    for model1 in hubModels:
        # Fix model name
        model1 = model1.replace("/", "-")

        if not model1 + ".csv" in os.listdir(simDir):
            # Check if this model that was skipped because it was too big
            if featureLimit is not None:
                rep = np.load(os.path.join(simDir, "..", model1 + "-Reps.npy"))
                if rep.shape[-1] > featureLimit:
                    print(f"{model1} has {rep.shape[-1]} features, skipped!")
                    continue
            print(f"Missing similarity from model: {model1}")
            missing += [model1]

    return missing


def compile_hub_sims(simDir):
    """
    Return a compiled dataframe of similarity data from simDir.
    """
    # Get list of sim files
    simFiles = glob.glob(os.path.join(simDir, "*.csv"))

    # Loop through sim files
    df = pd.DataFrame()
    for file in simFiles:
        tmp = pd.read_csv(file, index_col=0)
        df = pd.concat((df, tmp))

    return df




def clear_model():
    # Check if model variable exists
    if "model" in globals():
        del globals()["model"]
        tf.keras.backend.clear_session()
        gc.collect()



if __name__ == "__main__":
    print("Hello world!")