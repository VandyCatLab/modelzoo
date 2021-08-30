import glob
import pandas as pd
import os
import itertools
import numpy as np
import copy


def compile_correspondence(path, models_path):
    """
    Return dataframe of the correspondence data from path.
    """
    files = glob.glob(os.path.join(path, "*"))

    # Compile data
    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file, index_col=0)
        layerCol = list(range(11)) * int(len(tmp) / 11)
        tmp["layer"] = layerCol
        df = pd.concat((df, tmp))

    df["layer"] = df["layer"].astype(int)

    # Figure out which model combination is missing
    models = glob.glob(os.path.join(models_path, "*"))
    models = [model.split("/")[-1].split(".")[0] for model in models]
    missing = []
    for pair in itertools.combinations(models, 2):
        test = df[["model1", "model2"]] == pair
        test = test.all(axis=1)
        if not test.any():
            # Test reversed direction
            test = df[["model2", "model1"]] == pair
            test = test.all(axis=1)
            if not test.any():
                missing += [pair]
                continue

    return df, pd.DataFrame(missing, columns=["model1", "model2"])


def correspondence_missing_optimizer(missing):
    """
    Given the missing combinations, return an optimized list to load as few
    models as possible to complete the dataset.
    """
    data = missing.copy()
    modelPairs = {}
    models = list(np.unique(np.concatenate((data["model1"], data["model2"]))))

    while len(models) > 0:
        modelCounts = [[]] * len(models)
        for i, model in enumerate(models):
            modelCounts[i] = list(data["model2"].loc[data["model1"] == model])
            modelCounts[i] += list(data["model1"].loc[data["model2"] == model])

        modelN = np.array([len(count) for count in modelCounts])
        target = models.pop(np.argmax(modelN))
        modelPairs[target] = list(
            data["model1"].loc[data["model2"] == target]
        ) + list(data["model2"].loc[data["model1"] == target])
        data = data.loc[
            (data["model1"] != target) & (data["model2"] != target)
        ]

    return modelPairs


if __name__ == "__main__":
    path = "../outputs/masterOutput/correspondence/"
    modelsPath = "../outputs/masterOutput/models/"
    data, missing = compile_correspondence(path, modelsPath)
    modelPairs = correspondence_missing_optimizer(missing)
    modelPairs
