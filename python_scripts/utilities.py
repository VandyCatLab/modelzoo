import glob
import pandas as pd
import os
import itertools
import numpy as np
import scipy.stats as stats
import json


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


def compile_augment(path, augment, layer):
    """
    Return the data of certain type of baseline augment from path at a given
    layer.
    """
    files = glob.glob(os.path.join(path, f"*l{layer}-{augment}.csv"))
    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file)
        tmp["layer"] = layer
        df = pd.concat((df, tmp))

    if augment == "color":
        df["version"] = (df["version"] - 25) * (3 / 50)
    return df


def compile_baseline_dict(path, tests, layers):
    """
    Return a dictionary of the results from the list baseline tests, grouped
    by test then layer then metrics for d3 plotting.
    """
    metrics = {
        "rsa": "do_rsaNumba",
        "cca": "do_svcca",
        "cka": "do_linearCKANumba",
    }
    data = {test: {} for test in tests}
    for testKey in data.keys():
        layerData = {f"layer{layer}": {} for layer in layers}
        metricData = {metricKey: [] for metricKey in metrics.keys()}
        if testKey == "dropout":  # This is dropout
            df = compile_dropout(path)
            versions = df["dropRate"].unique()
            for layerKey in layerData.keys():
                for metricKey, metric in metrics.items():
                    for version in versions:
                        tmpData = df[metric].loc[
                            (df["dropRate"] == version)
                            & (df["layer"] == int(layerKey[5:]))
                        ]
                        mean = np.mean(tmpData)
                        ciLow, ciHigh = (
                            stats.t.interval(
                                alpha=0.95,
                                df=len(tmpData) - 1,
                                loc=mean,
                                scale=stats.sem(tmpData, nan_policy="omit"),
                            )
                            if stats.sem(tmpData, nan_policy="omit") > 0
                            else (mean, mean)  # No variance case catch
                        )

                        # Save data of a metric
                        metricData[metricKey] += [
                            {
                                "test": testKey,
                                "layer": layerKey,
                                "version": version,
                                "metric": metricKey,
                                "mean": mean,
                                "ciLow": ciLow,
                                "ciHigh": ciHigh,
                            }
                        ]
                # Save data of a layer
                layerData[layerKey] = metricData
            # Save data of a test
            data[testKey] = layerData

        else:  # Probably augmentation
            for layerKey in layerData.keys():
                df = compile_augment(path, testKey, int(layerKey[5:]))
                versions = df["version"].unique()

                for metricKey, metric in metrics.items():
                    if testKey == "translate":  # Translation test
                        directions = ["left", "right", "up", "down"]
                        dirKeys = [
                            f"{metric}-{direction}" for direction in directions
                        ]
                        df[metric] = df[dirKeys].mean(1)
                    for version in versions:
                        tmpData = df[metric].loc[df["version"] == version]
                        mean = np.mean(tmpData)
                        ciLow, ciHigh = (
                            stats.t.interval(
                                alpha=0.95,
                                df=len(tmpData) - 1,
                                loc=mean,
                                scale=stats.sem(tmpData, nan_policy="omit"),
                            )
                            if stats.sem(tmpData, nan_policy="omit") > 0
                            else (mean, mean)  # No variance case catch
                        )

                        # Save data of a metric
                        metricData[metricKey] += [
                            {
                                "test": testKey,
                                "layer": layerKey,
                                "version": version,
                                "metric": metricKey,
                                "mean": mean,
                                "ciLow": ciLow,
                                "ciHigh": ciHigh,
                            }
                        ]

                        if (
                            testKey == "translate"
                        ):  # Add directions for translation
                            for i, dirCol in enumerate(dirKeys):
                                metricData[metricKey][-1][
                                    directions[i]
                                ] = np.mean(
                                    df[dirCol].loc[df["version"] == version]
                                )

                    # Save data of a layer
                    layerData[layerKey] = metricData
                # Save data of a test
                data[testKey] = layerData

    return data


def compile_dropout(path):
    """
    Return the data from the baseline dropout representations from a path.
    """
    files = glob.glob(os.path.join(path, "*dropout*"))

    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file)
        df = pd.concat((df, tmp))

    return df


if __name__ == "__main__":
    path = "../outputs/masterOutput/baseline/"
    df = compile_dropout(path)
    df.to_csv(f"../outputs/masterOutput/baseline/compiled/dropout.csv")

    path = "../outputs/masterOutput/baseline/"
    augments = ["color", "translate", "zoom", "reflect"]
    layers = [3, 7, 11]

    for augment in augments:
        df = pd.DataFrame()
        for layer in layers:
            tmp = compile_augment(path, augment, layer)
            df = pd.concat((df, tmp))
        df.to_csv(f"../outputs/masterOutput/baseline/compiled/{augment}.csv")

    path = "../outputs/masterOutput/correspondence/"
    models_path = "../outputs/masterOutput/models/"
    results, missing = compile_correspondence(path, models_path)
    missing = correspondence_missing_optimizer(missing)
    results.to_csv(f"../outputs/masterOutput/correspondence.csv")

    # path = "../outputs/masterOutput/baseline/"
    # tests = ["color", "translate", "zoom", "reflect", "dropout"]
    # layers = [3, 7, 11]
    # data = compile_baseline_dict(path, tests, layers)
    # with open(os.path.join(path, "compiled", "baseline.json"), "w") as outfile:
    #     json.dump(data, outfile)

