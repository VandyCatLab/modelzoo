import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import os
import analysis
import itertools
import pandas as pd
import datetime


def setup_hub_model(info, batch_size, data_dir, slice):
    # Create model
    shape = info["shape"] if "shape" in info.keys() else [224, 224, 3]
    inp = tf.keras.Input(shape=shape)
    out = hub.KerasLayer(info["url"])(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)

    # Create dataset
    preprocFun = datasets.preproc(
        **info,
        labels=False,
    )
    dataset = datasets.get_imagenet_set(
        preprocFun, batch_size, data_dir=data_dir, slice=slice
    )
    return model, dataset


def get_reps(model, dataset, info, batch_size):
    """Manual batching to avoid memory problems."""
    # Num batches
    nBatches = len(dataset)

    dataset = dataset.as_numpy_iterator()

    if "outputIdx" in info.keys():
        # Get output size of model
        output_size = model.output_shape[info["outputIdx"]][1:]
    else:
        # Get output size of model
        output_size = model.output_shape[1:]

    # Create empty array to store representations
    reps = np.zeros((nBatches * batch_size, *output_size), dtype="float32")
    numImgs = 0
    for i, batch in enumerate(dataset):
        print(
            f"-- Working on batch {i} [{datetime.datetime.now()}]", flush=True
        )
        numImgs += len(batch)
        res = model.predict(batch)

        if "outputIdx" in info.keys():
            # Save representations
            reps[i * batch_size : (i + 1) * batch_size] = res[
                :, info["outputIdx"]
            ]
        else:
            # Save representations
            reps[i * batch_size : (i + 1) * batch_size] = res

    # Remove empty rows
    reps = reps[:numImgs]
    return reps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get representations from Tensorflow Hub networks using the validation set of ImageNet, intended to be used in HPC"
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        required=True,
        help="analysis to perform",
        choices=["reps", "similarity"],
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="name of the model to get representations from",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        help="index of the model to get representations from",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=64,
        help="batch size for the images passing through networks",
    )
    parser.add_argument(
        "--slice",
        "-s",
        type=str,
        help="slice string to use on data",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        help="directory of the dataset",
        default="~/tensorflow_datasets",
    )
    parser.add_argument(
        "--models_file",
        "-f",
        type=str,
        help=".json file with the hub model info",
        default="./hubModels.json",
    )
    parser.add_argument(
        "--feature_limit",
        "-l",
        type=int,
        help="the maximum number of features a representation can have",
        default=2048,
    )
    args = parser.parse_args()

    with open(args.models_file, "r") as f:
        hubModels = json.loads(f.read())

    # Get the model info
    if args.model is not None:
        modelName = args.model
    else:
        modelName = list(hubModels.keys())[args.index]

    if args.analysis == "reps":
        print(
            f"==== Working on model: {modelName} [{datetime.datetime.now()}] ====",
            flush=True,
        )
        fileName = f"../outputs/masterOutput/hubReps/{modelName.replace('/', '-')}-Reps.npy"
        if os.path.exists(fileName):
            print(f"Already completed, skipping.")
        else:
            model, dataset = setup_hub_model(
                hubModels[modelName],
                args.batch_size,
                args.data_dir,
                args.slice,
            )

            reps = get_reps(
                model, dataset, hubModels[modelName], args.batch_size
            )
            np.save(
                fileName,
                reps,
            )
            print(f"Saved {fileName}", flush=True)
    elif args.analysis == "similarity":
        print(
            f"==== Working on similarities for model: {modelName} [{datetime.datetime.now()}] ====",
            flush=True,
        )

        # Check if similarity file already exists
        fileName = f"../outputs/masterOutput/hubReps/hubSims/{modelName.replace('/', '-')}.csv"
        if os.path.exists(fileName):
            print(f"Already completed, skipping.")
        else:
            # Load representations
            modelRepsName = f"../outputs/masterOutput/hubReps/{modelName.replace('/', '-')}-Reps.npy"
            reps = np.load(modelRepsName)

            # If representations is not flat, average pool it
            if len(reps.shape) > 2:
                reps = np.mean(reps, axis=(1, 2))

            # Check if there's too many features
            if reps.shape[-1] > args.feature_limit:
                # Raise an error
                raise ValueError(
                    f"The number of features is too high: {reps.shape[0]}. [{datetime.datetime.now()}]"
                )

            # Get hub model names
            hubModelNames = list(hubModels.keys())

            # Find combinations of models and only keep combinations with this model
            modelCombinations = list(itertools.combinations(hubModelNames, 2))
            modelCombinations = [
                x for x in modelCombinations if x[0] == modelName
            ]

            # Similarity functions
            preprocFuns = [
                analysis.preprocess_peaRsaNumba,
                analysis.preprocess_eucRsaNumba,
                analysis.preprocess_speRsaNumba,
                analysis.preprocess_svcca,
                analysis.preprocess_ckaNumba,
            ]
            simFuns = [
                analysis.do_rsaNumba,
                analysis.do_rsaNumba,
                analysis.do_rsaNumba,
                analysis.do_svcca,
                analysis.do_linearCKANumba,
            ]
            analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]

            # Create dataframe to store results
            simDf = pd.DataFrame(columns=["model1", "model2"] + analysisNames)
            # Loop through hub model
            for _, pairModel in modelCombinations:
                # Print progress
                print(
                    f"--- Comparing against {pairModel} [{datetime.datetime.now()}]",
                    flush=True,
                )

                # Load hub model representations
                pairModelRepsName = f"../outputs/masterOutput/hubReps/{pairModel.replace('/', '-')}-Reps.npy"
                pairReps = np.load(pairModelRepsName)

                # If representations is not flat, average pool it
                if len(pairReps.shape) > 2:
                    pairReps = np.mean(pairReps, axis=(1, 2))

                # Check if there's too many features
                if pairReps.shape[-1] > args.feature_limit:
                    # Skip over this model
                    print(
                        f"Too many features from {pairModel}, skipping [{datetime.datetime.now()}]",
                        flush=True,
                    )
                    continue

                # Calculate similarity
                sims = analysis.multi_analysis(
                    reps, pairReps, preprocFuns, simFuns, names=analysisNames
                )

                # Add to dataframe
                simDf.loc[len(simDf.index)] = [modelName, pairModel] + [
                    sims[fun] for fun in sims.keys()
                ]

                # Delete representations for memory
                del pairReps

            # Save dataframe
            print(
                f"Saving similarities [{datetime.datetime.now()}]", flush=True
            )
            simDf.to_csv(fileName)
