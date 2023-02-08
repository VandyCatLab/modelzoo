import datasets
import tensorflow as tf
import tensorflow.core
import tensorflow_hub as hub
import json
import numpy as np
import os
import analysis
import itertools
import pandas as pd
import datetime
import torch
from torchvision import transforms
import cv2
from torchvision.models.feature_extraction import create_feature_extractor
import transformers

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
        print("output size = ")
        print(output_size)
        print(model.output_shape)

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

            if res[info["outputIdx"]].shape[0] == batch_size:
                reps[i * batch_size : (i + 1) * batch_size] = res[
                 info["outputIdx"]]
            else:
                reps[i * batch_size: i * batch_size + len(res[info["outputIdx"]])] = res[info["outputIdx"]]

        else:
            # Save representations
            if res.shape[0] == batch_size:
                reps[i * batch_size : (i + 1) * batch_size] = res
            else:
                reps[i * batch_size : i * batch_size + len(res)] = res

    # Remove empty rows
    reps = reps[:numImgs]
    print(reps.shape)
    return reps


def get_pytorch_reps(model, dataset, info, batch_size):
    """Manual batching to avoid memory problems."""
    # Num batches
    nBatches = len(dataset)

    #dataset = torch.utils.data.TensorDataset(dataset)
    x = info["shape"]
    x.insert(0, 1)
    print(x)
    temp_data = torch.rand(x)
    a = info["outputLayer"][1]
    b = info["outputLayer"][0]
    return_nodes = {
        a: b
    }
    print(return_nodes)
    model_int = create_feature_extractor(model, return_nodes=return_nodes)  # dict(layer4 = 'layer4.2.conv2'))
    intermediate_outputs = model_int(temp_data)
    intermediate = intermediate_outputs[b]


    if "outputIdx" in info.keys():
        # Get output size of model
        output_size = tuple(intermediate.shape)

    else:
        # Get output size of model
        output_size = tuple(intermediate.shape)
        print(output_size)
    # Create empty array to store representations
    reps = np.zeros((nBatches * batch_size, *output_size[1:]), dtype="float32")

    numImgs = 0
    for i, batch in enumerate(dataset):
        print(
            f"-- Working on batch {i} [{datetime.datetime.now()}]", flush=True
        )
        batch = batch.float()
        numImgs += len(batch)
        res_full = model_int(batch)
        res_data = res_full[b]
        res = res_data.detach().numpy()
        if "outputIdx" in info.keys():
            # Save representations

            if res[info["outputIdx"]].shape[0] == batch_size:
                reps[i * batch_size: (i + 1) * batch_size] = res[
                    info["outputIdx"]]
            else:
                reps[i * batch_size: i * batch_size + len(res[info["outputIdx"]])] = res[info["outputIdx"]]

        else:
            # Save representations
            if res.shape[0] == batch_size:
                reps[i * batch_size: (i + 1) * batch_size] = res
            else:
                reps[i * batch_size: i * batch_size + len(res)] = res

    # Remove empty rows
    reps = reps[:numImgs]
    print(reps.shape)
    return reps

def get_keras_model(hubModels):
    function = hubModels[modelName]['function']
    model_full = eval(function)
    inp = model_full.input
    layerName = model_full.layers[int(hubModels[modelName]['layerIdx'])].name
    out = model_full.get_layer(layerName).output
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model, model_full

def get_pytorch_hub_model(hubModels):
    function = hubModels[modelName]['function']
    model = eval("torch.hub.load" + function)
    return model

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
        choices=["reps", "similarity", "test_model"],
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
        choices=["./hubModels.json","./hubModels_keras.json","./hubModels_pytorch.json"],
        default="./hubModels.json"
    )
    parser.add_argument(
        "--feature_limit",
        "-l",
        type=int,
        help="the maximum number of features a representation can have",
        default=2048,
    )
    parser.add_argument(
        "--rep_name",
        type=str,
        help="name of the representation to save",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of the dataset to use, use 'test' for testing a model",
    )
    parser.add_argument(
        "--group", "-g", type=str, help="analysis group to store together"
    )
    parser.add_argument(
        "--simSet",
        type=str,
        default="all",
        help="which set of similarity functions to use",
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
        if args.rep_name is not None:
            fileName = f"../hubreps/{modelName.replace('/', '-')}-{args.rep_name}Reps.npy"
        else:
            fileName = f"../hubreps/{modelName.replace('/', '-')}-Reps.npy"
        if os.path.exists(fileName):
            print(f"Already completed, skipping.")
        else:
            # Creating models
            info = hubModels[modelName]
            shape = info["shape"] if "shape" in info.keys() else [224, 224, 3]

            if args.models_file == "./hubModels.json":
                # Create model from tfhub
                inp = tf.keras.Input(shape=shape)
                out = hub.KerasLayer(info["url"])(inp)
                model = tf.keras.Model(inputs=inp, outputs=out)

            elif args.models_file == "./hubModels_keras.json":
                # Create model from keras function
                model = get_keras_model(hubModels)[0]

            elif args.models_file == "./hubModels_pytorch.json":
                # Create model from pytorch hub
                model = get_pytorch_hub_model(hubModels)

            elif args.models_file == "./hubModels_transformers.json":
                # Create model from transformers package
                model, trans_preproc = get_transformer(info["pretrain_func"], info["preproc_func"])

            else:
                raise ValueError(f"Unknown models file {args.models_file}")

            if args.models_file == "./hubModels.json" or args.models_file == "./hubModels_keras.json":
                preprocFun = datasets.preproc(
                    **hubModels[modelName],
                    labels=False,
                )

                if args.dataset == "imagenet":
                    dataset = datasets.get_imagenet_set(
                        preprocFun,
                        args.batch_size,
                        args.data_dir,
                        slice=args.slice,
                    )
                elif args.dataset == "novset":
                    dataset = datasets.get_flat_dataset(
                        args.data_dir, preprocFun, args.batch_size
                    )
                elif args.dataset == "kriegset":
                    dataset = datasets.get_flat_dataset(
                        args.data_dir, preprocFun, args.batch_size
                    )
                else:
                    raise ValueError(f"Unknown dataset {args.dataset}")

                reps = get_reps(
                    model, dataset, hubModels[modelName], args.batch_size
                )

            elif args.models_file == "./hubModels_pytorch.json" \
                    or args.models_file == "./hubModels_pretrainedmodels.json" \
                    or args.models_file == "./hubModels_timm.json" \
                    or args.models_file == "./hubModels_transformers.json":

                # using pytorch model
                dataset = datasets.get_pytorch_dataset(
                    args.data_dir, info, args.batch_size
                )

                reps = get_pytorch_reps(
                    model, dataset, hubModels[modelName], args.batch_size
                )
            else:
                raise ValueError(f"Unknown models file {args.models_file}")

            np.save(fileName, reps)
            print(f"Saved {fileName}", flush=True)


    elif args.analysis == "similarity":
        print(
            f"==== Working on similarities for model: {modelName} [{datetime.datetime.now()}] ====",
            flush=True,
        )

        # Add analysis group
        basePath = "../hubReps/hubSims/"
        if args.group is not None:
            basePath += args.group + "/"
        if not os.path.exists(basePath):
            os.mkdir(basePath)
        # Check if similarity file already exists
        fileName = f"{basePath}{modelName.replace('/', '-')}.csv"
        if os.path.exists(fileName):
            print(f"Already completed, skipping.")
        else:
            # Load representations
            if args.rep_name is not None:
                modelRepsName = f"../hubReps/{modelName.replace('/', '-')}-{args.rep_name}Reps.npy"
            else:
                modelRepsName = f"../hubReps/{modelName.replace('/', '-')}-Reps.npy"
            reps = np.load(modelRepsName)

            # If representations is not flat, average pool it
            if len(reps.shape) > 2:
                reps = np.mean(reps, axis=(1, 2))

            # Check if there's too many features
            if (
                args.feature_limit is not None
                and reps.shape[-1] > args.feature_limit
            ):
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
            preprocFuns, simFuns, analysisNames = analysis.get_funcs(
                args.simSet
            )

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
                if args.rep_name is not None:
                    pairModelRepsName = f"../hubReps/{pairModel.replace('/', '-')}-{args.rep_name}Reps.npy"
                else:
                    pairModelRepsName = f"../hubReps/{pairModel.replace('/', '-')}-Reps.npy"
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

    elif args.analysis == "test_model":
        print(
            f"==== Working on testing model: {modelName} [{datetime.datetime.now()}] ====",
            flush=True,
        )

        if args.dataset == "novset" or "kreigset":

            predictions = []
            if args.models_file == "./hubModels_keras.json":
                preprocFun = datasets.preproc(
                    **hubModels[modelName],
                    labels=False,
                )
                dataset = datasets.get_flat_dataset(
                    args.data_dir, preprocFun, args.batch_size
                )
                model_full = get_keras_model(hubModels)[1]
                pred_func = "tf.keras.applications." + hubModels[modelName]['type'] + ".decode_predictions(preds, top=5)"
                predictions = []
                for i, batch in enumerate(dataset):
                    preds = model_full.predict(batch)
                    predictions.append(eval(pred_func))

                print('\n'.join(map(str,predictions[0])))
            elif args.models_file == "./hubModels_pytorch.json":
                dataset = datasets.get_pytorch_dataset(
                    args.data_dir, hubModels[modelName], args.batch_size
                )
                model = get_pytorch_hub_model(hubModels)
                with torch.no_grad():
                    info = hubModels[modelName]
                    for i, batch in enumerate(dataset):
                        if "outputIdx" in info.keys():
                            output = model(batch)
                        else:
                            output = model(batch)
                tensor = torch.round(output[0])
                cv2.imwrite(tensor, "tensorImage.png")

            else:
                raise ValueError(f"models file unsupported for testing {args.dataset}")


