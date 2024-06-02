import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import os
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import hubReps
import csv
import math
from itertools import chain
import pandas as pd
from scipy.spatial.distance import cdist
import utilities as utils


def image_list(data_dir):
    img_files = os.listdir(data_dir)
    img_files.sort()
    image_names = []
    for file in img_files:
        image_names.append(file)
    return image_names


def rep_maker(
    data_dir, modelFile, model_name, modelData, model, batch_size, jitter_pixels=0
):
    if "origin" in modelData.keys() and modelData["origin"] == "keras":
        # Override model input shape for keras models if input_shape isn't none
        inputShape = model.input_shape[1:]
        if not any([dim is None for dim in inputShape]):
            modelData["shape"] = model.input_shape[1:]

    # get dataset from datasets.py
    dataset = get_dataset(
        data_dir, modelFile, model_name, modelData, model, batch_size, jitter_pixels
    )
    # print('\nDataset size:', dataset, '\n')

    # get reps from hubReps.py
    # could have a pointer error with iterations.
    return get_reps(modelFile, model, dataset, modelData, batch_size)





def find_model(fileList, modelName):
    fileList = fileList.split(", ")
    for file in fileList:
        with open(file, "r") as f:
            hubModels = json.loads(f.read())
            if modelName in hubModels:
                modelFile = file
                modelData = hubModels[modelName]

    return modelFile, modelData


def get_model(modelName, modelFile, hubModels):
    if modelFile == "../data_storage/hubModel_storage/hubModels.json":
        # Creating models
        info = hubModels[modelName]
        shape = info["shape"] if "shape" in info.keys() else [224, 224, 3]
        # Create model from tfhub
        inp = tf.keras.Input(shape=shape)
        out = hub.KerasLayer(info["url"])(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
    elif modelFile == "../data_storage/hubModel_storage/hubModels_keras.json":
        # Create model from keras function
        model = hubReps.get_keras_model(hubModels, modelName)[0]
    elif (
        (modelFile == "../data_storage/hubModel_storage/hubModels_pytorch.json")
        or (modelFile == "../data_storage/hubModel_storage/hubModels_timm.json")
        or (modelFile == "../data_storage/hubModel_storage/hubModels_transformers.json")
        or (
            modelFile
            == "../data_storage/hubModel_storage/hubModels_pretrainedmodels.json"
        )
    ):
        # Create model from pytorch hub
        model = hubReps.get_pytorch_model(hubModels, modelFile, modelName)
        model.eval()
    else:
        raise ValueError(f"Unknown models file {modelFile}")

    return model


def get_dataset(
    data_dir, modelFile, modelName, modelData, model, batch_size=64, jitter_pixels=0
):
    if (
        modelFile == "../data_storage/hubModel_storage/hubModels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_keras.json"
    ):
        preprocFun = datasets.preproc(
            **modelData,
            labels=False,
        )
        dataset = datasets.get_flat_dataset(
            data_dir, preprocFun, batch_size=batch_size, jitter_pixels=jitter_pixels
        )

    elif (
        modelFile == "../data_storage/hubModel_storage/hubModels_pytorch.json"
        or modelFile
        == "../data_storage/hubModel_storage/hubModels_pretrainedmodels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_timm.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_transformers.json"
    ):

        # using pytorch model

        dataset = datasets.get_pytorch_dataset(
            data_dir,
            modelData,
            model,
            batch_size,
            modelName,
            jitter_pixels=jitter_pixels,
        )

    else:
        raise ValueError(f"Unknown models file {modelFile}")

    return dataset


def get_reps(modelFile, model, dataset, modelData, batch_size=64):

    if (
        modelFile == "../data_storage/hubModel_storage/hubModels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_keras.json"
    ):
        reps = hubReps.get_reps(model, dataset, modelData, batch_size)

        utils.clear_model()

    elif (
        modelFile == "../data_storage/hubModel_storage/hubModels_pytorch.json"
        or modelFile
        == "../data_storage/hubModel_storage/hubModels_pretrainedmodels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_timm.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_transformers.json"
    ):

        reps = hubReps.get_pytorch_reps(model, dataset, modelData, batch_size)

    return reps


if __name__ == "__main__":
    import argparse

    print("\n\n/////////////////////////////////////")
    os.environ["TORCH_HOME"] = "/data/modelnet/torch"

    parser = argparse.ArgumentParser(
        description="Get representations from Tensorflow Hub networks using the validation set of ImageNet, intended to be used in HPC"
    )
    parser.add_argument(
        "--model_name",
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
        "--data_dir",
        "-d",
        type=str,
        help="directory of the image dataset",
        default="../novset",
    )
    parser.add_argument(
        "--comp_dir",
        "-cd",
        type=str,
        help="directory of the compared images",
        default="./data_storage/temp_images",
    )
    parser.add_argument(
        "--model_files",
        "-f",
        type=str,
        help=".json file with the hub model info",
        default="../data_storage/hubModel_storage/hubModels.json, ../data_storage/hubModel_storage/hubModels_timm.json, ../data_storage/hubModel_storage/hubModels_keras.json, ../data_storage/hubModel_storage/hubModels_pytorch.json",
    )
    parser.add_argument(
        "--feature_limit",
        "-l",
        type=int,
        help="the maximum number of features a representation can have",
        default=2048,
    )
    parser.add_argument(
        "--reps_name",
        type=str,
        help="name of the representations to save",
        default="temp_reps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of the dataset to use, use 'test' for testing a model",
        default="test",
    )
    parser.add_argument(
        "--simSet",
        type=str,
        default="all",
        help="which set of similarity functions to use",
    )
    parser.add_argument(
        "--csv_file",
        "-cf",
        type=str,
        default="./data_storage/ziggerins_trials.csv",
        help="which csv of trial to use",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        help="which trial type to use",
    )
    parser.add_argument(
        "--jitter_pixels",
        type=int,
        default=0,
        help="number of pixels to jitter images left/right by",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=float,
        default=0.0,
        help="amount of noise to add",
    )
    parser.add_argument(
        "--encoding_noise",
        type=float,
        default=0.0,
        help="amount of encoding noise to add, based on durations",
    )
    parser.add_argument(
        "--memory_decay",
        type=float,
        default=0.0,
        help="amount of memory decay to add",
    )
    parser.add_argument(
        "--learning_adv",
        type=float,
        default=0.0,
        help="amount of learning advantage to add",
    )
    parser.add_argument(
        "--use_gpu", default=False, action="store_true", help="uses GPU"
    )
    args = parser.parse_args()

    if args.use_gpu:
        import torch

        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        import torch

        # Disable gpu for tensorflow
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Disable gpu for torch
        torch.set_default_tensor_type("torch.FloatTensor")

    badModels = ["nts-net"]
    # Get all model files and information
    fileList = args.model_files.split(", ")

    # Read each json file as a dictionary
    hubModels = {}
    for file in fileList:
        with open(file, "r") as f:
            tmp = json.loads(f.read())

            # Add modelFile to each model
            for model in tmp.keys():
                tmp[model]["modelFile"] = file

            hubModels.update(tmp)

    if args.model_name is not None:
        # Fill a list with just that first model
        modelList = [args.model_name]

    else:
        # Fill a list with all the models
        modelList = list(hubModels.keys())

    if args.test == "threeAFC":
        # Setup results file
        # Check if results already exists
        resultsPath = f"../data_storage/results/results_{args.test}"

        if args.jitter_pixels != 0:
            resultsPath += f"_jitter{args.jitter_pixels}"

        if args.noise != 0:
            resultsPath += f"_noise-{args.noise}"

            if args.learning_adv != 0:
                resultsPath += f"_learnAdv-{args.learning_adv}"

        if args.encoding_noise != 0:
            resultsPath += f"_encNoise-{args.encoding_noise}"

        resultsPath += ".csv"
        if os.path.exists(resultsPath):
            results = pd.read_csv(resultsPath)
        else:
            results = pd.DataFrame(
                columns=[
                    "ModelName",
                    "Response",
                    "Corr",
                    "Trial",
                    "CorrRes",
                    "Duration",
                    "Feedback",
                ]
            )

        # Load model
        missingModels = []
        for modelName in modelList:
            modelData = hubModels[modelName]
            modelFile = modelData["modelFile"]

            rep_path = f"../data_storage/results/3afc_reps/{modelName.replace('/', '-')}-3afc{'' if args.jitter_pixels == 0 else '_jitter' + str(args.jitter_pixels)}.npy"
            image_name_path = f"../data_storage/results/3afc_reps/{modelName.replace('/', '-')}-3afc{'' if args.jitter_pixels == 0 else '_jitter' + str(args.jitter_pixels)}.txt"
            ddir = "../data_storage/standalone/3AFC_set"
            if os.path.exists(rep_path):
                print(f"Already have reps for {args.test} from {modelName}")

                if modelName in results["ModelName"].values:
                    print(f"Already have results for 3afc: {modelName}")
                    # Get the accuracy for this model
                    acc = results[results["ModelName"] == modelName]["Corr"].mean()
                    print(f"Accuracy for {modelName}: {acc}")
                    continue
                else:
                    reps = np.load(rep_path)
                    img_names = []
                    with open(image_name_path, "r") as f:
                        for line in f:
                            img_names.append(line.strip())
            else:
                if modelName in badModels:
                    print(f"Skipping {modelName}")
                    missingModels.append(modelName)

                    continue

                try:
                    model = get_model(modelName, modelFile, hubModels)
                except Exception as e:
                    # Echo exception
                    print(f"Error loading model {modelName}: {e}")
                    missingModels.append(modelName)

                    continue

                print(f"New reps for {args.test}: {modelName}")

                # Get file list and save
                img_names = os.listdir(ddir)
                img_names.sort()
                # Save file list as text file
                with open(
                    image_name_path,
                    "w",
                ) as f:
                    for file in img_names:
                        f.write(file + "\n")

                # Adjust batch size based on number of parameters
                if "num_params" not in modelData.keys():
                    batch_size = args.batch_size
                else:
                    batch_size = int(
                        args.batch_size
                        * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                    )
                    batch_size = 2 if batch_size < 2 else batch_size

                try:
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                        args.jitter_pixels,
                    )
                except Exception as e:
                    print(f"Error making reps for {modelName}: {e}")
                    print(f"Attempting fallback batch_size")
                    batch_size = 2
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                        args.jitter_pixels,
                    )

                # Flatten reps
                reps = reps.reshape(reps.shape[0], -1)
                np.save(rep_path, reps)

            print(f"New results for {args.test}: {modelName}")
            modelResults = three_afc(
                modelName,
                img_names,
                reps,
                "../data_storage/three_AFC_trials.csv",
                noise=args.noise,
                learning_adv=args.learning_adv,
                encoding_noise=args.encoding_noise,
                jitter_pixels=args.jitter_pixels,
            )
            results = pd.concat([results, modelResults])

            # Save results
            results.to_csv(resultsPath, index=False)

    elif args.test == "many_odd":
        # Setup results file
        # Check if results already exists
        resultsPath = f"../data_storage/results/results_{args.test}"

        if args.jitter_pixels != 0:
            UserWarning(f"Jitter pixels does nothing for {args.test}")

        if args.noise != 0:
            resultsPath += f"_noise-{args.noise}"

        if args.encoding_noise != 0:
            resultsPath += f"_encNoise-{args.encoding_noise}"

        resultsPath += ".csv"
        if os.path.exists(resultsPath):
            results = pd.read_csv(resultsPath)
        else:
            results = pd.DataFrame(
                columns=[
                    "ModelName",
                    "Response",
                    "Corr",
                    "Trial",
                    "CorrRes",
                    "Duration",
                    "Feedback",
                ]
            )

        # Load model
        missingModels = []
        for modelName in modelList:
            modelData = hubModels[modelName]
            modelFile = modelData["modelFile"]

            rep_path = f"../data_storage/results/moo_reps/{modelName.replace('/', '-')}-moo.npy"
            image_name_path = f"../data_storage/results/moo_reps/{modelName.replace('/', '-')}-moo.txt"
            ddir = "../data_storage/standalone/MOO_set"
            if os.path.exists(rep_path):
                print(f"Already have reps for {args.test} from {modelName}")

                if modelName in results["ModelName"].values:
                    print(f"Already have results for moo: {modelName}")
                    # Get the accuracy for this model
                    acc = results[results["ModelName"] == modelName]["Corr"].mean()
                    print(f"Accuracy for {modelName}: {acc}")
                    continue
                else:
                    reps = np.load(rep_path)
                    img_names = []
                    with open(image_name_path, "r") as f:
                        for line in f:
                            img_names.append(line.strip())
            else:
                if modelName in badModels:
                    print(f"Skipping {modelName}")
                    missingModels.append(modelName)

                    continue
                try:
                    model = get_model(modelName, modelFile, hubModels)
                except Exception as e:
                    # Echo exception
                    print(f"Error loading model {modelName}: {e}")
                    missingModels.append(modelName)

                    continue

                print(f"New reps for {args.test}: {modelName}")

                # Get file list and save
                img_names = os.listdir(ddir)
                img_names.sort()
                # Save file list as text file
                with open(
                    image_name_path,
                    "w",
                ) as f:
                    for file in img_names:
                        f.write(file + "\n")

                # Adjust batch size based on number of parameters
                if "num_params" not in modelData.keys():
                    batch_size = args.batch_size
                else:
                    batch_size = int(
                        args.batch_size
                        * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                    )
                    batch_size = 2 if batch_size < 2 else batch_size

                try:
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )
                except Exception as e:
                    print(f"Error making reps for {modelName}: {e}")
                    print(f"Attempting fallback batch_size")
                    batch_size = 2
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )

                # Flatten reps
                reps = reps.reshape(reps.shape[0], -1)
                np.save(rep_path, reps)

            print(f"New results for {args.test}: {modelName}")
            modelResults = many_oddball(
                modelName,
                img_names,
                reps,
                "../data_storage/many_odd_trials.csv",
                noise=args.noise,
                encoding_noise=args.encoding_noise,
            )
            results = pd.concat([results, modelResults])

            # Save results
            results.to_csv(resultsPath, index=False)

    elif args.test == "learn_exemp":
        # Setup results file
        # Check if results already exists
        resultsPath = f"../data_storage/results/results_{args.test}"

        if args.jitter_pixels != 0:
            resultsPath += f"_jitter{args.jitter_pixels}"

        if args.noise != 0:
            resultsPath += f"_noise-{args.noise}"

            if args.learning_adv != 0:
                resultsPath += f"_learnAdv-{args.learning_adv}"

        if args.memory_decay != 0:
            raise NotImplementedError("Memory decay not implemented")
            resultsPath += f"_memDecay-{args.memory_decay}"
        resultsPath += ".csv"
        if os.path.exists(resultsPath):
            results = pd.read_csv(resultsPath)
        else:
            results = pd.DataFrame(
                columns=[
                    "ModelName",
                    "Response",
                    "Corr",
                    "Trial",
                    "Img",
                    "Target",
                    "CorrRes",
                    "FoilLevel",
                    "View",
                    "Noise",
                    "Foil1",
                    "Foil2",
                ]
            )

        missingModels = []
        for modelName in modelList:
            modelData = hubModels[modelName]
            modelFile = modelData["modelFile"]

            rep_path = f"../data_storage/results/le_reps/{modelName.replace('/', '-')}-learnExemp{'' if args.jitter_pixels == 0 else '_jitter' + str(args.jitter_pixels)}.npy"
            image_name_path = f"../data_storage/results/le_reps/{modelName.replace('/', '-')}-learnExemp{'' if args.jitter_pixels == 0 else '_jitter' + str(args.jitter_pixels)}.txt"
            ddir = "../data_storage/standalone/LE_set"
            if os.path.exists(rep_path):
                print(f"Already have reps for {args.test} from {modelName}")

                if modelName in results["ModelName"].values:
                    print(f"Already have results for le: {modelName}")
                    # Get the accuracy for this model
                    acc = results[results["ModelName"] == modelName]["Corr"].mean()
                    print(f"Accuracy for {modelName}: {acc}")
                    continue
                else:
                    reps = np.load(rep_path)
                    img_names = []
                    with open(image_name_path, "r") as f:
                        for line in f:
                            img_names.append(line.strip())
            else:
                try:
                    if modelName in badModels:
                        print(f"Skipping {modelName}")
                        missingModels.append(modelName)

                        continue

                    model = get_model(modelName, modelFile, hubModels)
                except Exception as e:
                    # Echo exception
                    print(f"Error loading model {modelName}: {e}")
                    missingModels.append(modelName)

                    continue

                print(f"New reps for {args.test}: {modelName}")

                # Get file list and save
                img_names = os.listdir(ddir)
                img_names.sort()
                # Save file list as text file
                with open(
                    image_name_path,
                    "w",
                ) as f:
                    for file in img_names:
                        f.write(file + "\n")

                # Adjust batch size based on number of parameters
                if "num_params" not in modelData.keys():
                    batch_size = args.batch_size
                else:
                    batch_size = int(
                        args.batch_size
                        * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                    )
                    batch_size = 2 if batch_size < 2 else batch_size

                try:
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                        jitter_pixels=args.jitter_pixels,
                    )
                except Exception as e:
                    print(f"Error making reps for {modelName}: {e}")
                    print(f"Attempting fallback batch_size")
                    batch_size = 2
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                        jitter_pixels=args.jitter_pixels,
                    )

                # Flatten reps
                reps = reps.reshape(reps.shape[0], -1)
                np.save(rep_path, reps)

            print(f"New results for {args.test}: {modelName}")
            modelResults = learning_exemplar(
                modelName,
                img_names,
                reps,
                "../data_storage/learning_exemplar_trials.csv",
                noise=args.noise,
                learning_adv=args.learning_adv,
                memory_decay=args.memory_decay,
            )
            results = pd.concat([results, modelResults])

            # Save results
            results.to_csv(resultsPath, index=False)

    else:
        print(f"Invalid test: {args.test}")

    print("Missing models:", missingModels)
