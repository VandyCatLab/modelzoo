import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import os
import hubReps
import pandas as pd
import utilities as utils
import click
import torch
from typing import Union
import timm

_BAD_MODELS = []

_MODEL_FILES = [
    "../data_storage/hubModel_storage/hubModels.json",
    "../data_storage/hubModel_storage/hubModels_timm.json",
    "../data_storage/hubModel_storage/hubModels_keras.json",
    "../data_storage/hubModel_storage/hubModels_pytorch.json",
]

_DATA_DIRS = [
    "../images/fribbes",
    "../images/greebes",
    "../images/yugos",
    "../images/ziggerins",
]

# Compile all models
ZOOMODELS = {}
for file in _MODEL_FILES:
    with open(file, "r") as f:
        tmp = json.loads(f.read())

        for modelName in tmp.keys():
            tmp[modelName]["modelFile"] = file

        ZOOMODELS.update(tmp)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_name", type=str, required=True)
@click.argument("dataset_name", type=str, required=True)
@click.option("--batch_size", default=128, help="Initial batch size")
@click.option(
    "--no_batch_scaling",
    default=False,
    is_flag=True,
    help="Don't scale batch size based on model size",
)
@click.option("--no_gpu", default=False, is_flag=True, help="Don't use GPU")
def extract(
    model_name: str,
    dataset_name: str,
    batch_size: int = 128,
    no_batch_scaling: bool = False,
    no_gpu: bool = False,
) -> None:
    """
    Extract representations for a given model and dataset. Either model and
    dataset can be set to "all" to extract representations for all models or
    all datasets.
    """
    if model_name == "all":
        print("Working through all models")
        modelList = list(ZOOMODELS.keys())
    else:
        if model_name not in ZOOMODELS:
            raise ValueError(f"Model {model_name} not found")

        print(f"Loading model {model_name}")
        modelList = [model_name]

    if dataset_name == "all":
        print("Working through all known datasets")
        dataDirs = _DATA_DIRS
    else:
        # Check if dataset_name is a valid directory
        if not os.path.isdir(dataset_name):
            raise ValueError(f"Dataset {dataset_name} not found")

        dataDirs = [dataset_name]

    if no_gpu:
        print("Disabling GPU ops")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.set_default_tensor_type("torch.FloatTensor")

    # Go through models
    missingModels = []
    for model_name in modelList:
        modelData = ZOOMODELS[model_name]

        # Skip bad models
        if model_name in _BAD_MODELS:
            print(f"Skipping bad model {model_name}")
            continue

        try:
            model = get_model(modelData)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            missingModels.append(model_name)
            continue

        # Determine batch size
        if not no_batch_scaling:
            if not "num_params" in modelData:
                print("Cannot scale batch size, we don't know parameter count")
            else:
                # TODO: Try different way to determine memory size of batches
                batch_size = int(
                    batch_size
                    * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                )
                batch_size = 2 if batch_size < 2 else batch_size

        # Loop through datasets
        for dataDir in dataDirs:
            print(f"Working on {model_name} with {dataDir}")

            # Override model input shape for keras models if input_shape isn't none
            if "origin" in modelData and modelData["origin"] == "keras":
                inputShape = model.input_shape[1:]
                if not any([dim is None for dim in inputShape]):
                    modelData["shape"] = model.input_shape[1:]

            data = get_dataset(
                data_dir=dataDir,
                model_data=modelData,
                model=model,
                batch_size=batch_size,
            )

            # TODO: Implement get reps


def get_dataset(
    data_dir: str,
    model_data: dict,
    model: Union[tf.keras.Model, torch.nn.Module],
    batch_size=64,
) -> Union[tf.data.Dataset, torch.utils.data.DataLoader]:
    if (  # Keras
        "hubModels.json" in model_data["modelFile"]
        or "keras" in model_data["modelFile"]
    ):
        preprocFun = datasets.preproc(
            **model_data,
            labels=False,
        )
        dataset = datasets.get_flat_dataset(
            data_dir,
            preprocFun,
            batch_size=batch_size,
        )

    elif (  # Pytorch
        "pytorch" in model_data["modelFile"]
        or "pretrainedmodels" in model_data["modelFile"]
        or "timm" in model_data["modelFile"]
        or "transformers" in model_data["modelFile"]
    ):
        dataset = datasets.get_pytorch_dataset(
            data_dir,
            model_data,
            model,
            batch_size,
        )

    else:
        raise ValueError(f"Unknown models {model_data}")

    return dataset


def get_model(model_info: dict) -> Union[tf.keras.Model, torch.nn.Module]:
    """Load model given model_info"""

    if "hubModels.json" in model_info["modelFile"]:  # Original set of models
        # Creating models
        shape = model_info["shape"] if "shape" in model_info else [224, 224, 3]
        # Create model from tfhub
        inp = tf.keras.Input(shape=shape)
        out = hub.KerasLayer(model_info["url"])(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)

    elif "keras" in model_info["modelFile"]:  # Keras applications
        # Create model from keras function
        model = get_keras_model(model_info)

    elif (  # Pytorch models
        ("pytorch" in model_info["modelFile"])
        or ("timm" in model_info["modelFile"])
        or ("transformers" in model_info["modelFile"])
        or ("pretrainedmodels" in model_info["modelFile"])
    ):
        # Create model from pytorch hub
        model = get_pytorch_model(model_info)
        model.eval()

    else:
        raise ValueError(f"Unknown models file {model_info['modelFile']}")

    return model


def get_keras_model(model_info: dict) -> tf.keras.Model:
    """Loading keras model via evaluating a function"""
    function = model_info["function"]
    model_full = eval(function)

    inp = model_full.input
    layerName = model_full.layers[int(model_info["layerIdx"])].name
    out = model_full.get_layer(layerName).output
    model = tf.keras.Model(inputs=inp, outputs=out)

    return model


def get_pytorch_model(model_info: dict) -> torch.nn.Module:
    """Load pytorch models either from timm, transformers, or torchhub"""

    if "timm" in model_info["modelFile"]:
        model = timm.create_model(modelName, pretrained=True, num_classes=0)

    elif "transformers" in model_info["modelFile"]:
        function = model_info["func"]
        model = eval("transformers." + function + f'.from_pretrained("{modelName}")')
        # Currently unused but it's possible to access internals of some transformers models
    else:
        function = model_info["function"]
        model = eval("torch.hub.load" + function)

    return model


def get_reps(modelFile, model, dataset, modelData, batch_size=64):

    if (
        modelFile == "../data_storage/hubModel_storage/hubModels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_keras.json"
    ):
        reps = hubReps.get_reps(model, dataset, modelData, batch_size)

        utils.clear_model("model")

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
    cli()
