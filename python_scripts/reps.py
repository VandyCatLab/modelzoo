import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import os
import utilities as utils
import click
import torch
from typing import Union, List
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import pretrainedmodels
import analysis
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

_BAD_MODELS = ["nts-net"]

_MODEL_FILES = [
    "../data_storage/hubModel_storage/hubModels.json",
    "../data_storage/hubModel_storage/hubModels_timm.json",
    "../data_storage/hubModel_storage/hubModels_keras.json",
    "../data_storage/hubModel_storage/hubModels_pytorch.json",
]

_DATA_DIRS = {
    "fribbles": "../images/fribbles",
    "greebles": "../images/greebles",
    "yufos": "../images/yufos",
    "ziggerins": "../images/ziggerins",
    "ecoset": "../images/ecoset",
    "CUB200": "../images/CUB200",
    "cars196": "../images/cars196",
    "kiani": "../images/kiani",
    "VGGFace": "../images/VGGFace",
}


# MARK: CLI
@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_name", type=str, required=True)
@click.argument("dataset", type=str, required=True)
@click.option("--batch_size", default=128, help="Initial batch size")
@click.option(
    "--batch_magnitude",
    default=3,
    help="How many magnitude to go down while scaling batch size",
)
@click.option(
    "--no_batch_scaling",
    default=False,
    is_flag=True,
    help="Don't scale batch size based on model size",
)
@click.option(
    "--gpu_id",
    default=0,
    help="GPU id to use, if -1, use CPU",
)
@click.option(
    "--backwards",
    default=False,
    is_flag=True,
    help="Work through models in reverse order (for parallel gpu usage)",
)
def extract(
    model_name: str,
    dataset: str,
    batch_size: int = 128,
    batch_magnitude: int = 3,
    no_batch_scaling: bool = False,
    gpu_id: int = 0,
    backwards: bool = False,
) -> None:
    """
    Extract representations for a given model and dataset. Either model and
    dataset can be set to "all" to extract representations for all models or
    all datasets.
    """
    zooModels = _get_model_dicts()
    if model_name == "all":
        click.echo("Working through all models")
        modelList = list(zooModels.keys())
    else:
        if model_name not in zooModels:
            raise ValueError(f"Model {model_name} not found")

        click.echo(f"Loading model {model_name}")
        modelList = [model_name]

    if dataset == "all":
        click.echo("Working through all known datasets")
        dataDirs = []
        dataNames = []
        for name, directory in _DATA_DIRS.items():
            dataDirs.append(directory)
            dataNames.append(name)
    else:
        # Check if dataset_name is a key in _DATA_DIRS
        if dataset in _DATA_DIRS:
            dataDirs = [_DATA_DIRS[dataset]]
            dataNames = [dataset]
        else:
            # Check if dataset_name is a valid directory
            if not os.path.isdir(dataset):
                raise ValueError(f"Dataset {dataset} not found")

            dataDirs = [dataset]
            dataNames = [dataset.split("/")[-1]]

    torch.set_default_dtype(torch.float32)
    if gpu_id == -1:
        click.echo("Disabling GPU ops")
        torch.set_default_device("cpu")
    else:
        torch.set_default_device(f"cuda:{gpu_id}")
        tfDevices = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(tfDevices[gpu_id], "GPU")

    if backwards:
        modelList = modelList[::-1]

    # Go through models
    missingModels = []
    for model_name in modelList:
        # Make all data paths
        simPaths = [
            f"../data_storage/RDMs/{model_name.replace('/', '-')}_{dataName}.npy"
            for dataName in dataNames
        ]

        # Check if all data paths exist
        if all([os.path.exists(path) for path in simPaths]):
            click.echo(f"Skipping {model_name}, all sim files exists")
            continue

        # Get model data
        modelData = zooModels[model_name]

        # Skip bad models
        if model_name in _BAD_MODELS:
            click.echo(f"Skipping bad model {model_name}")
            continue

        try:
            model = get_model(modelData)

            # Move pretrained models to GPU if needed
            if (
                "origin" in modelData
                and modelData["origin"] == "pretrainedmodels"
                and not gpu_id == -1
            ):
                model = model.cuda()
        except tf.errors.ResourceExhaustedError:
            click.echo(f"Out of GPU memory for {model_name}")

            # Attempt to load model in CPU
            try:
                model = get_model(modelData)
            except Exception as e:
                click.echo(f"Error loading CPU model {model_name}: {e}")
                missingModels.append(model_name)
                continue
        except torch.cuda.OutOfMemoryError:
            click.echo(f"Out of GPU memory for {model_name}")

            # Attempt to load model in CPU
            try:
                with torch.device("cpu"):
                    model = get_model(modelData, gpu_id=-1)
            except Exception as e:
                click.echo(f"Error loading CPU model {model_name}: {e}")
                missingModels.append(model_name)
                continue
        except Exception as e:
            click.echo(f"Error loading model {model_name}: {e}")
            missingModels.append(model_name)
            continue

        # Determine batch size
        if not no_batch_scaling:
            if isinstance(model, torch.nn.Module):
                # Get model parameters for pytorch models
                nParams = np.sum([p.numel() for p in model.parameters()])
            elif isinstance(model, tf.keras.Model) or isinstance(model, hub.KerasLayer):
                # Get model parameters for keras models
                nParams = np.sum([np.prod(p.shape) for p in model.get_weights()])
            elif "num_params" in modelData:
                nParams = modelData["num_params"]
            else:
                nParams = None

            if nParams is None:
                click.echo("Cannot scale batch size, we don't know parameter count")
            else:
                batch_size = int(
                    batch_size
                    * (1 / 2 ** int(np.log10(int(nParams)) - batch_magnitude))
                )
                # Clip batch size to 1
                batch_size = 1 if batch_size < 1 else batch_size

        # Loop through datasets
        for dataDir, simPath in zip(dataDirs, simPaths):
            # Check if simPath exists
            if os.path.exists(simPath):
                click.echo(f"Skipping {model_name} with {dataDir}, sim file exists")
                continue

            click.echo(f"Working on {model_name} with {dataDir}")

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

            try:
                reps = get_reps(
                    model=model,
                    dataset=data,
                    model_data=modelData,
                    batch_size=batch_size,
                )
            except (tf.errors.ResourceExhaustedError, torch.cuda.OutOfMemoryError):
                click.echo("Out of memory during inference, minimize batch size")
                # Notice this can error out again, just to avoid missing this data
                data = get_dataset(
                    data_dir=dataDir,
                    model_data=modelData,
                    model=model,
                    batch_size=1,
                )

                reps = get_reps(
                    model=model,
                    dataset=data,
                    model_data=modelData,
                    batch_size=1,
                )

            # Create an RDM
            click.echo("Calculating RDM")
            reps = analysis.preprocess_eucRsaNumba(reps)

            # Save
            np.save(simPath, reps)

        utils.clear_model()

    # Print bad models
    if len(missingModels) > 0:
        click.echo(f"Missing models: {missingModels}")


# MARK: Model functions
def _get_model_dicts(model_files: List[str] = _MODEL_FILES) -> dict:
    """Return model dictionary given the model files"""
    zooModels = {}
    for file in _MODEL_FILES:
        with open(file, "r") as f:
            tmp = json.loads(f.read())

            for modelName in tmp.keys():
                tmp[modelName]["modelFile"] = file
                tmp[modelName]["name"] = modelName

            zooModels.update(tmp)

    return zooModels


def get_model(model_info: dict) -> Union[tf.keras.Model, torch.nn.Module]:
    """Load model given model_info"""

    if "hubModels.json" in model_info["modelFile"]:  # Original set of models
        # Creating models from tf hub
        model = hub.KerasLayer(model_info["url"])

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
        model = timm.create_model(
            model_info["architecture"], pretrained=True, num_classes=0
        )

    elif "transformers" in model_info["modelFile"]:
        function = model_info["func"]
        model = eval(
            "transformers." + function + f'.from_pretrained("{model_info["name"]}")'
        )
    elif "function" in model_info:
        function = model_info["function"]
        model = eval("torch.hub.load" + function)
    elif "origin" in model_info and model_info["origin"] == "pretrainedmodels":
        # Make this model load first onto cpu
        with torch.device("cpu"):
            model = pretrainedmodels.__dict__[model_info["name"]](
                num_classes=1000, pretrained="imagenet"
            )

        # Move to GPU if available

    return model


# MARK: Dataset Functions
def get_dataset(
    data_dir: str,
    model_data: dict,
    model: Union[tf.keras.Model, torch.nn.Module],
    batch_size: int = 128,
) -> Union[tf.data.Dataset, torch.utils.data.DataLoader]:
    """Return dataset based on model data."""
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


# MARK: Reps functions
def get_reps(
    model: Union[tf.keras.Model, torch.nn.Module],
    dataset: Union[tf.data.Dataset, torch.utils.data.DataLoader],
    model_data: dict,
    batch_size: int = 128,
) -> np.ndarray:

    if (  # Keras
        "hubModels.json" in model_data["modelFile"]
        or "keras" in model_data["modelFile"]
    ):
        # Maybe add try-catch to fallback on old method
        try:
            reps = model.predict(dataset)
        except:
            reps = extract_reps(model, dataset, model_data, batch_size)

    elif (  # Pytorch
        "pytorch" in model_data["modelFile"]
        or "pretrainedmodels" in model_data["modelFile"]
        or "timm" in model_data["modelFile"]
        or "transformers" in model_data["modelFile"]
    ):
        reps = extract_pytorch_reps(model, dataset, model_data, batch_size)

    return reps


def extract_reps(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    model_data: dict,
    batch_size: int = 128,
) -> np.ndarray:
    """Return representations from model, uses manual batching to avoid memory TF memory leak."""
    # Num batches
    nBatches = len(dataset)

    if "outputIdx" in model_data:
        inpShape = dataset.element_spec.shape
        # Get output size of model
        output_size = model.compute_output_shape(inpShape)[model_data["outputIdx"]][1:]
    elif "origin" in model_data and model_data["origin"] == "keras":
        # Get output size of model
        output_size = model.output.shape[1:]
    else:
        inpShape = dataset.element_spec.shape
        # Get output size of model
        output_size = model.compute_output_shape(inpShape)[1:]

    # Create empty array to store representations
    reps = np.zeros((nBatches * batch_size, *output_size), dtype="float32")

    numImgs = 0
    with click.progressbar(dataset, length=nBatches, label="Extracting reps") as bar:
        for i, batch in enumerate(bar):
            numImgs += len(batch)

            with tf.device("gpu"):
                res = model(batch, training=False)

            if "outputIdx" in model_data.keys():
                # Save representations

                if res[model_data["outputIdx"]].shape[0] == batch_size:
                    reps[i * batch_size : (i + 1) * batch_size] = res[
                        model_data["outputIdx"]
                    ]
                else:
                    reps[
                        i * batch_size : i * batch_size
                        + len(res[model_data["outputIdx"]])
                    ] = res[model_data["outputIdx"]]

            else:
                # Save representations
                if res.shape[0] == batch_size:
                    reps[i * batch_size : (i + 1) * batch_size] = res
                else:
                    reps[i * batch_size : i * batch_size + len(res)] = res

    # Remove empty rows
    reps = reps[:numImgs]

    return reps


def extract_pytorch_reps(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    model_info: dict,
    batch_size: int,
) -> np.ndarray:
    """Return an array of representations from a pytorch model. Careful to
    move data to wherever the model is to handle low GPU memory cases."""
    # Check where model is (CPU or GPU)
    device = next(model.parameters()).device

    # Generate temporary data for shape
    if "shape" in model_info:
        shape = model_info["shape"][:]
    elif "input_size" in model_info:
        shape = model_info["input_size"][:]
    else:
        shape = [3, 224, 224]

    if len(shape) < 4:
        shape.insert(0, 1)
    tmpData = torch.rand(shape)

    # Make sure tmpData is on the same device as model
    tmpData = tmpData.to(device)

    # Figure out output shape
    layer = None
    if "outputLayer" in model_info:  # Need to extract intermediate layer
        if len(model_info["outputLayer"]) == 1:  # Layer within a block
            block = model_info["outputLayer"][1]
            return_nodes = {block: layer}
        else:  # Just a layer
            return_nodes = model_info["outputLayer"]

        layer = model_info["outputLayer"][0]
        model = create_feature_extractor(model, return_nodes=return_nodes)
        outputSize = tuple(model(tmpData)[layer].shape)[1:]
    else:
        outputSize = tuple(model(tmpData).shape)[1:]

    # Create empty array to store representations
    reps = np.zeros((len(dataset.dataset), *outputSize), dtype="float32")

    with click.progressbar(
        dataset, length=len(dataset), label="Extracting reps"
    ) as bar:
        for i, batch in enumerate(bar):
            # Change to float32
            batch = batch.float()

            # Make sure batch is on the same device as model
            batch = batch.to(device)

            # Extract representations
            rep = model(batch)

            if layer != None:
                rep = rep[layer]

            # Save into numpy
            rep = rep.detach().cpu().numpy()
            if "outputIdx" in model_info.keys():
                raise NotImplementedError(
                    "OutputIdx not implemented for pytorch models"
                )
            else:
                reps[i * batch_size : i * batch_size + len(rep)] = rep

    return reps


if __name__ == "__main__":
    cli()
