import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np


def setup_hub_model(info, batch_size, data_dir):
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
        preprocFun, batch_size, data_dir=data_dir
    )
    return model, dataset


def get_reps(model, dataset):
    """Manual batching to avoid memory problems."""
    dataset = dataset.as_numpy_iterator()
    results = []
    for i, batch in enumerate(dataset):
        print(f"-- Working on batch {i}")
        res = model.predict(batch)
        if "outputIdx" in info.keys():
            results += [res[info["outputIdx"]]]
        else:
            results += [res]
    results = np.concatenate(results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get representations from Tensorflow Hub networks using the validation set of ImageNet, intended to be used in HPC"
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
        "--data_dir",
        "-d",
        type=str,
        help="directory of the dataset",
        default="~/tensorflow_datasets",
    )
    parser.add_argument(
        "--models_file",
        "-f",
        type="str",
        help=".json file with the hub model info",
        default="./hubModels.json",
    )
    args = parser.parse_args()

    with open(args.models_file, "r") as f:
        hubModels = json.loads(f.read())

    # Get the model info
    if args.model is not None:
        modelName = args.model
    else:
        modelName = list(hubModels.keys())[args.index]

    print(f"==== Working on model: {modelName} ====")
    model, dataset = setup_hub_model(
        hubModels[modelName], args.batch_size, args.data_dir
    )

    reps = get_reps(model, dataset)
    np.save(f"../outputs/masterOutput/{modelName}-Reps.npy", reps)
