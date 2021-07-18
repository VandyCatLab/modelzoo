import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import cifar10
import analysis, datasets
import pandas as pd
import os


def yield_transforms(transform, model, layer_idx, dataset):
    """
    Yield transformed representations successfully from the dataset after 
    passing through the model to a certain layer_idx. Transforms control what 
    is yielded.

    Transform information:
        - 'reflect' yields only one representations
        - 'translate' yields a number a list of four representations equal to 
        the smaller dimension, translating the image in all four directions.
        - 'color' yields 51 representations after modifying the color channels
        with multiples of the PCA.
        - 'zoom' yields a number of representations equal to half the smaller
        dimension, zooming into the image.
    """
    # Set model to output reps at selected layer
    inp = model.input
    layer = model.layers[layer_idx]
    out = layer.output
    model = Model(inputs=inp, outputs=out)

    # Get reps for originals
    rep1 = model.predict(dataset, verbose=0)

    # Reflect and shift don't require remaking the dataset
    if transform == "reflect":
        print(" - Yielding 1 version.")
        transDataset = np.flip(dataset, axis=2)
        rep2 = model.predict(transDataset, verbose=0)
        yield 0, rep1, rep2

    elif transform == "translate":
        versions = (
            dataset.shape[1]
            if dataset.shape[1] <= dataset.shape[2]
            else dataset.shape[2]
        )

        print(f" - Yielding {versions} versions.")
        for v in tf.range(versions):
            print(f"Translating {v} pixels.", flush=True)
            # Generate transformed imageset
            transImg = tfa.image.translate(dataset, [v, 0])  # Right
            transImg = tf.concat(
                (transImg, tfa.image.translate(dataset, [-v, 0])), axis=0
            )  # Left
            transImg = tf.concat(
                (transImg, tfa.image.translate(dataset, [0, v])), axis=0
            )  # Down
            transImg = tf.concat(
                (transImg, tfa.image.translate(dataset, [0, -v])), axis=0
            )  # Up

            # Get average of all 4 directions
            rep2 = model.predict(transImg, verbose=0)
            # Split back out
            rep2 = tf.split(rep2, 4)

            yield v, rep1, rep2

    elif transform == "color":
        versions = 51
        alphas = np.linspace(-10, 10, versions)

        print("Do PCA on raw training set to get eigenvalues and -vectors")
        x_train = datasets.preprocess(datasets.x_trainRaw)
        x_train = x_train.reshape(x_train.shape[0], -1)
        cov = np.cov(x_train.T)
        values, vectors = np.linalg.eigh(cov)

        print(f" - Yielding {versions} versions.")
        for v in range(versions):
            transImg = dataset

            # Add multiple of shift
            alpha = alphas[v]
            print(f"Color shifting alpha: {alpha}.")
            change = np.dot(vectors, values * alpha)
            transImg[:, :, :, 0] += change[0]
            transImg[:, :, :, 1] += change[1]
            transImg[:, :, :, 2] += change[2]

            rep2 = model.predict(transImg, verbose=0)

            yield v, rep1, rep2

    elif transform == "zoom":
        smallDim = (
            dataset.shape[1]
            if dataset.shape[1] <= dataset.shape[2]
            else dataset.shape[2]
        )
        versions = smallDim // 2

        print(f" - Yielding {versions} versions.")
        for v in range(versions):
            print(f"Zooming {v} pixels.")
            # Generate transformed imageset
            transformed_dataset = dataset[
                :, v : smallDim - v, v : smallDim - v, :
            ]
            transformed_dataset = tf.image.resize(
                transformed_dataset, (smallDim, smallDim)
            )

            print(" - Now correlating...")
            rep2 = model.predict(transformed_dataset, verbose=0)

            yield v, rep1, rep2


def dropoutBaseline():
    """
    Another baseline analysis but the same image with different dropouts.
    """
    raise NotImplementedError


"""
Sanity check/Visualization functions
"""


def visualize_transform(transform, depth, img_arr):
    if transform == "reflect":
        # Depth doesn't matter
        transformed = np.flip(img_arr, axis=1)
        plt.imshow(transformed)

    elif transform == "color":
        # Depth = alpha value * 1000 (ints -100 : 100)
        alpha = depth / 1000
        img_reshaped = img_arr.reshape(-1, 3)
        cov = np.cov(img_reshaped.T)
        values, vectors = np.linalg.eig(cov)
        change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
        new_img = np.round(img_arr + change)
        transformed = np.clip(new_img, a_min=0, a_max=255, out=None)
        plt.imshow(transformed)

    elif transform == "zoom":
        dim = img_arr.shape[0]
        v = depth
        img = Image.fromarray(img_arr)
        new_img = img.crop((v, v, dim - v, dim - v))
        new_img = new_img.resize((dim, dim), resample=Image.BICUBIC)
        transformed = img_to_array(new_img)
        plt.imshow(transformed)

    elif transform == "shift":
        dim = img_arr.shape[0]
        v = depth
        empty = np.zeros((dim, dim, 3))
        up_transformed = np.concatenate(
            [img_arr[v:dim, :, :], empty[0:v, :, :]]
        )
        down_transformed = np.concatenate(
            [empty[0:v, :, :], img_arr[0 : dim - v, :, :]]
        )
        left_transformed = np.concatenate(
            [img_arr[:, v:dim, :], empty[:, 0:v, :]], axis=1
        )
        right_transformed = np.concatenate(
            [empty[:, 0:v, :], img_arr[:, 0 : dim - v, :]], axis=1
        )

        _, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(up_transformed)
        axarr[0, 1].imshow(down_transformed)
        axarr[1, 0].imshow(left_transformed)
        axarr[1, 1].imshow(right_transformed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform baseline analysis, intended to be used in HPC"
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        help="type of analysis to run",
        choices=["translate", "zoom", "reflect", "color", "dropout"],
    )
    parser.add_argument(
        "--model_index",
        "-i",
        type=int,
        help="model index to select weight and shuffle seeds",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, help="shuffle seed of the main model"
    )
    parser.add_argument(
        "--weight_seed", type=int, help="weight seed of the main model"
    )
    parser.add_argument(
        "--model_seeds",
        type=str,
        default="../outputs/masterOutput/modelSeeds.csv",
        help="file location for csv file with model seeds",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../outputs/masterOutput/dataset.npy",
        help="npy file path for the image dataset to use for analysis",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="../outputs/masterOutput/models",
        help="directory for all of the models",
    )
    parser.add_argument(
        "--layer_index",
        "-l",
        type=analysis._split_comma_str,
        default="layer indices, split by a comma",
    )
    args = parser.parse_args()

    # Load model
    model, modelName, _ = analysis.get_model_from_args(args)
    model = analysis.make_allout_model(model)

    # Load dataset
    print("Loading dataset", flush=True)
    dataset = np.load(args.dataset_file)
    print(f"dataset shape: {dataset.shape}", flush=True)

    # Prep analysis functions
    preprocFuns = [
        analysis.preprocess_rsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]

    basePath = "../outputs/masterOutput/baseline/"

    if args.analysis in ["translate", "zoom", "reflect", "color"]:
        simFunNames = [fun.__name__ for fun in simFuns]
        for layer in args.layer_index:
            # Get transforms generators
            transforms = yield_transforms(
                args.analysis, model, int(layer), dataset
            )

            # Create dataframe
            if args.analysis == "translate":
                directions = (
                    ["right"] * len(simFuns)
                    + ["left"] * len(simFuns)
                    + ["down"] * len(simFuns)
                    + ["up"] * len(simFuns)
                )
                colNames = [
                    f"{fun}-{direct}"
                    for direct, fun in zip(
                        directions, [fun.__name__ for fun in simFuns] * 4
                    )
                ]
                simDf = pd.DataFrame(columns=["version"] + colNames)
            else:
                simDf = pd.DataFrame(columns=["version"] + simFunNames)

            # Get similarity measure per transform
            for v, rep1, rep2 in transforms:
                if args.analysis == "translate":
                    # Calculate similarity for each direction
                    simDirs = []
                    for rep in rep2:
                        rep = np.array(rep)
                        simDirs += [
                            analysis.multi_analysis(
                                rep1, rep, preprocFuns, simFuns
                            )
                        ]

                    # Save all directions
                    tmp = [v]
                    for dic in simDirs:
                        tmp += [dic[key] for key in dic.keys()]

                    simDf.loc[len(simDf.index)] = tmp
                else:
                    sims = analysis.multi_analysis(
                        rep1, rep2, preprocFuns, simFuns
                    )
                    simDf.loc[len(simDf.index)] = [v] + [
                        sims[fun] for fun in sims.keys()
                    ]

            # Save
            outPath = os.path.join(
                basePath, f"{modelName[0:-3]}l{layer}-{args.analysis}.csv"
            )
            simDf.to_csv(outPath, index=False)

    elif args.analysis == "dropout":
        raise NotImplementedError

