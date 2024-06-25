import os
from typing import Optional, Callable, Union, Tuple
import shutil

import numpy as np
import PIL
import torch
import timm
from torchvision import transforms  # Actually needed
from PIL import Image  # Actually needed
import tensorflow as tf  # NOTE: TF MUST BE IMPORTED AFTER TORCH
import click


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
@click.argument("data_dir", type=str, required=True)
@click.argument("out_dir", type=str, required=True)
@click.option(
    "--n_images", type=int, default=1, help="Number of images per category to pick"
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for picking images, if none, the first image is picked",
)
@click.option("--overwrite", default=False, is_flag=True, help="Overwrite out_dir")
def pickimages(
    data_dir: str,
    out_dir: str,
    n_images: int = 1,
    seed: Union[int, None] = None,
    overwrite: bool = False,
) -> None:
    """
    Copy images from each category in data_dir to out_dir, renaming each to
    match the category name. Useful for picking images from real datasets.
    """
    # Check if out_dir exists
    if os.path.exists(out_dir):
        if overwrite:
            # Get confirmation
            click.echo(f"WARNING THIS DELETS EVERYTHING IN {out_dir}")
            click.confirm(f"Overwrite {out_dir}?", abort=True)
            shutil.rmtree(out_dir)
        else:
            raise ValueError(f"{out_dir} already exists, use --overwrite to overwrite")

    os.makedirs(out_dir)

    # Set seed if available
    if seed is not None:
        rng = np.random.default_rng(int(seed))

    # Get list of categories
    categories = os.listdir(data_dir)
    categories.sort()

    # Loop through categories
    for cat in categories:
        # Get list of images
        cat_dir = os.path.join(data_dir, cat)
        images = os.listdir(cat_dir)
        images.sort()

        # Pick images
        if seed is not None:
            images = rng.choice(images, n_images, replace=False)
        else:
            images = images[:n_images]

        # Copy images
        for img in images:
            img_path = os.path.join(cat_dir, img)
            out_path = os.path.join(out_dir, cat + "--" + img)
            shutil.copy(img_path, out_path)


# MARK: Model Datasets
class preproc:
    def __init__(
        self,
        shape=(224, 224, 3),
        dtype="float32",
        labels=False,
        numCat=None,
        scale=None,
        offset=None,
        preFun=None,
        origin=None,
        trans_params=None,
        **kwargs,
    ):
        self.origin = origin
        self.trans_params = trans_params
        self.shape = shape
        self.dtype = dtype
        self.preFun = preFun
        self.scale = scale
        self.offset = offset
        self.numCat = numCat
        self.labels = labels

    def __call__(
        self,
        img,
        label=None,
    ):
        # Rescale then cast to correct datatype
        img = tf.keras.preprocessing.image.smart_resize(img, self.shape[:2])
        img = tf.reshape(img, self.shape)
        img = tf.cast(img, self.dtype)

        # Apply override function
        if self.preFun is not None:
            if self.origin == "keras":
                proc = self.preFun + "(img)"
                img = eval(proc)
            else:
                img = self.preFun(img)

        # Rescale
        if self.scale is not None and self.offset is not None:
            img = tf.math.multiply(img, self.scale)
            img = tf.math.add(img, self.offset)

        if self.labels:
            return img, tf.one_hot(label, self.numCat)
        else:
            return img


def get_flat_dataset(
    data_dir: str,
    preprocFun: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    batch_size=64,
) -> tf.data.Dataset:
    """
    Return a dataset where all images are from data_dir. Assumes that it all
    fits in memory.
    """
    files = os.listdir(data_dir)
    files.sort()

    # Preprocess one image to see what size it is
    img = PIL.Image.open(os.path.join(data_dir, files[0]))

    # Remove alpha channel if it exists
    if img.mode == "RGBA":
        img = img.convert("RGB")

    img = np.array(img)
    if preprocFun is not None:
        img = preprocFun(img)

    # Preallocate
    nImgs = len(files)
    imgs = np.empty([nImgs] + list(img.shape))
    for i, file in enumerate(files):
        img = PIL.Image.open(os.path.join(data_dir, file))
        # Remove alpha channel if it exists
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # If grayscale, turn image into 3 channel image
        if img.mode == "L":
            img = img.convert("RGB")

        img = np.array(img)

        if preprocFun is not None:
            img = preprocFun(img)

        imgs[i] = img

    imgs = tf.data.Dataset.from_tensor_slices(imgs)
    imgs = imgs.batch(batch_size)
    imgs = imgs.prefetch(tf.data.AUTOTUNE)

    return imgs


def get_pytorch_dataset(
    data_dir: str,
    model_data: dict,
    model: torch.nn.Module,
    batch_size: int = 64,
) -> torch.utils.data.DataLoader:
    """
    Return a dataset where all images are from data_dir and uses torchvision preprocessing.
    Assumes that it all fits in memory.
    """
    files = os.listdir(data_dir)
    files.sort()

    nImgs = len(files)
    if "shape" in model_data:
        imgs = np.empty([nImgs] + list(model_data["shape"]))
    elif "input_size" in model_data:
        imgs = np.empty([nImgs] + list(model_data["input_size"]))
    else:
        imgs = np.empty([nImgs, 3, 224, 224])

    for i, file in enumerate(files):
        img = PIL.Image.open(os.path.join(data_dir, file))

        # Remove alpha channel if it exists
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # If grayscale image, convert to 3 channel image
        if img.mode == "L":
            img = img.convert("RGB")

        # using 'Compose' for general pytorch models
        if "trans_params" in model_data:
            # m, s used for default when normalize values not given, actually needed
            m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
            py_pre = "transforms.Compose(" + model_data["trans_params"] + ")"
            py_preproc = eval(py_pre)
            img = py_preproc(img)
            img = img.unsqueeze(0)
            imgs[i] = img

        # using preprocessing function for transformers
        elif "preproc_func" in model_data:
            if model_data["preproc_func"] != "None":
                py_preproc = eval(py_pre)
                img = py_preproc(img)
                img = img.unsqueeze(0)
            imgs[i] = img

        else:
            config = timm.data.resolve_data_config({}, model=model)
            transform = timm.data.transforms_factory.create_transform(**config)
            img = transform(img).unsqueeze(0)
            imgs[i] = img

    return torch.utils.data.DataLoader(imgs, batch_size=batch_size)


# MARK: Training Dataset
def make_train_data(
    shuffle_seed: int = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Apply ZCA Whitening and Global Contrast Normalization to CIFAR10 dataset
    """
    # Get training dataset info
    (x_trainRaw, y_trainRaw), (x_testRaw, y_testRaw) = (
        tf.keras.datasets.cifar10.load_data()
    )
    mean = np.mean(x_trainRaw)
    sd = np.std(x_trainRaw)

    y_train = np.copy(y_trainRaw)

    # Apply global contrast normalization
    x_train = (x_trainRaw - mean) / sd
    x_test = (x_testRaw - mean) / sd

    # Do ZCA whitening
    x_flat = x_train.reshape(x_train.shape[0], -1)

    vec, val, _ = np.linalg.svd(np.cov(x_flat, rowvar=False))
    prinComps = np.dot(vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T))

    x_train = np.dot(x_flat, prinComps).reshape(x_train.shape)
    testFlat = x_test.reshape(x_test.shape[0], -1)
    x_test = np.dot(testFlat, prinComps).reshape(x_test.shape)

    # Convert to one hot vector
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_testRaw, 10)

    trainData = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    trainData = (
        trainData.prefetch(tf.data.experimental.AUTOTUNE)
        .shuffle(x_train.shape[0], seed=shuffle_seed)
        .batch(128)
    )

    return trainData, testData


def make_predict_data(x, y, dtype=None):
    """
    Curate prediction set with 1000 images, 100 images for
    all 10 categories of CIFAR10
    """
    print("Making test data...")
    counts = [0] * 10
    x_predict = np.empty((1000, 32, 32, 3))
    if dtype is not None:
        x_predict = x_predict.astype(dtype)
    y_predict = np.empty((1000, 10))
    for img, label in zip(x, y):
        index = np.argmax(label)
        cur_count = counts[index]
        if cur_count != 100:
            x_predict[100 * index + cur_count] = img
            y_predict[100 * index + cur_count] = label
            counts[index] += 1
        # Finish once all 10 categories are full
        if all(count == 100 for count in counts):
            break

    print("Done!")
    return x_predict, y_predict


if __name__ == "__main__":
    cli()
