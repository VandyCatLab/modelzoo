import numpy as np
import os
import tensorflow as tf
import PIL
import torch
import timm
from typing import Optional, Callable


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

        # using 'Compose' for general pytorch models
        if "trans_params" in model_data:
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
        img = np.array(img)

        if preprocFun is not None:
            img = preprocFun(img)

        imgs[i] = img

    imgs = tf.data.Dataset.from_tensor_slices(imgs)
    imgs = imgs.batch(batch_size)
    imgs = imgs.prefetch(tf.data.AUTOTUNE)

    return imgs
