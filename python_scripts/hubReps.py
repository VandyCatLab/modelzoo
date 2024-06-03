import datasets
import tensorflow as tf

# import tensorflow.core
import tensorflow_hub as hub
import json
import numpy as np
import os
import analysis
import itertools
import pandas as pd
import datetime
import torch

# from torchvision import transforms
import cv2
from torchvision.models.feature_extraction import create_feature_extractor

# import transformers
import timm


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
        print(f"-- Working on batch {i} [{datetime.datetime.now()}]", flush=True)
        numImgs += len(batch)
        res = model.predict(batch)

        if "outputIdx" in info.keys():
            # Save representations

            if res[info["outputIdx"]].shape[0] == batch_size:
                reps[i * batch_size : (i + 1) * batch_size] = res[info["outputIdx"]]
            else:
                reps[i * batch_size : i * batch_size + len(res[info["outputIdx"]])] = (
                    res[info["outputIdx"]]
                )

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


def get_pytorch_reps(model, data_set, info, batch_size):
    """Manual batching to avoid memory problems."""
    # Num batches
    nBatches = len(data_set)
    b = None
    # dataset = torch.utils.data.TensorDataset(dataset)
    if "shape" in info:
        x = info["shape"]
    elif "input_size" in info:
        x = info["input_size"]
    if len(x) < 4:
        x.insert(0, 1)
    temp_data = torch.rand(x)
    if "outputLayer" in info:
        if len(info["outputLayer"]) > 1:
            a = info["outputLayer"][1]
            b = info["outputLayer"][0]
            return_nodes = {a: b}
        else:
            return_nodes = info["outputLayer"]
        model_int = create_feature_extractor(
            model, return_nodes=return_nodes
        )  # dict(layer4 = 'layer4.2.conv2'))
        intermediate_outputs = model_int(temp_data)
        intermediate = intermediate_outputs[b]
    else:
        model_int = model
        # print(f'\n\n\n{temp_data.shape}')
        intermediate = model_int(temp_data)

    if "outputIdx" in info.keys():
        # Get output size of model
        output_size = tuple(intermediate.shape)

    else:
        # Get output size of model
        output_size = tuple(intermediate.shape)
    # Create empty array to store representations
    reps = np.zeros((nBatches * batch_size, *output_size[1:]), dtype="float32")

    numImgs = 0
    for i, batch in enumerate(data_set):
        print(f"-- Working on batch {i} [{datetime.datetime.now()}]", flush=True)
        batch = batch.float()
        numImgs += len(batch)
        # print(f'\n\n\n{batch.shape}')
        res_full = model_int(batch)
        if b != None:
            res_data = res_full[b]
        else:
            res_data = res_full
        res = res_data.detach().cpu().numpy()
        if "outputIdx" in info.keys():
            # Save representations

            if res[info["outputIdx"]].shape[0] == batch_size:
                reps[i * batch_size : (i + 1) * batch_size] = res[info["outputIdx"]]
            else:
                reps[i * batch_size : i * batch_size + len(res[info["outputIdx"]])] = (
                    res[info["outputIdx"]]
                )

        else:
            # Save representations
            if res.shape[0] == batch_size:
                reps[i * batch_size : (i + 1) * batch_size] = res
            else:
                reps[i * batch_size : i * batch_size + len(res)] = res

    # Remove empty rows
    reps = reps[:numImgs]
    return reps
