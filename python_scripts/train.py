from typing import Tuple
import os
import random
from itertools import product
import gc

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import click

import datasets


@click.group()
def cli():
    pass


@cli.command()
@click.option("--conv", type=int, default=4, help="Number of convolutional layers")
@click.option("--dense", type=int, default=1, help="Number of dense layers")
@click.option("--augment", default=False, is_flag=True, help="Augment data")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--gpu_id", type=int, default=0, help="GPU ID, -1 for no GPU")
@click.option(
    "--overwrite", default=False, is_flag=True, help="Overwrite existing model"
)
def cnn(
    conv: int = 4,
    dense: int = 1,
    augment: bool = False,
    seed: int = 0,
    gpu_id: int = 0,
    overwrite: bool = False,
):
    """
    Train a single cnn model given conv, dense, augment, and seed.
    """
    if overwrite:
        click.confirm("Are you sure you want to overwrite the model?", abort=True)

    if gpu_id == -1:
        click.echo("Disabling GPU ops")
        tf.config.set_visible_devices([], "GPU")
    else:
        tfDevices = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(tfDevices[gpu_id], "GPU")

    # Check if the model already exists
    if not overwrite and os.path.exists(
        f"../data_storage/models/cnn{seed:02d}_dense{dense}_conv{conv}{'_augment' if augment else ''}"
    ):
        click.echo("Model already exists, skipping training")
        return

    trainData, testData = datasets.make_train_data(shuffle_seed=seed)
    testData = testData.prefetch(tf.data.experimental.AUTOTUNE).batch(128)

    model = make_cnn(
        input_shape=(32, 32, 3),
        output_shape=10,
        conv=conv,
        dense=dense,
        augment=augment,
        seed=seed,
    )
    model.summary()

    model = train(trainData, testData, model, conv, dense, augment, seed)


@cli.command()
@click.option(
    "--conv_range",
    type=int,
    nargs=2,
    help="Inclusive range of number of convolutional layers",
)
@click.option(
    "--dense_range",
    type=int,
    nargs=2,
    help="Inclusive range of number of dense layers",
)
@click.option("--augment", default=False, is_flag=True, help="Augment data")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--gpu_id", type=int, default=0, help="GPU ID, -1 for no GPU")
@click.option(
    "--backwards",
    default=False,
    is_flag=True,
    help="Go through parameter combinations backwards (for multi-gpu)",
)
@click.option(
    "--overwrite", default=False, is_flag=True, help="Overwrite existing model"
)
def multiple(
    conv_range: Tuple[int, int] = [4, 13],
    dense_range: Tuple[int, int] = [1, 10],
    augment: bool = False,
    seed: int = 0,
    gpu_id: int = 0,
    backwards: bool = False,
    overwrite: bool = False,
):
    """
    Train multiple cnn models given ranges of conv and dense. Use backwards to
    go through parameter combinations backwards.
    """
    if overwrite:
        click.confirm(
            "Are you sure you want to overwrite models while training?", abort=True
        )

    if gpu_id == -1:
        click.echo("Disabling GPU ops")
        tf.config.set_visible_devices([], "GPU")
    else:
        tfDevices = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(tfDevices[gpu_id], "GPU")

    # Make ranges
    convRange = range(conv_range[0], conv_range[1] + 1)
    denseRange = range(dense_range[0], dense_range[1] + 1)
    combos = list(product(convRange, denseRange))

    if backwards:
        combos = combos[::-1]

    # Loop through combos
    for conv, dense in combos:
        if not overwrite and os.path.exists(
            f"../data_storage/models/cnn{seed:02d}_dense{dense}_conv{conv}{'_augment' if augment else ''}"
        ):
            click.echo("Model already exists, skipping training")
            continue

        # Reset seeds
        os.environ["PYTHONHASHSEED"] = str(0)
        np.random.seed(0)
        tf.random.set_seed(0)
        random.seed(0)

        trainData, testData = datasets.make_train_data(shuffle_seed=seed)
        testData = testData.prefetch(tf.data.experimental.AUTOTUNE).batch(128)

        model = make_cnn(
            input_shape=(32, 32, 3),
            output_shape=10,
            conv=conv,
            dense=dense,
            augment=augment,
            seed=seed,
        )
        model.summary()

        model = train(trainData, testData, model, conv, dense, augment, seed)

        # Delete model to attempt to preserve gpu memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()


def make_cnn(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    output_shape: int = 10,
    dense: int = 2,
    conv: int = 4,
    augment: bool = False,
    seed: int = 0,
) -> tf.keras.Model:
    """
    Create the All-CNN-C model.
    """
    kernelInit = tf.keras.initializers.HeNormal(seed)
    l2Reg = tf.keras.regularizers.l2(1e-5)

    if conv < 2:
        raise ValueError("Number of convolutional layers must be at least 2")

    # Calculate convolutional layer split
    conv1Convs = int(np.ceil(conv / 2))
    conv2Convs = conv - conv1Convs

    inputs = tf.keras.Input(shape=input_shape, name="input")

    if augment:
        x = tf.keras.layers.RandomFlip(seed=seed, name="augment_flip")(inputs)
        x = tf.keras.layers.RandomTranslation(
            height_factor=0.16,
            width_factor=0.16,
            fill_mode="constant",
            seed=seed,
            name="augment_translate",
        )(x)

        # Add the first convolutional layer
        x = layers.Conv2D(
            96,
            (3, 3),
            padding="same",
            bias_regularizer=l2Reg,
            kernel_regularizer=l2Reg,
            kernel_initializer=kernelInit,
            bias_initializer="zeros",
            activation="relu",
            name="block1_conv1",
        )(x)
    else:
        # Add the first convolutional layer
        x = layers.Conv2D(
            96,
            (3, 3),
            padding="same",
            bias_regularizer=l2Reg,
            kernel_regularizer=l2Reg,
            kernel_initializer=kernelInit,
            bias_initializer="zeros",
            activation="relu",
            name="block1_conv1",
        )(inputs)

    # Add the rest of the first block of convolutional layers
    for i in range(1, conv1Convs):
        x = layers.Conv2D(
            96,
            (3, 3),
            padding="same",
            bias_regularizer=l2Reg,
            kernel_regularizer=l2Reg,
            kernel_initializer=kernelInit,
            bias_initializer="zeros",
            activation="relu",
            name=f"block1_conv{i + 1}",
        )(x)

    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="block1_pool")(x)
    x = layers.Dropout(0.5, name="block1_drop")(x)

    # Add the first convolutional layer of the second block
    x = layers.Conv2D(
        192,
        (3, 3),
        padding="same",
        bias_regularizer=l2Reg,
        kernel_regularizer=l2Reg,
        kernel_initializer=kernelInit,
        bias_initializer="zeros",
        activation="relu",
        name="block2_conv1",
    )(x)

    # Add the rest of the second block of convolutional layers
    for i in range(1, conv2Convs):
        x = layers.Conv2D(
            192,
            (3, 3),
            padding="same",
            bias_regularizer=l2Reg,
            kernel_regularizer=l2Reg,
            kernel_initializer=kernelInit,
            bias_initializer="zeros",
            activation="relu",
            name=f"block2_conv{i + 1}",
        )(x)

    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="block2_pool")(x)
    x = layers.Dropout(0.5, name="block2_drop")(x)

    x = layers.Conv2D(
        192,
        (3, 3),
        padding="valid",
        bias_regularizer=l2Reg,
        kernel_regularizer=l2Reg,
        kernel_initializer=kernelInit,
        bias_initializer="zeros",
        activation="relu",
        name="block3_conv1",
    )(x)

    for i in range(dense):
        x = layers.Conv2D(
            192,
            (1, 1),
            padding="valid",
            bias_regularizer=l2Reg,
            kernel_regularizer=l2Reg,
            kernel_initializer=kernelInit,
            bias_initializer="zeros",
            activation="relu",
            name=f"dense{i + 1}",
        )(x)

    x = layers.Conv2D(
        output_shape,
        (1, 1),
        padding="valid",
        bias_regularizer=l2Reg,
        kernel_regularizer=l2Reg,
        kernel_initializer=kernelInit,
        bias_initializer="zeros",
        activation="relu",
        name="classifier",
    )(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Softmax(name="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.9, clipnorm=500
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(
    trainData,
    testData,
    model,
    conv: int = 4,
    dense: int = 1,
    augment: bool = True,
    seed: int = 0,
):
    """
    Return a trained model using the given data. The finished model is saved
    with a training log. The conv, dense, augment, and seed arguments don't do
    anything except for saving the model with the correct name.
    """

    # Setup learning rate schedule
    def _scheduler(epoch, lr):
        if epoch == 200 or epoch == 250 or epoch == 300:
            return lr * 0.1
        return lr

    lrSchedule = tf.keras.callbacks.LearningRateScheduler(_scheduler)

    # Setup CSV logger
    csvLogger = tf.keras.callbacks.CSVLogger(
        f"../data_storage/models/cnn{seed:02d}_dense{dense}_conv{conv}{'_augment' if augment else ''}.csv",
        append=False,
    )

    model.fit(
        trainData,
        epochs=350,
        validation_data=testData,
        callbacks=[lrSchedule, csvLogger],
    )

    # Save model
    model.save(
        f"../data_storage/models/cnn{seed:02d}_dense{dense}_conv{conv}{'_augment' if augment else ''}",
        save_format="tf",
        include_optimizer=True,
        overwrite=True,
    )

    return model


if __name__ == "__main__":
    # Setup determinism
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)

    cli()
