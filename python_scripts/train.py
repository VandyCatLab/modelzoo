from typing import Tuple
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import click

import datasets


@click.group()
def cli():
    pass


@cli.command()
@click.option("--conv", type=int, default=4, help="Number of convolutional layers")
@click.option("--dense", type=int, default=2, help="Number of dense layers")
@click.option("--augment", default=False, is_flag=True, help="Augment data")
@click.option("--seed", type=int, default=0, help="Random seed")
def all_cnn_c(
    conv: int = 4,
    dense: int = 2,
    augment: bool = False,
    seed: int = 0,
):
    trainData, testData = datasets.make_train_data(shuffle_seed=seed)

    # TODO: Review how necessary this was
    # x_predict = np.array([x for x, _ in testData.as_numpy_iterator()])
    # y_predict = np.array([y for _, y in testData.as_numpy_iterator()])
    # x_predict, y_predict = datasets.make_predict_data(x_predict, y_predict)

    testData = testData.prefetch(tf.data.experimental.AUTOTUNE).batch(128)

    model = make_all_cnn(
        input_shape=(32, 32, 3),
        output_shape=10,
        conv=conv,
        dense=dense,
        augment=augment,
        seed=seed,
    )
    model.summary()

    model.fit(
        trainData,
        epochs=350,
        validation_data=testData,
    )


def make_all_cnn(
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

    x = layers.Conv2D(
        96,
        (3, 3),
        strides=2,
        padding="same",
        bias_regularizer=l2Reg,
        kernel_regularizer=l2Reg,
        kernel_initializer=kernelInit,
        bias_initializer="zeros",
        activation="relu",
        name="block1_pool",
    )(x)
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

    x = layers.Conv2D(
        192,
        (3, 3),
        strides=2,
        padding="same",
        bias_regularizer=l2Reg,
        kernel_regularizer=l2Reg,
        kernel_initializer=kernelInit,
        bias_initializer="zeros",
        activation="relu",
        name="block2_pool",
    )(x)
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


# Scheduler callback
def scheduler(epoch, lr):
    if epoch == 200 or epoch == 250 or epoch == 300:
        return lr * 0.1
    return lr


LR_Callback = LearningRateScheduler(scheduler)


# Trajectory callback
class Trajectory_Callback(Callback):
    """
    Pre: Must define i, x_predict
    """

    def __init__(self, modelName, actDir, predictData):
        super().__init__()
        self.modelName = modelName
        self.actDir = actDir
        self.predictData = predictData

        # Create directory if it doesn't exist
        if not os.path.exists(self.actDir):
            print("Creating directory: " + self.actDir)
            os.makedirs(self.actDir)

    def get_acts(self, model, layer_arr, x_predict):
        """
        Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
        Post: Returns list of activations over x_predict for relevant layers in this particular model instance
        """
        inp = model.input
        acts_list = []

        for layer in layer_arr:
            print("Layer", str(layer))
            out = model.layers[layer].output
            temp_model = tf.keras.Model(inputs=inp, outputs=out)
            # Predict on x_predict, transpose for spearman
            print("Getting activations...")
            acts = temp_model.predict(x_predict)
            acts_list.append(acts)

        return acts_list

    def on_epoch_end(self, epoch, logs=None):
        layer_arr = [-2]
        if epoch in [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            49,
            99,
            149,
            199,
            249,
            299,
            349,
        ]:
            print(
                "\n\nModel:",
                self.modelName,
                " at epoch ",
                str(int(epoch) + 1),
            )
            acts = self.get_acts(self.model, layer_arr, self.predictData)

            np.save(
                self.actDir + "/" + self.modelName + "e" + str(int(epoch) + 1) + ".npy",
                acts,
            )
            print("\n")


# Cut off training if local minimum hit
class Early_Abort_Callback(Callback):
    """
    Pre: abort is set to False at the beginning of each training instance
    """

    def on_epoch_end(self, epoch, logs=None):
        global abort
        if epoch > 100 and logs.get("accuracy") <= 0.8:
            abort = True
            print("Acc:", logs.get("accuracy"))
            self.model.stop_training = True


if __name__ == "__main__":
    # Setup determinism
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)

    cli()
