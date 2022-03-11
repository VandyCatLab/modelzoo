import sys
import numpy as np
import pickle
import os
import random
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Activation,
    Dense,
    Layer,
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras import backend as K
import analysis, datasets

# Make GPU training deterministic
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(0)

# Function to make models
def init_all_cnn_c(seed: int):

    model = Sequential()
    model.add(
        Dropout(0.2, input_shape=(32, 32, 3))
    )  # input shape from keras cifar10 example
    model.add(
        Conv2D(
            96,
            (3, 3),
            input_shape=(32, 32, 3),
            padding="same",
            bias_regularizer=l2(1e-5),
            kernel_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            96,
            (3, 3),
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            96,
            (3, 3),
            strides=2,
            padding="same",
            bias_regularizer=l2(1e-5),
            kernel_regularizer=l2(1e-5),
            bias_initializer="zeros",
            activation="relu",
        )
    )

    model.add(Dropout(0.5))
    model.add(
        Conv2D(
            192,
            (3, 3),
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            192,
            (3, 3),
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            192,
            (3, 3),
            strides=2,
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )

    model.add(Dropout(0.5))
    model.add(
        Conv2D(
            192,
            (3, 3),
            padding="valid",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            192,
            (1, 1),
            padding="valid",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            10,
            (1, 1),
            padding="valid",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Activation("softmax"))

    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9, clipnorm=500),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def krieg_all_cnn_c(seed: int):
    """
    Implement the Kriegeskorte version of All_CNN_C, differences from the
    perfect implementation from the original paper commented out.
    """

    model = Sequential()
    # model.add(Dropout(0.2, input_shape=(32, 32, 3), seed=0)) # No initial dropout
    model.add(
        Conv2D(
            96,
            (3, 3),
            input_shape=(32, 32, 3),
            padding="same",
            bias_regularizer=l2(1e-5),
            kernel_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            96,
            (3, 3),
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            96,
            (3, 3),
            strides=2,
            padding="same",
            bias_regularizer=l2(1e-5),
            kernel_regularizer=l2(1e-5),
            bias_initializer="zeros",
            activation="relu",
        )
    )

    model.add(Dropout(0.5))  # No dropout described but code has it
    model.add(
        Conv2D(
            192,
            (3, 3),
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            192,
            (3, 3),
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            192,
            (3, 3),
            strides=2,
            padding="same",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )

    model.add(Dropout(0.5))  # No dropout described but code has it
    model.add(
        Conv2D(
            192,
            (3, 3),
            padding="valid",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            192,
            (1, 1),
            padding="valid",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            10,
            (1, 1),
            padding="valid",
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5),
            kernel_initializer=he_normal(seed),
            bias_initializer="zeros",
            activation="relu",
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Activation("softmax"))

    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9, clipnorm=500),
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
                self.actDir
                + "/"
                + self.modelName
                + "e"
                + str(int(epoch) + 1)
                + ".npy",
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
    parser = argparse.ArgumentParser(
        description="Trains a single CNN, hopefully to completion."
    )
    parser.add_argument(
        "--train_differences",
        "-t",
        type=str,
        choices=["seed", "item", "category"],
        help="type of training difference to use",
    )
    parser.add_argument(
        "--shuffle_seed",
        "-s",
        type=int,
        help="seed that dictates the randomization of the dataset order",
    )
    parser.add_argument(
        "--weight_seed",
        "-w",
        type=int,
        help="seed that dictates the random initialization of weights",
    )
    parser.add_argument(
        "--model_index",
        "-i",
        type=int,
        help="model parameter index that dictate either the shuffle and weight seeds or the data seed",
    )
    args = parser.parse_args()

    # Make GPU training deterministic
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(0)

    # Set seeds
    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)

    if args.train_differences == "seed":
        if args.model_index is not None:
            print(f"Using model index: {args.model_index}")
            # Create list
            nums = np.arange(0, 10)
            weights, shuffles = np.meshgrid(nums, nums)

            weights = np.expand_dims(weights.flatten(), 1)
            shuffles = np.expand_dims(shuffles.flatten(), 1)

            modelSeeds = np.concatenate((weights, shuffles), 1)

            weightSeed, shuffleSeed = modelSeeds[args.model_index, :]
        elif args.shuffle_seed is not None and args.weight_seed is not None:
            print(
                f"Using shuffle seed: {args.shuffle_seed} and weight seed: {args.weight_seed}"
            )
            weightSeed = args.weight_seed
            shuffleSeed = args.shuffle_seed

        else:
            raise ValueError(
                "Missing either both shuffle and weight seed or a model index."
            )

        # Prepare data
        trainData, testData = datasets.make_train_data(
            shuffle_seed=shuffleSeed, augment=True
        )
        x_predict, y_predict = datasets.make_predict_data(
            testData[0], testData[1]
        )

        # Create model
        model = krieg_all_cnn_c(seed=weightSeed)

        # Set flag to true if converges to local min
        abort = False
        trajCallback = Trajectory_Callback(
            modelName=f"w{weightSeed}s{shuffleSeed}",
            actDir="../outputs/masterOutput/representations",
            predictData=x_predict,
        )
        history = model.fit(
            trainData,
            epochs=350,
            verbose=2,
            validation_data=testData.prefetch(
                tf.data.experimental.AUTOTUNE
            ).batch(128),
            callbacks=[LR_Callback, trajCallback],
        )

        if not abort:
            print(
                f'Saving, final validation acc: {history.history["val_accuracy"][-1]}'
            )
            save_model(
                model,
                "../outputs/masterOutput/models/w"
                + str(weightSeed)
                + "s"
                + str(shuffleSeed)
                + ".pb",
            )
        else:
            print(
                f'Stuck at local minimum, final validation acc: {history.history["val_accuracy"][-1]}'
            )
            exit(1)
    elif args.train_differences == "item":
        print("Training on item differences")
        trainData, testData = datasets.make_train_data(
            shuffle_seed=2022, augment=True, data_seed=args.model_index
        )
        x_predict = np.array([x for x, _ in testData.as_numpy_iterator()])
        y_predict = np.array([y for _, y in testData.as_numpy_iterator()])
        x_predict, y_predict = datasets.make_predict_data(x_predict, y_predict)

        # Create model
        model = krieg_all_cnn_c(seed=2022)

        # Set flag to true if converges to local min
        abort = False
        trajCallback = Trajectory_Callback(
            modelName=f"model{args.model_index}",
            actDir="../outputs/masterOutput/representations/item",
            predictData=x_predict,
        )
        history = model.fit(
            trainData,
            epochs=350,
            verbose=2,
            validation_data=testData.prefetch(
                tf.data.experimental.AUTOTUNE
            ).batch(128),
            callbacks=[LR_Callback, trajCallback],
        )

        if not abort:
            print(
                f'Saving, final validation acc: {history.history["val_accuracy"][-1]}'
            )
            save_model(
                model,
                "../outputs/masterOutput/models/itemDiff/model"
                + str(args.model_index)
                + ".pb",
            )
        else:
            print(
                f'Stuck at local minimum, final validation acc: {history.history["val_accuracy"][-1]}'
            )
            exit(1)
    elif args.train_differences == "category":
        print("Training on category differences")
        trainData, testData = datasets.make_train_data(
            shuffle_seed=2022,
            augment=True,
            data_seed=args.model_index,
            cat_weighting=True,
        )
        x_predict, y_predict = datasets.make_predict_data(
            testData[0], testData[1]
        )

        # Create model
        model = krieg_all_cnn_c(seed=2022)

        # Set flag to true if converges to local min
        abort = False
        trajCallback = Trajectory_Callback(
            modelName=f"model{args.model_index}",
            actDir="../outputs/masterOutput/representations/cat",
            predictData=x_predict,
        )
        history = model.fit(
            trainData,
            epochs=350,
            verbose=2,
            validation_data=testData.prefetch(
                tf.data.experimental.AUTOTUNE
            ).batch(128),
            callbacks=[LR_Callback, trajCallback],
        )

        if not abort:
            print(
                f'Saving, final validation acc: {history.history["val_accuracy"][-1]}'
            )
            save_model(
                model,
                "../outputs/masterOutput/models/catDiff/model"
                + str(args.model_index)
                + ".pb",
            )
        else:
            print(
                f'Stuck at local minimum, final validation acc: {history.history["val_accuracy"][-1]}'
            )
            exit(1)

    else:
        print("No training difference argument given, treating as main")
