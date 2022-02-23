import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds
import PIL
import random


# Get training information
(x_trainRaw, y_trainRaw), (x_testRaw, y_testRaw) = cifar10.load_data()
mean = np.mean(x_trainRaw)
sd = np.std(x_trainRaw)


def make_train_data(shuffle_seed=None, set_seed=False, augment=False):
    """
    Apply ZCA Whitening and Global Contrast Normalization to CIFAR10 dataset
    """
    # Set seed values only if calling program hasn't already, otherwise it will override
    if set_seed:
        seed_value = 0
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    print("Making train data...")
    print("GCN...")
    # Apply global contrast normalization
    x_train = (x_trainRaw - mean) / sd
    x_test = (x_testRaw - mean) / sd
    print("ZCA...")
    # Do ZCA whitening
    x_flat = x_train.reshape(x_train.shape[0], -1)

    vec, val, _ = np.linalg.svd(np.cov(x_flat, rowvar=False))
    prinComps = np.dot(
        vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T)
    )

    x_train = np.dot(x_flat, prinComps).reshape(x_train.shape)
    testFlat = x_test.reshape(x_test.shape[0], -1)
    x_test = np.dot(testFlat, prinComps).reshape(x_test.shape)

    # Convert to one hot vector
    y_train = tf.keras.utils.to_categorical(y_trainRaw, 10)
    y_test = tf.keras.utils.to_categorical(y_testRaw, 10)

    trainData = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    testData = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if augment:
        trainData = (
            trainData.prefetch(tf.data.experimental.AUTOTUNE)
            .shuffle(x_train.shape[0], seed=shuffle_seed)
            .map(augmentData)
            .batch(128)
        )
    else:
        trainData = (
            trainData.prefetch(tf.data.experimental.AUTOTUNE)
            .shuffle(x_train.shape[0], seed=shuffle_seed)
            .batch(128)
        )

    print("Done!")
    return trainData, testData


def augmentData(image, label):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    x = tf.random.uniform((), minval=-5, maxval=5, dtype=tf.dtypes.int64)
    y = tf.random.uniform((), minval=-5, maxval=5, dtype=tf.dtypes.int64)
    image = tfa.image.translate(images=image, translations=[x, y])

    return image, label


def preprocess(imgset):
    """
    Intended for use with baseline only: take a regular image set and apply ZCA whitening and GCN
    Note that a lot of code is copied over from make_train_data because we need info from x_train
    to do consistent preprocessing on the transformed images
    """
    print("Making train data...")

    print("GCN...")
    # Apply global contrast normalization
    x_train = (x_trainRaw - mean) / sd
    imgset = (imgset - mean) / sd
    print("ZCA...")
    # Do ZCA whitening
    x_flat = x_train.reshape(x_train.shape[0], -1)

    vec, val, _ = np.linalg.svd(np.cov(x_flat, rowvar=False))
    prinComps = np.dot(
        vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T)
    )

    testFlat = imgset.reshape(imgset.shape[0], -1)
    imgset = np.dot(testFlat, prinComps).reshape(imgset.shape)

    return imgset


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
            y_predict[100 * index + cur_count] = label.numpy()
            counts[index] += 1
        # Finish once all 10 categories are full
        if all(count == 100 for count in counts):
            break

    print("Done!")
    return x_predict, y_predict


def create_imagenetv2_set(preprocFun, examples=1, outshape=(224, 224)):
    data, info = tfds.load(
        "imagenet_v2",
        split="test",
        with_info=True,
        shuffle_files=False,
        as_supervised=True,
    )
    data = data.take(info.splits["test"].num_examples)
    numClasses = info.features["label"].num_classes
    labelCounts = {label: 0 for label in range(numClasses)}
    imgs = np.empty((numClasses * examples, outshape[0], outshape[1], 3))
    labels = np.empty(numClasses * examples, dtype="uint16")
    idx = 0

    for image, label in tfds.as_numpy(data):
        # if idx == 0:
        #     plt.imshow(image)
        #     print(label)
        if labelCounts[label] < examples:
            # Save image after preprocessing
            image = tf.keras.preprocessing.image.smart_resize(image, outshape)
            image = preprocFun(image)
            imgs[idx] = image
            labels[idx] = label

            labelCounts[label] += 1
            idx += 1

    labels = tf.one_hot(labels, depth=numClasses)
    return imgs, labels


def get_imagenet_set(preprocFun, batch_size, data_dir, slice=None):
    """
    Return ImageNet dataset for testing. Assumes that it all fits in memory.
    """
    split = f"validation{slice}" if slice is not None else "validation"
    dataset = tfds.load(
        "imagenet2012",
        split=split,
        as_supervised=True,
        shuffle_files=False,
        data_dir=data_dir,
    )

    dataset = dataset.map(preprocFun, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_novset_imgs(data_dir, preprocFun=None, batch_size=64):
    """
    Return novel image set images. Assumes that it all fits in memory.
    """
    files = os.listdir(data_dir)
    imgs = np.empty([len(files)] + list(preprocFun.shape))

    for i, file in enumerate(files):
        img = PIL.Image.open(os.path.join(data_dir, file))
        img = np.array(img)
        if preprocFun is not None:
            img = preprocFun(img)
        imgs[i] = img

    imgs = tf.data.Dataset.from_tensors(imgs)
    imgs = imgs.batch(batch_size)
    imgs = imgs.prefetch(tf.data.AUTOTUNE)
    return imgs


class preproc:
    def __init__(
        self,
        shape=(224, 224, 3),
        dtype="float32",
        labels=False,
        numCat=None,
        scale=None,
        offset=None,
        fun=None,
        **kwargs,
    ):
        self.shape = shape
        self.dtype = dtype
        self.fun = fun
        self.scale = scale
        self.offset = offset
        self.numCat = numCat
        self.labels = labels

    def __call__(self, img, label=None):
        # Rescale then cast to correct datatype
        img = tf.keras.preprocessing.image.smart_resize(img, self.shape[:2])
        img = tf.reshape(img, self.shape)
        img = tf.cast(img, self.dtype)

        # Apply override function
        if self.fun is not None:
            img = self.fun(img)

        # Rescale
        if self.scale is not None and self.offset is not None:
            img = tf.math.multiply(img, self.scale)
            img = tf.math.add(img, self.offset)

        if self.labels:
            return img, tf.one_hot(label, self.numCat)
        else:
            return img


def create_cinic10_set(
    dataPath="/data/CINIC10/test", examples=10, dtype="float64"
):
    """
    Return a part of CINIC10 dataset for testing. Each class will have the
    number of examples, global contrast normalized,
    """
    # Prepare statistics
    cats = os.listdir("/data/CINIC10/train")

    # Loop through each category
    images = np.empty((0, 32, 32, 3))
    for cat in cats:
        catImgs = os.listdir("/data/CINIC10/train/" + cat)
        # Shuffle image list
        random.shuffle(catImgs)

        # Loop through each images in category
        for imgName in catImgs[0:100]:
            # Load image
            img = PIL.Image.open("/data/CINIC10/train/" + cat + "/" + imgName)

            # If image is grayscale, convert to RGB
            if len(img.getbands()) == 1:
                img = img.convert("RGB")

            # Convert to numpy array
            img = np.array(img)

            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            # Concatenate to images
            images = np.concatenate((images, img))

    # Calculate dataset statistics
    imgMean = np.mean(images)
    imgStd = np.std(images)

    # ZCA whitening
    images = (images - imgMean) / imgStd
    imgFlat = images.reshape(images.shape[0], -1)
    vec, val, _ = np.linalg.svd(np.cov(imgFlat, rowvar=False))
    prinComps = np.dot(
        vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T)
    )

    # Work on our dataset
    cats = os.listdir(dataPath)

    # Loop through each category
    images = np.empty((0, 32, 32, 3))
    labels = np.empty(0)
    for i, cat in enumerate(cats):
        catImgs = os.listdir(dataPath + "/" + cat)
        # Shuffle image list
        random.shuffle(catImgs)

        # Loop through each images in category
        for imgName in catImgs[0:examples]:
            # Load image
            img = PIL.Image.open(dataPath + "/" + cat + "/" + imgName)

            # If image is grayscale, convert to RGB
            if len(img.getbands()) == 1:
                img = img.convert("RGB")

            # Convert to numpy array
            img = np.array(img)

            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            # Concatenate to images
            images = np.concatenate((images, img))
            labels = np.append(labels, i)

    # # Apply global contrast normalization
    # images = (images - imgMean) / imgStd

    # # Apply ZCA whitening
    # imagesFlat = images.reshape(images.shape[0], -1)
    # images = np.dot(imagesFlat, prinComps).reshape(images.shape)
    # # Cast to dtype
    # images = tf.cast(images, dtype)
    images = preprocess(images)

    # Convert to numpy arrays
    # images = np.array(images)
    labels = np.array(labels)
    return images, labels


def create_imagenet_subset(
    data_dir,
    slice=None,
    examples=10,
    dtype="float64",
    preprocFun=None,
    save_dir=None,
):
    split = f"validation{slice}" if slice is not None else "validation"
    dataset = tfds.load(
        "imagenet2012",
        split=split,
        as_supervised=True,
        shuffle_files=False,
        data_dir=data_dir,
    )

    # Create dictionary for examples
    outImgs = {i: [] for i in range(1000)}
    for example in dataset.take(len(dataset)):
        img, idx = example

        idx = int(idx.numpy())
        # Check if there are enough examples and add
        if len(outImgs[idx]) < examples:
            if preprocFun is not None:
                img = preprocFun(img, idx)
            outImgs[idx].append(img)

        # Check if we have enough examples
        if all([len(outImgs[i]) == examples for i in range(1000)]):
            break

    # Save images
    if save_dir is not None:
        for i in range(1000):
            for j, img in enumerate(outImgs[i]):
                PIL.Image.fromarray(img.numpy()).save(
                    save_dir + f"/{i}_{j}.png"
                )
    return outImgs


if __name__ == "__main__":
    # Check if dataset is deterministic
    # dataset = np.load("../outputs/masterOutput/dataset.npy")
    # dataset2, labels = make_predict_data(
    #     preprocess(x_testRaw), tf.one_hot(y_testRaw, 10)
    # )
    # np.save("../outputs/masterOutput/labels.npy", labels)

    # preproc = tf.keras.applications.mobilenet_v3.preprocess_input
    # data, labels = create_imagenet_set(preprocFun=preproc, examples=10)
    # np.save("../outputs/masterOutput/bigDataset.npy", data)
    # np.save("../outputs/masterOutput/bigLabels.npy", labels)
    # model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3))
    # model.compile(metrics=["top_k_categorical_accuracy"])
    # results = model.evaluate(data, labels)
    # print(results)

    # Test imagenet
    preprocFun = preproc(
        shape=(32, 32, 3),
        dtype=tf.float32,
        # scale=1.0 / 255,
        # offset=0,
        labels=False,
    )

    data = get_novset_imgs("/data/novset", preprocFun)
    np.save("../outputs/masterOutput/novsetDataSmall.npy", data)
    # data = get_imagenet_set(preprocFun, 256)

    # random.seed(2021)
    # dataset, labels = create_cinic10_set(examples=100)
    # np.save("../outputs/masterOutput/cinicData.npy", dataset)
    # np.save("../outputs/masterOutput/cinicLabels.npy", labels)

    # out = create_imagenet_subset(
    #     "/data/tensorflow_datasets", preprocFun=preprocFun
    # )
    # out

    # # Combine all images into array
    # imgs = []
    # labels = []
    # for key, value in out.items():
    #     imgs.extend(value)
    #     labels.append(key)

    # # Convert to array
    # imgs = np.array(imgs)
    # labels = np.array(labels)

    # # Apply global contrast normalization
    # imgs = (imgs - mean) / sd
    # print("ZCA...")
    # # Do ZCA whitening
    # x_flat = imgs.reshape(imgs.shape[0], -1)

    # vec, val, _ = np.linalg.svd(np.cov(x_flat, rowvar=False))
    # prinComps = np.dot(
    #     vec, np.dot(np.diag(1.0 / np.sqrt(val + 0.00001)), vec.T)
    # )

    # imgs = np.dot(x_flat, prinComps).reshape(imgs.shape)

    # np.save("../outputs/masterOutput/imagenetSubsetSmall.npy", imgs)
    # np.save("../outputs/masterOutput/imagenetSubsetLabels.npy", labels)
    # imgs
