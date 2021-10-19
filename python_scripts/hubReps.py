import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np


def setup_hub_model(info, batch_size):
    # Create model
    shape = info["shape"] if "shape" in info.keys() else [224, 224, 3]
    inp = tf.keras.Input(shape=shape)
    out = hub.KerasLayer(info["url"])(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)

    # Create dataset
    preprocFun = datasets.preproc(**info, labels=False)
    dataset = datasets.get_imagenet_set(preprocFun, batch_size)
    return model, dataset


if __name__ == "__main__":
    with open("./hubModels.json", "r") as f:
        hubModels = json.loads(f.read())

    for name, info in hubModels.items():
        print(f"==== Working on model: {name} ====")
        model, dataset = setup_hub_model(info, 64)
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
        np.save(f"'../outputs/masterOutput/{name}-Reps.npy", results)

    # print(hubModels)
    # inputShape = (224, 224, 3)
    # modelURL = (
    #     "https://tfhub.dev/emilutz/vgg19-block5-conv2-unpooling-encoder/1"
    # )

    # preprocFun = datasets.preproc(
    #     labels=False,
    # )
    # data = datasets.get_imagenet_set(preprocFun, 256)

    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=inputShape),
    #         hub.KerasLayer(modelURL),
    #     ]
    # )

    # rep = model.predict(data)
