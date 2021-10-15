import datasets
import tensorflow as tf
import tensorflow_hub as hub

if __name__ == "__main__":
    inputShape = (224, 224, 3)
    modelURL = (
        "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    )

    preprocFun = datasets.preproc(
        shape=(224, 224, 3),
        dtype=tf.float32,
        scale=1.0 / 255,
        offset=0,
        labels=False,
    )
    data = datasets.get_imagenet_set(preprocFun, 256)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=inputShape),
            hub.KerasLayer(modelURL),
        ]
    )

    rep = model.predict(data)
