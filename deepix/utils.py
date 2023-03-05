import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model


def print_model_layers(model):
    for layer in model.layers:
        print(layer.name)
        print(layer.output)


def ouput_model_arch_to_image(model, path):
    plot_model(model, to_file=path, show_shapes=True)


def pridict(model, path):


    image = tf.io.decode_image(tf.io.read_file(path),
                               channels=3)
    image = tf.image.resize(image, [299, 299])
    image /= 255.0
    image = tf.expand_dims(image, axis=0)
    p = model.predict(image)
    print(p)
    a = p[-1]
    ind = np.argpartition(a[0], -10)[-10:]


