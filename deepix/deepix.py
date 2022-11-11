import tensorflow as tf
from keras import backend
from keras import layers
from keras import models
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def create_acg2vec_pixiv_predict_model(pretrained_model_path):
    deepdanbooru_pretrained_model = load_deepdanbooru_pretrained_model(pretrained_model_path)
    x = deepdanbooru_pretrained_model.output
    x = stack2(x, 512, 10, stride1=1, name= 'deepix_conv5')
    x = layers.GlobalAveragePooling2D(name='deepix_avg_pool')(x)
    bookmark_predict = _output_layer(x, 'bookmark_predict', 10, 'softmax')
    view_predict = _output_layer(x, 'view_predict', 10, 'softmax')
    sanity_predict = _output_layer(x, 'sanity_predict', 10, 'softmax')
    restrict_predict = _output_layer(x, 'restrict_predict', 3, 'softmax')
    x_restrict_predict = _output_layer(x, 'x_restrict_predict', 3, 'softmax')
    #tag_predict = _output_layer(x, 'tag_predict', 10240, 'sigmoid')
    output = [bookmark_predict, view_predict, sanity_predict, restrict_predict, x_restrict_predict,
              #tag_predict
              ]
    # Create model.
    model = models.Model(inputs=deepdanbooru_pretrained_model.input,  outputs=output, name='acg2vec_pixiv_predict')

    return model


def load_deepdanbooru_pretrained_model(path):
    deepdanbooru_pretrained_model = tf.keras.models.load_model(path,compile=False)
    outputs = deepdanbooru_pretrained_model.get_layer('activation_96').output
    feature_extract_model = tf.keras.Model(inputs=deepdanbooru_pretrained_model.input, outputs=outputs)
    return feature_extract_model


def _output_layer(x, output_name, output_dim, output_activation):
    #x = layers.GlobalAveragePooling2D(name=output_name + '_avg_pool')(x)
    x = layers.Dense(output_dim, activation=output_activation, name=output_name, dtype="float32")(x)
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x
