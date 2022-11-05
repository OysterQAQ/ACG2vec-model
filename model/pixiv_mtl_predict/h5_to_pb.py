import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras import mixed_precision


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_Activation'`
            for the activation and `name + '_BatchNorm'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    return x


# 需要修改精度
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

model = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/00000003_weight.h5', compile=False)
# model.save_weights('/Volumes/Data/oysterqaq/Desktop/00000003_weight.h5')
model.save("/Volumes/Data/oysterqaq/Desktop/label_predict_model")
# model.summary()
# outputs=model.get_layer('tag_predict_avg_pool').output
outputs = model.get_layer('post_bn').output
# outputs=conv2d_bn(outputs, 1536, 1)
outputs = GlobalAveragePooling2D()(outputs)
feature_extract_model = keras.Model(inputs=model.input, outputs=outputs)
# feature_extract_model.summary()
feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/feature_extract_model")


# model.save("")


def export_model_as_float32(temporary_model, checkpoint_path, export_path):
    """
    Hotfix for exporting mixed precision model as float32.
    """
    checkpoint = tf.train.Checkpoint(model=temporary_model)

    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=checkpoint_path, max_to_keep=3
    )

    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    temporary_model.save(export_path, include_optimizer=False)
