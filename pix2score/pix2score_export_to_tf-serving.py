import json

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling1D
from tensorflow import keras
from tensorflow.keras import mixed_precision
from deepix_train import build_model
from pix2score.utils import ouput_model_arch_to_image

policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)
np.set_printoptions(suppress=True)

class Base64DecoderLayer(tf.keras.layers.Layer):
  """
  Convert a incoming base 64 string into an bitmap with rgb values between 0 and 1
  target_size e.g. [width,height]
  """

  def __init__(self, target_size):
    self.target_size = target_size
    super(Base64DecoderLayer, self).__init__()

  def byte_to_img(self, byte_tensor):
    # base64 decoding id done by tensorflow serve, when using b64 json
    byte_tensor = tf.io.decode_base64(byte_tensor)
    imgs_map = tf.io.decode_image(byte_tensor,channels=3)
    imgs_map.set_shape((None, None, 3))
    img = tf.image.resize(imgs_map, self.target_size)
    img = tf.cast(img, dtype=tf.float32) / 255
    return img

  def call(self, input, **kwargs):
    with tf.device("/cpu:0"):
      imgs_map = tf.map_fn(self.byte_to_img, input, dtype=tf.float32)
    return imgs_map


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


def check_and_initialize_nan_weights(model):
    for layer in model.layers:
        for weight in layer.weights:
            try:
                nan_indices = tf.debugging.check_numerics(weight, "NaN detected in weight")
            except Exception as e:
                # assert "Checking b : Tensor had NaN values" in e.message
                print("NaN detected in weight:", weight.name)
                # layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)), \
                #                    layer.bias_initializer(shape=np.asarray(layer.bias.shape))])
                # 对 NaN 权重进行初始化
                nan_mask = tf.math.is_nan(weight)
                weight.assign(tf.where(nan_mask, tf.ones_like(weight), weight))


config_name = 'deepix_v4'
config_path = 'config/' + config_name + '.json'
with open(config_path, 'r') as load_f:
    model_config = json.load(load_f)
model = build_model(model_config)
for layer in model.layers:
    try:
        layer.summary()
    except Exception as e:
        continue
# trained_model = tf.keras.models.load_model('/Volumes/Home/oysterqaq/Desktop/0182.ckpt',compile=False)
# model.set_weight(trained_model.get_weights())
model.load_weights('/Volumes/Home/oysterqaq/Downloads/00000200.h5')
# check_and_initialize_nan_weights(model)
image = tf.io.decode_image(tf.io.read_file('/Volumes/Home/oysterqaq/Downloads/4.jpeg'),
                           channels=3)
image_2 = tf.io.decode_image(tf.io.read_file('/Volumes/Home/oysterqaq/Downloads/3.jpeg'),
                           channels=3)
image = tf.image.resize(image, [224, 224])
image_2 = tf.image.resize(image_2, [224, 224])
image /= 255.0
image_2 /= 255.0
image = tf.stack([ image,image_2])
# layer_outputs = [image]  # 存储每一层的输出
# for layer in model.layers:
#
#     if hasattr(layer, 'layers'):
#       for l in layer.layers:
#         try:
#           output = l(layer_outputs[-1])
#           layer_outputs.append(output)
#         except Exception as e:
#           print(layer_outputs)
#
#     else:
#       output = layer(layer_outputs[-1])
#       layer_outputs.append(output)

# bn.set_moving_variance(bn.moving_variance_initializer(shape=np.asarray(bn.moving_variance.shape)))
# if hasattr(bn, 'moving_mean_initializer') and \
#         hasattr(bn, 'moving_variance_initializer'):
#     bn.set_weights([bn.kernel_initializer(shape=np.asarray(bn.kernel.shape)),bn.bias_initializer(shape=np.asarray(bn.bias.shape))])
#ouput_model_arch_to_image(res, '/Volumes/Home/oysterqaq/Desktop/res.jpg')
# res = model.get_layer('resnet101v2')
# bn = res.get_layer('conv5_block3_1_bn')
# post_bn = res.get_layer('post_bn')
# bn_2 = res.get_layer('conv5_block3_2_bn')
#bn_2.set_weights([bn_2.weights[0],bn_2.weights[1],bn_2.moving_mean_initializer(shape=np.asarray(bn_2.moving_mean.shape)),bn_2.moving_variance_initializer(shape=np.asarray(bn_2.moving_variance.shape))])
#post_bn.set_weights([post_bn.weights[0],post_bn.weights[1],post_bn.moving_mean_initializer(shape=np.asarray(post_bn.moving_mean.shape)),post_bn.moving_variance_initializer(shape=np.asarray(post_bn.moving_variance.shape))])
# input = tf.keras.Input(shape=(224, 224, 3), name="input")
# sub = keras.Model(inputs=res.input, outputs=res.get_layer('conv5_block3_out').output)
# output = sub(input)
# submodel = keras.Model(inputs=input, outputs=output)
# submodel.summary()
# print(submodel(image))

p = model.predict(image)
print(p)

# #导出
export_model_as_float32(model,'/Volumes/Home/oysterqaq/Desktop/00000200.h5','/Volumes/Home/oysterqaq/Desktop/pix2score')
pix2score = keras.models.load_model('/Volumes/Home/oysterqaq/Desktop/pix2score', compile=False)
inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
x=Base64DecoderLayer([224,224])(inputs)
x=pix2score(x)
base64_model = keras.Model(inputs=inputs, outputs=x)
base64_model.save("/Volumes/Home/oysterqaq/Desktop/pix2score_base64_input")


# pix2score = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/pix2score_base64_input', compile=False)
# #
# # style_model.summary()
# import base64
# pic = open("/Volumes/Home/oysterqaq/Desktop/110058474_p0_master1200.jpg", "rb")
# pic_base64 = base64.urlsafe_b64encode(pic.read())
# #
# print(base64_model(tf.stack([tf.convert_to_tensor(pic_base64)])))
