import json

import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling1D
from tensorflow import keras
from tensorflow.keras import mixed_precision
from deepix_train import build_model
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)



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
config_name = 'deepix_v4'
config_path = 'config/' + config_name + '.json'
with open(config_path, 'r') as load_f:
    model_config = json.load(load_f)
model = build_model(model_config)

trained_model = tf.keras.models.load_model('/Volumes/Home/oysterqaq/Desktop/0182.ckpt',compile=False)
model.set_weight(trained_model.get_weights())
model.load_weights('/Volumes/Home/oysterqaq/PycharmProjects/ACG2vec-model/pix2score/model_weight_history/deepix_v4/00000181.h5')
image = tf.io.decode_image(tf.io.read_file('/Volumes/Home/oysterqaq/Downloads/109281305_p0_master1200.jpg'),
                               channels=3)
image = tf.image.resize(image, [224, 224])
image /= 255.0
image = tf.expand_dims(image, axis=0)
p = model.predict(image)
print(p)

export_model_as_float32(model,'/Volumes/Home/oysterqaq/PycharmProjects/ACG2vec-model/pix2score/model_weight_history/deepix_v4/00000181.h5','/Volumes/Data/oysterqaq/Desktop/pix2score')

# model.load_weights('/Volumes/Data/oysterqaq/Desktop/00000111.h5')
#
# pix2score = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/pix2score', compile=False)
#
# inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
# x=Base64DecoderLayer([224,224])(inputs)
# x=pix2score(x)
# base64_model = keras.Model(inputs=inputs, outputs=x)
#
# base64_model.save("/Volumes/Data/oysterqaq/Desktop/pix2score_base64_input")




pix2score = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/pix2score_base64_input', compile=False)
#
# style_model.summary()
import base64
pic = open("/Volumes/Data/oysterqaq/Desktop/107776952_p0_square1200.jpg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())

print(pix2score(tf.stack([tf.convert_to_tensor(pic_base64)])))
