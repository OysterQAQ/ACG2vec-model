import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling1D
from tensorflow import keras
from tensorflow.keras import mixed_precision

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
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


model = keras.models.load_model('/Volumes/Data/oysterqaq/PycharmProjects/ACG2vec-model/pix2score/model-resnet_custom_v3.h5', compile=False)
gram_matrix_layer=tf.keras.layers.Lambda(gram_matrix)
add = gram_matrix_layer(model.get_layer('add').output)
add_1 = gram_matrix_layer(model.get_layer('add_1').output)
add_2 = gram_matrix_layer(model.get_layer('add_2').output)
add = GlobalMaxPooling1D()(add)
add_1 = GlobalMaxPooling1D()(add_1)
add_2 = GlobalMaxPooling1D()(add_2)
stack=tf.stack([add,add_1,add_2],1)
output=GlobalMaxPooling1D()(stack)
#归一化
#output=tf.math.l2_normalize(output)
style_feature_extract_model = keras.Model(inputs=model.input, outputs=output)
style_feature_extract_model.summary()


inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
x=Base64DecoderLayer([512,512])(inputs)
x=style_feature_extract_model(x)
base64_input_style_feature_extract_model = keras.Model(inputs=inputs, outputs=x)

import base64
# pic = open("/Volumes/Data/oysterqaq/Desktop/004538_188768.jpg", "rb")
# pic_base64 = base64.urlsafe_b64encode(pic.read())

#print(pic_base64)
##转成base64后的字符串格式为 b'图片base64字符串'，前面多了 b'，末尾多了 '，所以需要截取一下

#print(base64_input_style_feature_extract_model(tf.stack([tf.convert_to_tensor(pic_base64)])))
base64_input_style_feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model_base64_input")

#style_feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model")

# #
# style_model = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model', compile=False)
# #
# # style_model.summary()
# image = tf.io.decode_image(tf.io.read_file('/Volumes/Data/oysterqaq/Desktop/004538_188768.jpg'),
#                                channels=3)
# image = tf.image.resize(image, [512, 512])
# image /= 255.0
# image = tf.expand_dims(image, axis=0)
# p = style_feature_extract_model.predict(image)
# print(p)

