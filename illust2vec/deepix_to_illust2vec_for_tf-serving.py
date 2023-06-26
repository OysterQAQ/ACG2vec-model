import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
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




model = keras.models.load_model('/Volumes/Data/oysterqaq/PycharmProjects/ACG2vec-model/pix2score/model-resnet_custom_v3.h5', compile=False)
outputs = model.get_layer('add_43').output
outputs = GlobalAveragePooling2D()(outputs)
feature_extract_model = keras.Model(inputs=model.input, outputs=outputs)

inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
x=Base64DecoderLayer([512,512])(inputs)
#x = tf.keras.layers.Lambda(preprocess_input, name='decode_image_bytes')(inputs)

#model.summary()


x = feature_extract_model(x)
feature_extract_model = keras.Model(inputs=inputs, outputs=x)
feature_extract_model.summary()

import base64


# pic = open("/Volumes/Data/oysterqaq/Desktop/004538_188768.jpg", "rb")
# pic_base64 = base64.urlsafe_b64encode(pic.read())
#
# print(pic_base64)
# ##转成base64后的字符串格式为 b'图片base64字符串'，前面多了 b'，末尾多了 '，所以需要截取一下
#
# print(feature_extract_model(tf.stack([tf.convert_to_tensor(pic_base64)])))

feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/img_semantics_feature_extract_model_f32_base64_input")

