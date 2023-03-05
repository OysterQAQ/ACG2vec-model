import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling1D
from tensorflow import keras
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


model = keras.models.load_model('/Volumes/Data/oysterqaq/PycharmProjects/ACG2vec-model/deepix/model-resnet_custom_v3.h5', compile=False)
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
style_feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model")

# #
# style_model = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model', compile=False)
# #
# # style_model.summary()
# image = tf.io.decode_image(tf.io.read_file('/Volumes/Data/oysterqaq/Desktop/105802871_p0_master1200.jpeg'),
#                                channels=3)
# image = tf.image.resize(image, [512, 512])
# image /= 255.0
# image = tf.expand_dims(image, axis=0)
# p = style_feature_extract_model.predict(image)
# print(p)

