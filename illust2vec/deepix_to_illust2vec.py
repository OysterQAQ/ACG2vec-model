import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)


model = keras.models.load_model('/path/to/deepix_tf32.h5', compile=False)
outputs = model.get_layer('add_31').output
outputs = GlobalAveragePooling2D()(outputs)
feature_extract_model = keras.Model(inputs=model.input, outputs=outputs)
feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/feature_extract_model")

