import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("/Volumes/Home/oysterqaq/Desktop/cugan-pro-no-denoise-up2x_with_tile_tfjs") # path to the SavedModel directory

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
with open('/Volumes/Home/oysterqaq/Desktop/cugan-pro-no-denoise-up2x_with_tile.tflite', 'wb') as f:
  f.write(tflite_model)