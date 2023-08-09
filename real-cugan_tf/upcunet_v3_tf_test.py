import base64

import tensorflow as tf

model = tf.saved_model.load('/Volumes/Home/oysterqaq/Desktop/cugan')
pic = open("inputs/31726597.png", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())


##转成base64后的字符串格式为 b'图片base64字符串'，前面多了 b'，末尾多了 '，所以需要截取一下

#print(model(tf.stack([tf.convert_to_tensor(pic_base64)])))
imgs_map = tf.io.decode_image(tf.io.read_file("inputs/1-1.png"), channels=3)
#input=tf.stack([tf.convert_to_tensor(pic_base64)])
#y = model(tf.expand_dims(imgs_map, axis=0))

file_path = "output/1333.png"

with open(file_path, "wb") as file:
    binary_data = model(tf.stack([tf.convert_to_tensor(pic_base64)])).numpy()
# Example binary data
    file.write(binary_data)