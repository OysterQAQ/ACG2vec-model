import base64

import tensorflow as tf

model = tf.saved_model.load('/Volumes/Home/oysterqaq/Desktop/cugan-pro-conservative-up2x_with_tile_tfjs')
pic = open("inputs/test4.png", "rb")

##转成base64后的字符串格式为 b'图片base64字符串'，前面多了 b'，末尾多了 '，所以需要截取一下

#print(model(tf.stack([tf.convert_to_tensor(pic_base64)])))
imgs_map = tf.io.decode_image(tf.io.read_file("inputs/test4.png"), channels=3)


file_path = "output/1333.png"

with open(file_path, "wb") as file:
    binary_data = model.serve(tf.stack([tf.convert_to_tensor(imgs_map)]))
    x = tf.unstack(binary_data, axis=2)
    x=x[0]
    print(x.shape)
    rows = tf.unstack(x, axis=1)
    rows = [tf.concat(tf.unstack(row), axis=0) for row in rows]
    rows = tf.concat(rows, axis=1)
    output_image = tf.io.encode_png(
            rows
        )

    file.write(output_image.numpy())