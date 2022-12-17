import gc
import time

import pandas as pd
import pymysql
import tensorflow as tf
import urllib3

AUTOTUNE = tf.data.experimental.AUTOTUNE


def label_str_to_tensor(labelStr):
    return tf.convert_to_tensor(list(map(float, bytes.decode(labelStr.numpy()).split(','))), dtype=tf.float32)


def load_and_preprocess_image_from_url(url):
    try:
        image = tf.io.decode_image(
            httpclient.request('GET', bytes.decode(url.numpy()).replace("10.0.0.5", "local.ipv4.host")).data,
            channels=3)
        image = tf.image.resize(image, [224, 224])
        image /= 255.0
        return image
    except Exception as e:
        print(url)


def load_and_preprocess_image_from_url_warp(url):
    return tf.py_function(load_and_preprocess_image_from_url, [url], Tout=(tf.float32))


def build_label_data(*args):
    return [label_str_to_tensor(i) for i in args]


def build_label_data_warp(bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label):
    return tf.py_function(build_label_data,
                          [bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label],
                          Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))


def build_dataset_for_test(deepix_train_index, offset, batch_size):
    time_start = time.time()
    data_from_db = pd.read_sql(sql, db_connection, params=[deepix_train_index, deepix_train_index + offset])
    time_end = time.time()
    print('查询sql耗时：', time_end - time_start, 's')
    img_dataset = tf.data.Dataset.from_tensor_slices(data_from_db.img_path.values)
    label_dataset = tf.data.Dataset.from_tensor_slices((data_from_db.bookmark_label.values,
                                                        data_from_db.view_label.values,
                                                        data_from_db.sanity_label.values,
                                                        data_from_db.restrict_label.values,
                                                        data_from_db.x_restrict_label.values))
    img_dataset = img_dataset.map(load_and_preprocess_image_from_url_warp, num_parallel_calls=AUTOTUNE)
    label_dataset = label_dataset.map(build_label_data_warp, num_parallel_calls=AUTOTUNE)
    dataset = tf.data.Dataset.zip((img_dataset, label_dataset))
    #dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.map(_fixup_shape, num_parallel_calls=AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    time_end = time.time()
    del data_from_db
    gc.collect()
    print('构建dataset耗时：', time_end - time_start, 's')
    return dataset


def _fixup_shape(images, labels):
    images.set_shape([None, None, None, 3])
    labels[0].set_shape([None, 10])
    labels[1].set_shape([None, 10])
    labels[2].set_shape([None, 10])
    labels[3].set_shape([None, 3])
    labels[4].set_shape([None, 3])
    return images, labels


sql = '''
select img_path,

           REGEXP_REPLACE(bookmark_label, '\\\\[|\\\\]', '') as bookmark_label ,
              REGEXP_REPLACE(view_label, '\\\\[|\\\\]', '') as view_label,
              REGEXP_REPLACE(sanity_label, '\\\\[|\\\\]', '') as sanity_label,
              REGEXP_REPLACE(restrict_label, '\\\\[|\\\\]', '') as restrict_label,
              REGEXP_REPLACE(x_restrict_label, '\\\\[|\\\\]', '') as x_restrict_label
              

from deepix_data
where illust_id between %s and %s  order by illust_id
'''

db_connection = pymysql.connect(host="local.ipv4.host", user="root", password="Cheerfun.dev", db="deepix")
httpclient = urllib3.PoolManager()

offset = 50000
