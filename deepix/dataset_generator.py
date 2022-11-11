import time

import pandas as pd
import redis
import sqlalchemy
import tensorflow as tf
import urllib3

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 数据集构建
# 获取索引
redis_index_key = 'deepix_illust_index'
redis_epoch_key = 'deepix_epoch_index'
redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
sql = '''
select illust_id,img_path,

           REGEXP_REPLACE(bookmark_label, '\\\\[|\\\\]', '') as bookmark_label ,
              REGEXP_REPLACE(view_label, '\\\\[|\\\\]', '') as view_label,
              REGEXP_REPLACE(sanity_label, '\\\\[|\\\\]', '') as sanity_label,
              REGEXP_REPLACE(restrict_label, '\\\\[|\\\\]', '') as restrict_label,
              REGEXP_REPLACE(x_restrict_label, '\\\\[|\\\\]', '') as x_restrict_label
              -- ,REGEXP_REPLACE(label, '\\\\[|\\\\]', '') as label

from deepix_data 
where illust_id < %s  order by illust_id desc limit %s 
'''
engine = sqlalchemy.create_engine('mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/deepix?charset=utf8')
offset = 64000
httpclient = urllib3.PoolManager()
min_deepix_train_index = 20000000
max_deepix_train_index = 90000000


# deepix_train_index = 50000000
def generate_data_from_db():
    #deepix_train_index = int(redis_conn.get(redis_index_key))
    deepix_train_index = max_deepix_train_index if redis_conn.get(redis_index_key)  is None else int(redis_conn.get(redis_index_key))
    if (deepix_train_index < min_deepix_train_index):
        deepix_train_index = max_deepix_train_index
        redis_conn.set(redis_index_key, deepix_train_index)
        epoch_index = int(redis_conn.get(redis_epoch_key))
        redis_conn.set(redis_epoch_key, epoch_index + 1)
    while deepix_train_index > min_deepix_train_index:
        time_start = time.time()
        data_from_db = pd.read_sql(sql, engine, params=[deepix_train_index, offset])
        time_end = time.time()
        print('\n查询sql耗时：', time_end - time_start, 's')
        print('\n当前训练到' + str(deepix_train_index))
        deepix_train_index = int(data_from_db.illust_id.values[-1])
        redis_conn.set(redis_index_key, deepix_train_index)
        #注释了label
        for img_path, bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label in zip(
                data_from_db.img_path.values, data_from_db.bookmark_label.values, data_from_db.view_label.values,
                data_from_db.sanity_label.values, data_from_db.restrict_label.values,
                data_from_db.x_restrict_label.values,
                #data_from_db.label.values
        ):
            yield load_and_preprocess_image_from_url_py(img_path), {
                "bookmark_predict": label_str_to_tensor_py(bookmark_label),
                "view_predict": label_str_to_tensor_py(view_label),
                "sanity_predict": label_str_to_tensor_py(sanity_label),
                "restrict_predict": label_str_to_tensor_py(restrict_label),
                "x_restrict_predict": label_str_to_tensor_py(x_restrict_label),
                #"tag_predict": label_str_to_tensor_py(label)
            }
        del data_from_db


def label_str_to_tensor(labelStr):
    # l = list(map(float, bytes.decode(labelStr.numpy()).split(',')))
    # return tf.convert_to_tensor(l, dtype=tf.float32)
    a = tf.strings.to_number(tf.strings.split(labelStr, sep=","))
    # print(a)
    return a

def label_str_to_tensor_py(labelStr):
    l = list(map(float, labelStr.split(',')))
    return tf.convert_to_tensor(l, dtype=tf.float32)




def load_and_preprocess_image_from_url(url):
    try:
        image = tf.io.decode_image(
            httpclient.request('GET', bytes.decode(url.numpy()).replace('10.0.0.5', 'local.ipv4.host')).data,
            channels=3)
        image = tf.image.resize(image, [299, 299])
        image /= 255.0
        return image
    except Exception as e:
        print(url)

def load_and_preprocess_image_from_url_py(url):
    try:
        image = tf.io.decode_image(
            httpclient.request('GET', url.replace('10.0.0.5', 'local.ipv4.host')).data,
            channels=3)
        image = tf.image.resize(image, [299, 299])
        image /= 255.0
        return image
    except Exception as e:
        print(url)


def load_and_preprocess_image_from_url_warp(url):
    return tf.py_function(load_and_preprocess_image_from_url, [url], Tout=(tf.float32))


def build_label_data(*args):
    list = [label_str_to_tensor(i) for i in args]
    return list


def build_label_data_warp(bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label
                          #, label
                          ):
    return tf.py_function(build_label_data,
                          [bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label
                              #, label
                           ],
                          Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
                                #, tf.float32
                                ))


def _fixup_shape(images, labels):
    images.set_shape([None, None, None, 3])
    labels["bookmark_predict"].set_shape([None, 10])
    labels["view_predict"].set_shape([None, 10])
    labels["sanity_predict"].set_shape([None, 10])
    labels["restrict_predict"].set_shape([None, 3])
    labels["x_restrict_predict"].set_shape([None, 3])
    #labels["tag_predict"].set_shape([None, 10240])
    return images, labels


def map_img_and_label(x, y):
    return load_and_preprocess_image_from_url_warp(x), build_label_data_warp(y[0], y[1], y[2], y[3], y[4]
                                                                             #, y[5]
                                                                             )


def build_dataset(batch_size,test=False):
    if test:
        global min_deepix_train_index
        min_deepix_train_index = 60000000
        global  max_deepix_train_index
        max_deepix_train_index = 70000000

    dataset = tf.data.Dataset.from_generator(generate_data_from_db,
                                             output_types=(
                                             tf.float32, {"bookmark_predict": tf.float32, "view_predict": tf.float32,
                                                          "sanity_predict": tf.float32, "restrict_predict": tf.float32,
                                                          "x_restrict_predict": tf.float32
                                                 #, "tag_predict": tf.float32
                                                          })
                                             )

    dataset = dataset.shuffle(batch_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.map(_fixup_shape, num_parallel_calls=AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset
