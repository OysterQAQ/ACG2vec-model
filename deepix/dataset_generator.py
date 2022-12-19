import json
import time

import pandas as pd
import redis
import sqlalchemy
import tensorflow as tf
import urllib3

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSetGenerator:
    def __init__(self, batch_size, test=False, input_size=[224, 224]):
        self.batch_size = batch_size
        self.test = test
        self.input_size = input_size
        self.redis_index_key = 'deepix_illust_index'
        self.redis_epoch_key = 'deepix_epoch_index'
        self.redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
        self.sql_2 = '''
            select image_sync.illust_id as illust_id, total_bookmarks,total_view,sanity_level ,image_urls
            from image_sync
                    left join illusts i on image_sync.illust_id = i.illust_id
            where image_sync.illust_id < %s
            order by illust_id desc
            limit %s 
            '''
        self.engine = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/deepix?charset=utf8')
        self.engine_2 = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/pixivic_crawler?charset=utf8')
        self.offset = 160000
        self.offset_2 = 160000
        self.httpclient = urllib3.PoolManager()
        self.min_deepix_train_index = 20000000
        self.max_deepix_train_index = 80000000

    def generate_data_from_db(self):
        deepix_train_index = self.max_deepix_train_index if self.redis_conn.get(self.redis_index_key) is None else int(
            self.redis_conn.get(self.redis_index_key))
        if (deepix_train_index < self.min_deepix_train_index):
            deepix_train_index = self.max_deepix_train_index
            self.redis_conn.set(self.redis_index_key, deepix_train_index)
            epoch_index = int(self.redis_conn.get(self.redis_epoch_key))
            self.redis_conn.set(self.redis_epoch_key, epoch_index + 1)
        bookmark_discretization = tf.keras.layers.Discretization(
            bin_boundaries=[10, 30, 50, 70, 100, 130, 170, 220, 300, 400, 550, 800, 1300, 2700],
            output_mode='one_hot', )
        view_discretization = tf.keras.layers.Discretization(
            bin_boundaries=[500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6500, 8500, 12000, 19000, 35000],
            output_mode='one_hot', )
        sanity_discretization = tf.keras.layers.Discretization(bin_boundaries=[2, 4, 6, 7], output_mode='one_hot', )
        while deepix_train_index > self.min_deepix_train_index:
            time_start = time.time()
            data_from_db = pd.read_sql(self.sql_2, self.engine_2, params=[deepix_train_index, self.offset])
            time_end = time.time()
            print('\n查询sql耗时：', time_end - time_start, 's')
            print('\n当前训练到' + str(deepix_train_index))
            deepix_train_index = int(data_from_db.illust_id.values[-1])
            self.redis_conn.set(self.redis_index_key, deepix_train_index)
            # 注释了label
            for image_urls, bookmark_label, view_label, sanity_label in zip(
                    data_from_db.image_urls.values, data_from_db.total_bookmarks.values, data_from_db.total_view.values,
                    data_from_db.sanity_level.values):
                image_urls = json.load(image_urls)
                for image_url in image_urls:
                    yield self.load_and_preprocess_image_from_url_with_catch(image_url['medium']), {
                        "bookmark_predict": bookmark_discretization(bookmark_label),
                        "view_predict": view_discretization(view_label),
                        "sanity_predict": sanity_discretization(sanity_label),
                    }

            del data_from_db

    def generate_data_from_db_for_test(self):
        bookmark_discretization = tf.keras.layers.Discretization(
            bin_boundaries=[10, 30, 50, 70, 100, 130, 170, 220, 300, 400, 550, 800, 1300, 2700],
            output_mode='one_hot', )
        view_discretization = tf.keras.layers.Discretization(
            bin_boundaries=[500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6500, 8500, 12000, 19000, 35000],
            output_mode='one_hot', )
        sanity_discretization = tf.keras.layers.Discretization(bin_boundaries=[2, 4, 6, 7], output_mode='one_hot', )
        time_start = time.time()
        data_from_db = pd.read_sql(self.sql_2, self.engine_2, params=[60000000, 160000])
        time_end = time.time()
        print('\n查询sql耗时：', time_end - time_start, 's')
        deepix_train_index = int(data_from_db.illust_id.values[-1])
        self.redis_conn.set(self.redis_index_key, deepix_train_index)
        # 注释了label
        label = []
        feature = []
        for image_urls, bookmark_label, view_label, sanity_label in zip(
                data_from_db.image_urls.values, data_from_db.total_bookmarks.values, data_from_db.total_view.values,
                data_from_db.sanity_level.values):
            image_urls = json.load(image_urls)
            for image_url in image_urls:
                try:
                    label.append(self.load_and_preprocess_image_from_url_py(image_url['medium']))
                    feature.append({
                        "bookmark_predict": bookmark_discretization(bookmark_label),
                        "view_predict": view_discretization(view_label),
                        "sanity_predict": sanity_discretization(sanity_label),
                    })

                # yield load_and_preprocess_image_from_url_py(image_url['medium']), {
                #     "bookmark_predict": bookmark_discretization(bookmark_label),
                #     "view_predict": view_discretization(view_label),
                #     "sanity_predict": sanity_discretization(sanity_label),
                # }
                except Exception as e:
                    print(image_url)
        del data_from_db
        return label, feature

    def load_and_preprocess_image_from_url(self,url):
        try:
            image = tf.io.decode_image(
                self.httpclient.request('GET', bytes.decode(url.numpy()).replace('10.0.0.5', 'local.ipv4.host')).data,
                channels=3)
            image = tf.image.resize(image, self.input_size)
            image /= 255.0
            return image
        except Exception as e:
            print(url)

    def load_and_preprocess_image_from_url_py(self,url):
        # try:
        image = tf.io.decode_image(
            self.httpclient.request('GET', url.replace('https://i.pximg.net', 'http://local.ipv4.host:8888')).data,
            channels=3)
        image = tf.image.resize(image, self.input_size)
        image /= 255.0
        return image
        # except Exception as e:
        #     print(url)

    def load_and_preprocess_image_from_url_with_catch(self,url):
        try:
            image = tf.io.decode_image(
                self.httpclient.request('GET', url.replace('https://i.pximg.net', 'http://local.ipv4.host:8888')).data,
                channels=3)
            image = tf.image.resize(image, self.input_size)
            image /= 255.0
            return image
        except Exception as e:
            print(url)

    def _fixup_shape(self,images, labels):
        images.set_shape([None, None, None, 3])
        labels["bookmark_predict"].set_shape([None, 15])
        labels["view_predict"].set_shape([None, 15])
        labels["sanity_predict"].set_shape([None, 5])
        return images, labels

    def build_dataset(self):
        if self.test:
            return self.build_dataset_for_test()

        dataset = tf.data.Dataset.from_generator(self.generate_data_from_db,
                                                 output_types=(
                                                     tf.float32,
                                                     {"bookmark_predict": tf.float32, "view_predict": tf.float32,
                                                      "sanity_predict": tf.float32
                                                      })
                                                 )

        dataset = dataset.shuffle(self.batch_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.map(self._fixup_shape, num_parallel_calls=AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        return dataset

    def build_dataset_for_test(self):
        label, feature = self.generate_data_from_db_for_test()
        dataset = tf.data.Dataset.from_tensor_slices((feature, label))
        dataset = dataset.shuffle(self.batch_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.map(self._fixup_shape, num_parallel_calls=AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        return dataset


# 数据集构建


# 获取索引
# redis_index_key = 'deepix_illust_index'
# redis_epoch_key = 'deepix_epoch_index'
# redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
# sql = '''
# select illust_id,img_path,
#
#            REGEXP_REPLACE(bookmark_label, '\\\\[|\\\\]', '') as bookmark_label ,
#               REGEXP_REPLACE(view_label, '\\\\[|\\\\]', '') as view_label,
#               REGEXP_REPLACE(sanity_label, '\\\\[|\\\\]', '') as sanity_label,
#               REGEXP_REPLACE(restrict_label, '\\\\[|\\\\]', '') as restrict_label,
#               REGEXP_REPLACE(x_restrict_label, '\\\\[|\\\\]', '') as x_restrict_label
#               -- ,REGEXP_REPLACE(label, '\\\\[|\\\\]', '') as label
#
# from deepix_data
# where illust_id < %s  order by illust_id desc limit %s
# '''
#
# sql_2 = '''
# select image_sync.illust_id, total_bookmarks,total_view,sanity_level ,image_urls
# from image_sync
#          left join illusts i on image_sync.illust_id = i.illust_id
# where illust_id < %s
# order by illust_id desc
# limit %s
# '''
#
# engine = sqlalchemy.create_engine('mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/deepix?charset=utf8')
# engine_2 = sqlalchemy.create_engine(
#     'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/pixivic_crawler?charset=utf8')
# offset = 160000
# offset_2 = 160000
# httpclient = urllib3.PoolManager()
# min_deepix_train_index = 20000000
# max_deepix_train_index = 80000000


# deepix_train_index = 50000000
# def generate_data_from_db():
#     deepix_train_index = max_deepix_train_index if redis_conn.get(redis_index_key) is None else int(
#         redis_conn.get(redis_index_key))
#     if (deepix_train_index < min_deepix_train_index):
#         deepix_train_index = max_deepix_train_index
#         redis_conn.set(redis_index_key, deepix_train_index)
#         epoch_index = int(redis_conn.get(redis_epoch_key))
#         redis_conn.set(redis_epoch_key, epoch_index + 1)
#     while deepix_train_index > min_deepix_train_index:
#         time_start = time.time()
#         data_from_db = pd.read_sql(sql, engine, params=[deepix_train_index, offset])
#         time_end = time.time()
#         print('\n查询sql耗时：', time_end - time_start, 's')
#         print('\n当前训练到' + str(deepix_train_index))
#         deepix_train_index = int(data_from_db.illust_id.values[-1])
#         redis_conn.set(redis_index_key, deepix_train_index)
#         # 注释了label
#         for img_path, bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label in zip(
#                 data_from_db.img_path.values, data_from_db.bookmark_label.values, data_from_db.view_label.values,
#                 data_from_db.sanity_label.values, data_from_db.restrict_label.values,
#                 data_from_db.x_restrict_label.values,
#                 # data_from_db.label.values
#         ):
#             yield load_and_preprocess_image_from_url_py(img_path), {
#                 "bookmark_predict": label_str_to_tensor_py(bookmark_label),
#                 "view_predict": label_str_to_tensor_py(view_label),
#                 "sanity_predict": label_str_to_tensor_py(sanity_label),
#                 "restrict_predict": label_str_to_tensor_py(restrict_label),
#                 "x_restrict_predict": label_str_to_tensor_py(x_restrict_label),
#                 # "tag_predict": label_str_to_tensor_py(label)
#             }
#         del data_from_db


# def generate_data_from_db():
#     deepix_train_index = max_deepix_train_index if redis_conn.get(redis_index_key) is None else int(
#         redis_conn.get(redis_index_key))
#     if (deepix_train_index < min_deepix_train_index):
#         deepix_train_index = max_deepix_train_index
#         redis_conn.set(redis_index_key, deepix_train_index)
#         epoch_index = int(redis_conn.get(redis_epoch_key))
#         redis_conn.set(redis_epoch_key, epoch_index + 1)
#     bookmark_discretization = tf.keras.layers.Discretization(
#         bin_boundaries=[10, 30, 50, 70, 100, 130, 170, 220, 300, 400, 550, 800, 1300, 2700], output_mode='one_hot', )
#     view_discretization = tf.keras.layers.Discretization(
#         bin_boundaries=[500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6500, 8500, 12000, 19000, 35000],
#         output_mode='one_hot', )
#     sanity_discretization = tf.keras.layers.Discretization(bin_boundaries=[2, 4, 6, 7], output_mode='one_hot', )
#     while deepix_train_index > min_deepix_train_index:
#         time_start = time.time()
#         data_from_db = pd.read_sql(sql_2, engine_2, params=[deepix_train_index, offset])
#         time_end = time.time()
#         print('\n查询sql耗时：', time_end - time_start, 's')
#         print('\n当前训练到' + str(deepix_train_index))
#         deepix_train_index = int(data_from_db.illust_id.values[-1])
#         redis_conn.set(redis_index_key, deepix_train_index)
#         # 注释了label
#         for image_urls, bookmark_label, view_label, sanity_label in zip(
#                 data_from_db.image_urls.values, data_from_db.total_bookmarks.values, data_from_db.total_view.values,
#                 data_from_db.sanity_level.values):
#             image_urls = json.load(image_urls)
#             for image_url in image_urls:
#                 yield load_and_preprocess_image_from_url_with_catch(image_url['medium']), {
#                     "bookmark_predict": bookmark_discretization(bookmark_label),
#                     "view_predict": view_discretization(view_label),
#                     "sanity_predict": sanity_discretization(sanity_label),
#                 }
#
#         del data_from_db


# def generate_data_from_db_for_test():
#     bookmark_discretization = tf.keras.layers.Discretization(
#         bin_boundaries=[10, 30, 50, 70, 100, 130, 170, 220, 300, 400, 550, 800, 1300, 2700], output_mode='one_hot', )
#     view_discretization = tf.keras.layers.Discretization(
#         bin_boundaries=[500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6500, 8500, 12000, 19000, 35000],
#         output_mode='one_hot', )
#     sanity_discretization = tf.keras.layers.Discretization(bin_boundaries=[2, 4, 6, 7], output_mode='one_hot', )
#     time_start = time.time()
#     data_from_db = pd.read_sql(sql_2, engine_2, params=[60000000, 160000])
#     time_end = time.time()
#     print('\n查询sql耗时：', time_end - time_start, 's')
#     deepix_train_index = int(data_from_db.illust_id.values[-1])
#     redis_conn.set(redis_index_key, deepix_train_index)
#     # 注释了label
#     label = []
#     feature = []
#     for image_urls, bookmark_label, view_label, sanity_label in zip(
#             data_from_db.image_urls.values, data_from_db.total_bookmarks.values, data_from_db.total_view.values,
#             data_from_db.sanity_level.values):
#         image_urls = json.load(image_urls)
#         for image_url in image_urls:
#             try:
#                 label.append(load_and_preprocess_image_from_url_py(image_url['medium']))
#                 feature.append({
#                     "bookmark_predict": bookmark_discretization(bookmark_label),
#                     "view_predict": view_discretization(view_label),
#                     "sanity_predict": sanity_discretization(sanity_label),
#                 })
#
#             # yield load_and_preprocess_image_from_url_py(image_url['medium']), {
#             #     "bookmark_predict": bookmark_discretization(bookmark_label),
#             #     "view_predict": view_discretization(view_label),
#             #     "sanity_predict": sanity_discretization(sanity_label),
#             # }
#             except Exception as e:
#                 print(image_url)
#     del data_from_db
#     return label, feature


def label_str_to_tensor(labelStr):
    # l = list(map(float, bytes.decode(labelStr.numpy()).split(',')))
    # return tf.convert_to_tensor(l, dtype=tf.float32)
    a = tf.strings.to_number(tf.strings.split(labelStr, sep=","))
    # print(a)
    return a


def label_str_to_tensor_py(labelStr):
    l = list(map(float, labelStr.split(',')))
    return tf.convert_to_tensor(l, dtype=tf.float32)


# def load_and_preprocess_image_from_url(url):
#     try:
#         image = tf.io.decode_image(
#             httpclient.request('GET', bytes.decode(url.numpy()).replace('10.0.0.5', 'local.ipv4.host')).data,
#             channels=3)
#         image = tf.image.resize(image, [512, 512])
#         image /= 255.0
#         return image
#     except Exception as e:
#         print(url)


# def load_and_preprocess_image_from_url_py(url):
#     try:
#         image = tf.io.decode_image(
#             httpclient.request('GET', url.replace('10.0.0.5', 'local.ipv4.host')).data,
#             channels=3)
#         image = tf.image.resize(image, [512, 512])
#         image /= 255.0
#         return image
#     except Exception as e:
#         print(url)

# def load_and_preprocess_image_from_url_py(url):
#     # try:
#     image = tf.io.decode_image(
#         httpclient.request('GET', url.replace('https://i.pximg.net', 'http://local.ipv4.host:8888')).data,
#         channels=3)
#     image = tf.image.resize(image, [512, 512])
#     image /= 255.0
#     return image
#     # except Exception as e:
#     #     print(url)
#
#
# def load_and_preprocess_image_from_url_with_catch(url):
#     try:
#         image = tf.io.decode_image(
#             httpclient.request('GET', url.replace('https://i.pximg.net', 'http://local.ipv4.host:8888')).data,
#             channels=3)
#         image = tf.image.resize(image, [512, 512])
#         image /= 255.0
#         return image
#     except Exception as e:
#         print(url)


# def load_and_preprocess_image_from_url_warp(url):
#     return tf.py_function(load_and_preprocess_image_from_url, [url], Tout=(tf.float32))


# def build_label_data(*args):
#     list = [label_str_to_tensor(i) for i in args]
#     return list


# def build_label_data_warp(bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label
#                           # , label
#                           ):
#     return tf.py_function(build_label_data,
#                           [bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label
#                            # , label
#                            ],
#                           Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
#                                 # , tf.float32
#                                 ))


# def _fixup_shape(images, labels):
#     images.set_shape([None, None, None, 3])
#     labels["bookmark_predict"].set_shape([None, 15])
#     labels["view_predict"].set_shape([None, 15])
#     labels["sanity_predict"].set_shape([None, 5])
#     return images, labels
#
#
# def map_img_and_label(x, y):
#     return load_and_preprocess_image_from_url_warp(x), build_label_data_warp(y[0], y[1], y[2], y[3], y[4]
#
#                                                                              )
#
#
# def build_dataset(batch_size, test=False):
#     if test:
#         return build_dataset_for_test(batch_size)
#
#     dataset = tf.data.Dataset.from_generator(generate_data_from_db,
#                                              output_types=(
#                                                  tf.float32,
#                                                  {"bookmark_predict": tf.float32, "view_predict": tf.float32,
#                                                   "sanity_predict": tf.float32
#                                                   })
#                                              )
#
#     dataset = dataset.shuffle(batch_size, reshuffle_each_iteration=True)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(buffer_size=AUTOTUNE)
#     dataset = dataset.map(_fixup_shape, num_parallel_calls=AUTOTUNE)
#     dataset = dataset.apply(tf.data.experimental.ignore_errors())
#     return dataset
#
#
# def build_dataset_for_test(batch_size):
#     label, feature = generate_data_from_db_for_test()
#     dataset = tf.data.Dataset.from_tensor_slices((feature, label))
#     dataset = dataset.shuffle(batch_size, reshuffle_each_iteration=True)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(buffer_size=AUTOTUNE)
#     dataset = dataset.map(_fixup_shape, num_parallel_calls=AUTOTUNE)
#     dataset = dataset.apply(tf.data.experimental.ignore_errors())
#     return dataset
