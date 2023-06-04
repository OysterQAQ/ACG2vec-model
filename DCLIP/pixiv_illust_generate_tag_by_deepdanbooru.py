import os

import pandas as pd
import pymysql
import redis
import sqlalchemy
import tensorflow as tf
import urllib3
from tensorflow import keras
from tensorflow.keras import mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
engine = sqlalchemy.create_engine(
    'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/acg2vec?charset=utf8')
# sql1="select illust_id from image_sync_2022_12_22 where is_finish=1 and illust_id>%s  order by illust_id limit %s"
sql = """
select a.illust_id as illust_id,
       REGEXP_REPLACE(JSON_EXTRACT(image_urls, '$[*].medium'), '\\\\[|\\\\]| |"', '') as image_urls
from acg2vec.pixiv_illust_danbooru_style_tag a
         left join pixivic_crawler.illusts b on a.illust_id = b.illust_id where a.illust_id > %s limit  %s
     """

sql_insert = "insert ignore into pixiv_illust_danbooru_style_tag_by_deepdanbooru values ( %s, %s, %s)"


db = pymysql.connect(host='local.ipv4.host',
                     user='root',
                     password='Cheerfun.dev',
                     database='acg2vec')
httpclient = urllib3.PoolManager()
redis_index_key = 'pixiv_illust_generate_tag_by_deepdanbooru_index'
redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
illust_id_index = int(redis_conn.get(redis_index_key))
deepdanbooru = keras.models.load_model('/Volumes/Data/oysterqaq/Downloads/deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5', compile=False)

#style_model = keras.models.load_model('/root/style_feature_extract_model', compile=False)
def load_tags(tags_path):
    with open(tags_path, "r") as tags_stream:
        tags = [tag for tag in (tag.strip() for tag in tags_stream) if tag]
        return tags


tags=load_tags(os.path.join("/Volumes/Data/oysterqaq/Downloads/deepdanbooru-v3-20211112-sgd-e28", "tags.txt"))

import math
import json


def lst_trans(lst, n):
    m = int(math.ceil(len(lst) / float(n)))
    sp_lst = []
    for i in range(n):
        sp_lst.append(lst[i * m:(i + 1) * m])
    return sp_lst


while illust_id_index <= 108495448:
    data_from_db = pd.read_sql(sql, engine, params=[illust_id_index, 64])
    illust_list = []
    flag = True
    for i in range(len(data_from_db.illust_id)):
        flag = True
        illust_id = data_from_db.illust_id[i]
        img_url_string = data_from_db.image_urls[i]
        image_urls = img_url_string.split(',')
        img_list = []

        # 先处理成（illust_id,image_url的形式）
        for i, image_url in enumerate(image_urls):
            try:
                resp = httpclient.request('GET', image_url.replace('https://i.pximg.net',
                                                                   'http://local.ipv4.host:8888'))
                if resp.status != 200:
                    continue
                image_raw = resp.data
                image = tf.io.decode_image(image_raw, channels=3)
                image = tf.image.resize(image, [512, 512])
                image /= 255.0
                illust_list.append({"illust_id": illust_id, "page": i + 1, "image": image})

            except Exception as e:
                print(e)
                flag = False
                continue

    if len(illust_list) == 0:
        continue
    n = int(len(illust_list) / 64)
    illust_list_list = []
    if n == 0:
        illust_list_list.append(illust_list)
    else:
        illust_list_list = lst_trans(illust_list, n)
    for i, illust_list_e in enumerate(illust_list_list):
        images = list(map(lambda x: x["image"], illust_list_e))


        tag_features = deepdanbooru.predict_on_batch(tf.stack(images))

        for o, tag_feature in zip(illust_list_e, tag_features):
            #将tag_faeture过滤出大于0.5 转位标签
            tag_list=[]
            for index in range(len(tag_feature)):
                if tag_feature[index]>0.6:
                    tag_list.append(tags[index])


            cursor = db.cursor()
            cursor.execute(
                sql_insert ,
                (o["illust_id"], o["page"], json.dumps(tag_list))
            )

            db.commit()
    if flag:
        illust_id_index = data_from_db.illust_id[len(data_from_db.illust_id)-1]
        redis_conn.set(redis_index_key, str(illust_id_index))