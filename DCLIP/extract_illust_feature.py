import json
import math

import torch
import clip
from torch import optim, nn
import time
import warnings
warnings.filterwarnings("ignore")
from DCLIP.data_generator import DanbooruIterableDataset
from PIL import Image
# Latest Update : 18 July 2022, 09:55 GMT+7
import gc
import os
import pandas as pd
import pymysql
import redis
import sqlalchemy
import tensorflow as tf
from io import BytesIO

import urllib3
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)  # Must set jit=False for training
checkpoint = torch.load("model_checkpoint/dclip_5.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
#checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
#checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
#checkpoint['model_state_dict']["vocab_size"] = model.vocab_size


model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
gc.collect()

engine = sqlalchemy.create_engine(
    'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/pixivic_rec?charset=utf8')
# sql1="select illust_id from image_sync_2022_12_22 where is_finish=1 and illust_id>%s  order by illust_id limit %s"
sql = """
      select illusts.illust_id, REGEXP_REPLACE(JSON_EXTRACT(image_urls, '$[*].medium'), '\\\\[|\\\\]| |"', '') as image_urls
from  pixivic_crawler.illusts where illust_id > %s and total_bookmarks > 50 limit  %s
     """
sql_insert = "insert into img_clip_feature (illust_id, page, clip_feature) values ( %s, %s, %s)"
db = pymysql.connect(host='local.ipv4.host',
                     user='root',
                     password='Cheerfun.dev',
                     database='pixivic_rec')
redis_index_key = 'illust_clip_feature_extract_index'
redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
illust_id_index = int(redis_conn.get(redis_index_key))
httpclient = urllib3.PoolManager()


def lst_trans(lst, n):
    m = int(math.ceil(len(lst) / float(n)))
    sp_lst = []
    for i in range(n):
        sp_lst.append(lst[i * m:(i + 1) * m])
    return sp_lst


while illust_id_index < 105592139:
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
                img = Image.open(BytesIO(resp.data))
                image =preprocess(img)
                illust_list.append({"illust_id": illust_id, "page": i + 1, "image": image})

            except Exception as e:
                print(e)
                flag = False
                # continue
    if len(illust_list) == 0:
        continue
    n = int(len(illust_list) / 64)
    if n == 0:
        illust_list = []
        illust_list.append(illust_list)
    else:
        illust_list = lst_trans(illust_list, n)
    for i, illust_list_e in enumerate(illust_list):
        images = list(map(lambda x: x["image"], illust_list_e))
        clip_feature= model.encode_image(images)
        for o, r1 in zip(illust_list_e, clip_feature):
            cursor = db.cursor()
            cursor.execute(
                sql_insert % \
                (o["illust_id"], o["page"], json.dumps(r1.tolist()))
            )
            db.commit()
    if flag:
        illust_id_index = data_from_db.illust_id[-1]
        redis_conn.set(redis_index_key, str(illust_id_index))
