import json
import math

import torch
import clip
from torch import optim, nn
import time

from PIL import Image
# Latest Update : 18 July 2022, 09:55 GMT+7
import gc
import os
import pandas as pd
import pymysql
import redis
import sqlalchemy
from io import BytesIO
from data_generator_for_pixiv import PixivIterableDataset
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, _ = clip.load("ViT-L/14", device=device)  # Must set jit=False for training
checkpoint = torch.load("model_checkpoint/dclip_7.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size


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
sql_insert = "insert ignore into img_clip_feature (illust_id, page, clip_feature) values ( %s, %s, '%s')"
db = pymysql.connect(host='local.ipv4.host',
                     user='root',
                     password='Cheerfun.dev',
                     database='pixivic_rec')

ds = PixivIterableDataset()
dataloader = torch.utils.data.DataLoader(ds, num_workers=10, batch_size=256)
for batch in dataloader:

    images, illust_info = batch
    images = images.to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)

    for illust_id, page, r1 in zip(illust_info["illust_id"], illust_info["page"], image_features):
        cursor = db.cursor()
        cursor.execute(
            sql_insert % \
            (illust_id.numpy(), page.numpy(), json.dumps(r1.tolist()))
        )
        db.commit()


