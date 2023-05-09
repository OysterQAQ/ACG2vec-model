import torch
import math
import pymysql
import redis
import sqlalchemy
import pandas as pd
import time
from sentence_transformers import InputExample
import gc
import clip
import os
import json
from itertools import groupby
import re
from PIL import Image

is_contain = re.compile(r'[A-Za-z]',re.S)
class KVIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start=0, end=2996459,offset=100,preprocess=None):
        super(KVIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.index=start
        self.offset = offset
        self.redis_index_key = 'DCLIP_index'
        self.redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
        self.sql = 'select id,path,tags from danbooru_illust where id > %s limit %s'
        self.engine = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/deepix?charset=utf8')
        self.preprocess=preprocess

    def _sample_generator(self,worker_id,start,end):
        index=start
        while index < end:
            #time_start = time.time()
            try:
                data_from_db = pd.read_sql(self.sql, self.engine, params=[index, self.offset])
            except:
                time.sleep(10)
                continue
            #data_from_db = pd.read_sql(self.sql, self.engine, params=[index,self.offset])
            length=len(data_from_db.sentence_1)
            index = data_from_db.id[length-1]
            time_end = time.time()
            #print('\n查询sql耗时：', time_end - time_start, 's')
            #print('worker'+str(worker_id)+'\n当前训练到' + str(index))
            for i in range(length):
                #加载并且缩放图片
                img = self.preprocess(Image.open("path"+data_from_db.path[i]))
                #处理标签
                tags = json.loads(data_from_db.tags[i])
                #优先选择人物和作品标签

                category_group=groupby(tags, key=lambda x: (x["category"]))
                character_list=category_group[4]
                work_list=category_group[5]
                general_list=category_group[0]
                caption =""
                for character in character_list:
                    if range(work_list)!=0:
                        #去除括号内作品内容
                        character=re.sub(u"\\(.*?\\)", "", character)
                    caption+=character.replace("_", " ")
                    caption+=","
                caption = caption[:-1]
                if range(work_list)!=0:
                    caption+="from "
                for work in work_list:
                    caption+=work.replace("_", " ")
                    caption+=" "
                # 普通标签
                if range(general_list)!=0:
                    caption+="with tag "
                for general in general_list:
                    if general.find("girl") == -1 and general.find("boy") == -1 and len(re.findall(is_contain,general))!=0:
                        caption += general
                        caption += ","
                caption = caption[:-1]
                #标签汇总成语句
                #tokenize语句
                #返回
                # 过长截断 不行的话用huggingface的
                text = clip.tokenize(texts=caption,truncate=True)
                #处理逻辑
                yield InputExample(img,text)
            del data_from_db
            gc.collect()

    def __len__(self):
        return self.end - self.start -6000000#有些跳过的条目

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
            worker_id = 0
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        sample_iterator = self._sample_generator(worker_id,iter_start,iter_end)
        return sample_iterator

            # else:  # in a worker process
            #     # split workload
            #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            #     worker_id = worker_info.id
            #     iter_start = self.start + worker_id * per_worker
            #     iter_end = min(iter_start + per_worker, self.end)

        #return iter(range(iter_start, iter_end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
#ds = KVIterableDataset(start=1, end=72708121,offset=2400)
#ds = KVIterableDataset(start=1, end=72708121,offset=2400)

# # Single-process loading
#print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
#
# # Mult-process loading with two worker processes
# # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
#print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
#
# # With even more workers
# print(list(torch.utils.data.DataLoader(ds, num_workers=12)))