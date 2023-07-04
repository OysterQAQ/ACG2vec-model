import gc
import json
import math
import os
import re
import time
from io import BytesIO
import urllib3
import clip
import pandas as pd
import redis
import sqlalchemy
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

is_contain = re.compile(r'[A-Za-z]', re.S)
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def is_image(filename):
    ext = os.path.splitext(filename)[-1]
    if ext.lower() in ['.jpg', '.jpeg', '.gif', '.bmp', '.png']:
        return True
    else:
        return False
class DanbooruIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start=0, end=2996459, offset=100, preprocess=_transform(224)):
        super(DanbooruIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.index = start
        self.offset = offset
        self.redis_index_key = 'DCLIP_index'
        self.redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
        self.sql = 'select id,path,tags from danbooru_illust where id > %s limit %s'
        self.engine = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/acg2vec?charset=utf8')
        self.preprocess = preprocess

    def _sample_generator(self, worker_id, start, end):
        index = start
        while index < end:
            # time_start = time.time()
            try:
                data_from_db = pd.read_sql(self.sql, self.engine, params=[index, self.offset])
            except:
                time.sleep(10)
                continue
            # data_from_db = pd.read_sql(self.sql, self.engine, params=[index,self.offset])
            length = len(data_from_db.id)
            index = data_from_db.id[length - 1]

            # print('\n查询sql耗时：', time_end - time_start, 's')
            # print('worker'+str(worker_id)+'\n当前训练到' + str(index))
            for i in range(length):
                # 加载并且缩放图片
                if not is_image(data_from_db.path[i]):
                    continue

                try:
                    img = self.preprocess(
                        Image.open(data_from_db.path[i].replace("./", "/mnt/lvm/danbooru2021/danbooru2021/")))
                except Exception as e:
                    #print(e)
                    continue

                #img = self.preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/107776952_p0_square1200.jpg"))
                # 处理标签
                tags = json.loads(data_from_db.tags[i])
                # 优先选择人物和作品标签
                category_group = {}
                for tag in tags:
                    category_group.setdefault(tag["category"], []).append(tag)

                # category_group=groupby(tags, key=lambda x: (x["category"]))
                character_list = category_group[4] if 4 in category_group else []
                # 作品需要过滤以bad开头的

                work_list = list(filter(
                    lambda e:
                               e["name"] != "original"
                            , category_group[3])) if 3 in category_group else []
                # work_list=  category_group[5] if 5 in category_group else []
                general_list = category_group[0] if 0 in category_group else []
                caption = ""
                caption_2 = None
                for character in character_list:
                    if len(work_list) != 0:
                        # 去除括号内作品内容
                        character["name"] = re.sub(u"\\(.*?\\)", "", character["name"])
                    caption += character["name"].replace("_", " ")
                    caption += ","
                caption = caption[:-1]
                if len(caption) != 0:
                    caption += " "
                if len(work_list) != 0:
                    caption += "from "
                for work in work_list:
                    caption += work["name"].replace("_", " ")
                    caption += " "
                # 普通标签
                if len(character_list)!=0 and len(general_list) != 0:
                    caption += "with "
                if len(general_list) > 20:
                    general_list_1 = general_list[:int(len(general_list) / 2)]
                    general_list_2 = general_list[int(len(general_list) / 2):]
                    caption_2 = caption
                    for general in general_list_1:
                        if len(
                                re.findall(is_contain, general["name"])) != 0:
                            caption_2 += general["name"].replace("_", " ")
                            caption_2 += ","
                    caption_2 = caption_2[:-1]
                    for general in general_list_2:
                        if len(
                                re.findall(is_contain, general["name"])) != 0:
                            caption += general["name"].replace("_", " ")
                            caption += ","
                    caption = caption[:-1]
                else:
                    for general in general_list:
                        # 如果标签数据目大于20 则拆分成两个caption
                        if len(
                                re.findall(is_contain, general["name"])) != 0:
                            caption += general["name"].replace("_", " ")
                            caption += ","
                    caption = caption[:-1]

                # 标签汇总成语句
                # tokenize语句
                # 返回
                # 过长截断 不行的话用huggingface的
                text_1 = clip.tokenize(texts=caption, truncate=True)
                text_2= None
                if caption_2 is not None:
                    text_2 = clip.tokenize(texts=caption_2, truncate=True)
                # 处理逻辑

                # print(img)
                yield img, text_1[0]
                if text_2 is not None:
                    yield img, text_2[0]
            del data_from_db
            gc.collect()

    def __len__(self):
        return self.end - self.start

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
        sample_iterator = self._sample_generator(worker_id, iter_start, iter_end)
        return sample_iterator

        # else:  # in a worker process
        #     # split workload
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)

        # return iter(range(iter_start, iter_end))


class PixivIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start=0, end=108489283, offset=100, preprocess=_transform(224)):
        super(PixivIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.index = start
        self.offset = offset
        self.redis_index_key = 'DCLIP_pixiv_index'
        self.redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
        self.sql = """
       select a.illust_id                                                                                      as id,
       a.tag_list                                                                                              as tags,
       REGEXP_REPLACE(JSON_EXTRACT(b.image_urls, concat('$[', (page - 1), '].medium')), '\\\\[|\\\\]| |"', '') as path
from (select * from acg2vec.pixiv_illust_with_danbooru_style_tag where illust_id >= %s limit %s) as a
         left join pixivic_crawler.illusts as b on a.illust_id = b.illust_id
        """
        self.engine = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/acg2vec?charset=utf8')
        self.preprocess = preprocess
        self.httpclient = urllib3.PoolManager()

    def _sample_generator(self, worker_id, start, end):
        index = start
        while index < end:
            # time_start = time.time()
            try:
                data_from_db = pd.read_sql(self.sql, self.engine, params=[index, self.offset])
            except:
                time.sleep(10)
                continue
            # data_from_db = pd.read_sql(self.sql, self.engine, params=[index,self.offset])
            length = len(data_from_db.id)
            index = data_from_db.id[length - 1]

            # print('\n查询sql耗时：', time_end - time_start, 's')
            # print('worker'+str(worker_id)+'\n当前训练到' + str(index))
            for i in range(length):
                tags = json.loads(data_from_db.tags[i])
                if (len(tags)==0):
                    continue
                # 加载并且缩放图片
                try:
                    resp = self.httpclient.request('GET', data_from_db.path[i].replace('https://i.pximg.net',
                                                                            'http://local.ipv4.host:8888'))
                    if resp.status != 200:
                        continue
                    img = Image.open(BytesIO(resp.data))
                    img = self.preprocess(
                        img)
                except Exception as e:
                    print(e)
                    continue

                #img = self.preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/107776952_p0_square1200.jpg"))
                # 处理标签

                # 优先选择人物和作品标签
                category_group = {}
                for tag in tags:
                    category_group.setdefault(tag["category"], []).append(tag)

                # category_group=groupby(tags, key=lambda x: (x["category"]))
                character_list = category_group[4] if 4 in category_group else []
                # 作品需要过滤以bad开头的

                work_list = list(filter(
                    lambda e:
                               e["name"] != "original"
                            , category_group[3])) if 3 in category_group else []
                # work_list=  category_group[5] if 5 in category_group else []
                general_list = category_group[0] if 0 in category_group else []
                caption = ""
                caption_2 = None
                for character in character_list:
                    if len(work_list) != 0:
                        # 去除括号内作品内容
                        character["name"] = re.sub(u"\\(.*?\\)", "", character["name"])
                    caption += character["name"].replace("_", " ")
                    caption += ","
                caption = caption[:-1]
                if len(caption)!=0:
                    caption += " "

                if len(work_list) != 0:
                    caption += "from "
                for work in work_list:
                    caption += work["name"].replace("_", " ")
                    caption += " "
                # 普通标签
                if len(character_list)!=0 and len(general_list) != 0:
                    caption += "with "
                if len(general_list) > 20:
                    general_list_1 = general_list[:int(len(general_list) / 2)]
                    general_list_2 = general_list[int(len(general_list) / 2):]
                    caption_2 = caption
                    for general in general_list_1:
                        if len(
                                re.findall(is_contain, general["name"])) != 0:
                            caption_2 += general["name"].replace("_", " ")
                            caption_2 += ","
                    caption_2 = caption_2[:-1]
                    for general in general_list_2:
                        if len(
                                re.findall(is_contain, general["name"])) != 0:
                            caption += general["name"].replace("_", " ")
                            caption += ","
                    caption = caption[:-1]
                else:
                    for general in general_list:
                        # 如果标签数据目大于20 则拆分成两个caption
                        if len(
                                re.findall(is_contain, general["name"])) != 0:
                            caption += general["name"].replace("_", " ")
                            caption += ","
                    caption = caption[:-1]

                # 标签汇总成语句
                # tokenize语句
                # 返回
                # 过长截断 不行的话用huggingface的
                text_1 = clip.tokenize(texts=caption, truncate=True)
                text_2= None
                if caption_2 is not None:
                    text_2 = clip.tokenize(texts=caption_2, truncate=True)
                # 处理逻辑
                # print(img)
                yield img, text_1[0]
                if text_2 is not None:
                    yield img, text_2[0]
            del data_from_db
            gc.collect()

    def __len__(self):
        return self.end - self.start

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
        sample_iterator = self._sample_generator(worker_id, iter_start, iter_end)
        return sample_iterator

        # else:  # in a worker process
        #     # split workload
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)

        # return iter(range(iter_start, iter_end))


# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
# ds = KVIterableDataset(start=1, end=72708121,offset=2400)
# ds = KVIterableDataset(start=1, end=72708121,offset=2400)

# # Single-process loading
# print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
#
# # Mult-process loading with two worker processes
# # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
# print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
#
# # With even more workers
# print(list(torch.utils.data.DataLoader(ds, num_workers=12)))




# ds = DanbooruIterableDataset(start=5000, end=6000, offset=10, )
# dataloader = torch.utils.data.DataLoader(ds, num_workers=0)
# for samples, targets in dataloader:
#     print(targets.shape)
# ds = PixivIterableDataset(start=5000, end=6000, offset=10, )
# dataloader = torch.utils.data.DataLoader(ds, num_workers=0)
# for samples, targets in dataloader:
#      print(targets.shape)
