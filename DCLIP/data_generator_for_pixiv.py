import gc
import json
import math
import os
import re
import time
import urllib3
import clip
import pandas as pd
import redis
import sqlalchemy
import torch
from PIL import Image
from io import BytesIO
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


class PixivIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start=0, end=108187734, offset=1000, preprocess=_transform(224)):
        super(PixivIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.index = start
        self.offset = offset
        self.redis_index_key = 'DCLIP_index'
        self.redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
        self.sql = """
      select illusts.illust_id, REGEXP_REPLACE(JSON_EXTRACT(image_urls, '$[*].medium'), '\\\\[|\\\\]| |"', '') as image_urls
from  pixivic_crawler.illusts where illust_id > %s and total_bookmarks > 50 limit  %s
     """
        self.engine = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/pix2score?charset=utf8')
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
            length = len(data_from_db.illust_id)
            index = data_from_db.illust_id[length - 1]

            # print('\n查询sql耗时：', time_end - time_start, 's')
            # print('worker'+str(worker_id)+'\n当前训练到' + str(index))
            for i in range(length):
                # 加载并且缩放图片
                illust_id = data_from_db.illust_id[i]
                img_url_string = data_from_db.image_urls[i]
                image_urls = img_url_string.split(',')


                # 先处理成（illust_id,image_url的形式）
                for i, image_url in enumerate(image_urls):
                    try:
                        resp = self.httpclient.request('GET', image_url.replace('https://i.pximg.net',
                                                                           'http://local.ipv4.host:8888'))
                        if resp.status != 200:
                            continue
                        img = Image.open(BytesIO(resp.data))
                        image = self.preprocess(img)
                        yield image, {"illust_id": illust_id, "page": i + 1}
                    except Exception as e:
                        print(e)
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
#     print(samples.shape)
