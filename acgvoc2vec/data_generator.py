import torch
import math
import pymysql
import redis
import sqlalchemy
import pandas as pd
import time
from sentence_transformers import InputExample
class KVIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end,offset):
        super(KVIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.index=start
        self.offset = offset
        self.redis_index_key = 'acgvoc2vec_index'
        self.redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)
        self.sql = 'select id,sentence_1,sentence_2 from fine_tune_st_dataset where id > %s limit %s'
        self.engine = sqlalchemy.create_engine(
            'mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/deepix?charset=utf8')

    def _sample_generator(self):
        while self.index < self.end:
            time_start = time.time()
            data_from_db = pd.read_sql(self.sql, self.engine, params=[self.index,self.offset])
            self.index = data_from_db.id[self.offset-1]
            time_end = time.time()
            print('\n查询sql耗时：', time_end - time_start, 's')
            print('\n当前训练到' + str(self.index))
            for i in range(len(data_from_db.sentence_1)):
                if str.isspace(data_from_db.sentence_1[i]) or str.isspace(data_from_db.sentence_2[i]) or \
                        data_from_db.sentence_1[i].startswith('萌娘百科讨论:'):
                    continue
                yield InputExample(texts=[data_from_db.sentence_1[i], data_from_db.sentence_2[i]])
            del data_from_db

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        sample_iterator = self._sample_generator()
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
#ds = KVIterableDataset(start=1, end=72708121,offset=2400000)

# # Single-process loading
#print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
#
# # Mult-process loading with two worker processes
# # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
# print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
#
# # With even more workers
# print(list(torch.utils.data.DataLoader(ds, num_workers=12)))