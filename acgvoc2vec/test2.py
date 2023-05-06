from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import json
from transformers import AutoTokenizer
import tensorflow as tf
modelA = SentenceTransformer('/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch')
modelB = AutoModel.from_pretrained('/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch')

#Our sentences we like to encode
sentences = ['やかん',
    'ケトル',
    'デッサン',
    '最古シリーズ',
    '静物',
    'pixiv最古絵',
    '伝説の始まり',
    'やかんタグ最古',
    'やかん5000users入り',
    '全てはここから始まった',
    ]


#Sentences are encoded by calling model.encode()
print(modelA.tokenize(sentences))
embeddings = modelA.encode(sentences)
print(embeddings)
embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float16)
print(embeddings)
embeddings =tf.keras.layers.GlobalAveragePooling1D()(tf.expand_dims(embeddings, axis=0))
print(embeddings)
print(json.dumps(tf.keras.layers.GlobalAveragePooling1D()(tf.expand_dims(embeddings, axis=0))[0].tolist()))
