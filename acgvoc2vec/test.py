import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling1D
from tensorflow import keras
from tensorflow.keras import mixed_precision

#model = keras.models.load_model('/Volumes/Data/oysterqaq/ACG2vec/docker/tf-serving/models/acgvoc2vec')
import os
import requests
import tempfile
import json
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
from transformers import TFBertForSequenceClassification, BertTokenizerFast, BertConfig

from transformers import AutoTokenizer
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
sequences = ["I love the new TensorFlow update in transformers.",""]

batch = tokenizer(sequences, padding=True)
print(tokenizer(sequences, padding= True))
req=[]
for s in sequences:
    b = tokenizer(s)
    b = dict(b)
    req.append(b)
print(req)
model.predict(req)

