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
model_name = 'OysterQAQ/ACGVoc2vec'
tokenizer = AutoTokenizer.from_pretrained(model_name)
sequences = ["背景"]

batch = tokenizer(sequences, padding=True,)
req = []
for s in sequences:
    b = tokenizer(s)
    b = dict(b)
    req.append(b)

req = {"instances": req}
print(req)
# Convert the batch into a proper dict

# Put the example into a list of size 1, that corresponds to the batch size
# The REST API needs a JSON that contains the key instances to declare the examples to process
# input_data = {"instances": batch}

# Query the REST API, the path corresponds to http://host:port/model_version/models_root_folder/model_name:method
r = requests.post("http://localhost:8501/v1/models/acgvoc2vec:predict", data=json.dumps(req))
print(r.text)
# Parse the JSON result. The results are contained in a list with a root key called "predictions"
# and as there is only one example, takes the first element of the list
result = json.loads(r.text)["predictions"][0]
# The returned results are probabilities, that can be positive/negative hence we take their absolute value
