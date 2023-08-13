import base64
import json
import time

import requests
import tensorflow as tf

pic = open("inputs/31726597.png", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())
pic2 = open("inputs/test4.png", "rb")
pic_base642 = base64.urlsafe_b64encode(pic2.read())
print(pic_base64)
req = {"instances": [{"b64_input_bytes":str(pic_base642).replace("b'","").replace("'","")}]}
#print(json.dumps(req))
time_start = time.time()
r = requests.post("http://192.168.123.147:18501/v1/models/cugan:predict", data=json.dumps(req))
time_end = time.time()
print('\n推理耗时：', time_end - time_start, 's')
#print(r.text)
file_path = "output/1333.png"

with open(file_path, "wb") as file:
    #binary_data = model.serve(tf.stack([tf.convert_to_tensor(pic_base64)])).numpy()
# Example binary data
    file.write(base64.urlsafe_b64decode(json.loads(r.text)["predictions"][0]))
