import requests
import torch
import urllib
import clip
from io import BytesIO
import  numpy as np

np.set_printoptions(suppress=True)

from PIL import Image
from clip.model import build_model
import  tensorflow as tf
import  numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
import base64
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

keras_model = tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/clip_img_base64input', compile=False)

image = preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg")).unsqueeze(0).to(device)


pic = open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())

with torch.no_grad():
    image_features = clip_model.encode_image(image)
    a=image_features


b=keras_model(tf.stack([tf.convert_to_tensor(pic_base64)]))
print(a.detach().numpy())
print(b.numpy())
print(np.isclose(a,
                    b.numpy(),
                     atol=1e-1))
