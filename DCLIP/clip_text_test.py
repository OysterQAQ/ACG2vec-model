
import requests
import torch
import urllib
import clip
from io import BytesIO
import base64
from PIL import Image
from clip.model import build_model
import  tensorflow as tf
import  numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt


keras_text_model=tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/dclip_text', compile=False)



device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
checkpoint = torch.load("/Volumes/Data/oysterqaq/Desktop/dclip_7.pt",map_location=torch.device('cpu'))

clip_model.load_state_dict(checkpoint['model_state_dict'])

text=["girl","men","cat","school"]
text = clip.tokenize(text)
print(text)
print(text.shape)
with torch.no_grad():
    a = clip_model.encode_text(text)



b=keras_text_model(tf.convert_to_tensor(text))

print(a)
print(b)


