
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

keras_img_model = tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/clip_img_base64input', compile=False)
keras_text_model=tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/clip_text', compile=False)



device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)


image = preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg")).unsqueeze(0).to(device)
text=["girl","men","cat","school"]
text = clip.tokenize(text)
print(text.shape)
with torch.no_grad():
    a = clip_model.encode_image(image)
    logits_per_image, logits_per_text = clip_model(
        image.to(device),
        text.to(device)
    )
    torch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print(torch_probs)
pic = open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())
keras_img_model_feature=keras_img_model(tf.stack([tf.convert_to_tensor(pic_base64)]))
keras_text_model_feature=keras_text_model(tf.convert_to_tensor(text))


print(clip_model.encode_text(text))
print(keras_text_model_feature)

keras_img_model_feature = keras_img_model_feature / tf.norm(keras_img_model_feature, axis=-1, keepdims=True)
keras_text_model_feature = keras_text_model_feature / tf.norm(keras_text_model_feature, axis=-1, keepdims=True)
logit_scale = tf.Variable(np.ones([]) * np.log(1 / 0.07), dtype=tf.float32, name="logit_scale")

# cosine similarity as logits
logit_scale = tf.exp(logit_scale)
keras_logits_per_image = logit_scale * keras_img_model_feature @ tf.transpose(keras_text_model_feature)
keras_logits_per_text = logit_scale * keras_text_model_feature @ tf.transpose(keras_img_model_feature)

tf_probs = tf.nn.softmax(keras_logits_per_image, axis=1)
tf_probs = np.array(tf_probs)
print(tf_probs)

