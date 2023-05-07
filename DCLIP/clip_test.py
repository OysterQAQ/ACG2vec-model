import requests
import torch
import urllib
import clip
from PIL import Image
from clip.model import build_model
import  tensorflow as tf
import  numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def normalize_image(image, mean, std):
    for channel in range(3):
        image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    return image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
# text = clip.tokenize(["Manaka Komaki", "yoshioka chie", "yoshioka_chie"]).to(device)
# print(text)
# torch.save(model.state_dict(), '/Volumes/Data/oysterqaq/Desktop/clip.pt')
# state_dict = torch.load('/Volumes/Data/oysterqaq/Desktop/clip.pt')
#
# model = build_model(state_dict).to(device)
#
# if str(device) == "cpu":
#      model.float()
#
# text = clip.tokenize(["Manaka Komaki", "yoshioka chie", "yoshioka_chie"]).to(device)
# print(text)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# image_url = "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fc-ssl.duitang.com%2Fuploads%2Fitem%2F201312%2F05%2F20131205171927_uhtyi.thumb.1000_0.jpeg&refer=http%3A%2F%2Fc-ssl.duitang.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1686046217&t=7050715717cbdcf2fd999afe1bb9dd1d"
# preprocess= _transform(224)
# image = preprocess(
#   # Image.open(requests.get(image_url).content)
#    Image.open('/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg')
# ).unsqueeze(0)
#
# print(image.shape)
# image = image.permute(0, 2, 3, 1).detach().numpy()
# plt.imshow(image[0])
# plt.show()
# print(image)
# #imgs_map = tf.io.decode_image(requests.get(image_url).content, channels=3)
#
#
# imgs_map = tf.io.decode_image(tf.io.read_file('/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg'), channels=3)
# print(imgs_map.shape)
# imgs_map.set_shape((None, None, 3))
# # img = tf.image.central_crop(
# #    imgs_map
# # )
#
# def crop_center(image):
#     h, w = image.shape[-3], image.shape[-2]
#     if h > w:
#         cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
#     else:
#         cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
#     return tf.image.resize(cropped_image, (224, 224), method=tf.image.ResizeMethod.BICUBIC)
#
# #imgs_map=crop_center(imgs_map)
#
# imgs_map = tf.keras.layers.Resizing(224, 224,crop_to_aspect_ratio=True)(imgs_map)
# #imgs_map = tf.image.resize(imgs_map, [224,224], method='bicubic')
# mean= tf.constant([0.48145466, 0.4578275, 0.40821073])
# std= tf.constant([0.26862954, 0.26130258, 0.27577711])
# img = tf.cast(imgs_map, dtype=tf.float32) / 255
# img =tf.math.subtract(img,mean)
# img =tf.math.divide(img,std)
# plt.imshow(img)
# plt.show()
# print(img.shape)
# print(img)


# std=[0.26862954, 0.26130258, 0.27577711]
# mean= [0.48145466, 0.4578275, 0.40821073]
#
# img_np =np.array(imgs_map) / 255.0
# print(img_np.shape)
#
# img_np[..., 0] -= mean[0]
# img_np[..., 1] -= mean[1]
# img_np[..., 2] -= mean[2]
# if std is not None:
#     img_np[..., 0] /= std[0]
#     img_np[..., 1] /= std[1]
#     img_np[..., 2] /= std[2]
# # img_np = normalize_image(img_np,
# #                           mean=[0.48145466, 0.4578275, 0.40821073],
# #                           std=[0.26862954, 0.26130258, 0.27577711])
#
# print(img_np)

class Base64DecoderLayer(tf.keras.layers.Layer):
    """
    Convert a incoming base 64 string into an bitmap with rgb values between 0 and 1
    target_size e.g. [width,height]
    """

    def __init__(self, target_size):
        self.target_size = target_size
        self.mean = tf.constant([0.48145466, 0.4578275, 0.40821073])
        self.std = tf.constant([0.26862954, 0.26130258, 0.27577711])
        self.resize_layer=tf.keras.layers.Resizing(target_size[0],target_size[0],crop_to_aspect_ratio=True)

        super(Base64DecoderLayer, self).__init__()


    def byte_to_img(self, byte_tensor):
        # base64 decoding id done by tensorflow serve, when using b64 json
        byte_tensor = tf.io.decode_base64(byte_tensor)
        imgs_map = tf.io.decode_image(byte_tensor, channels=3)

        imgs_map.set_shape((None, None, 3))
        imgs_map = self.resize_layer(imgs_map)
        img = tf.cast(imgs_map, dtype=tf.float32) / 255
        img = tf.math.subtract(img, self.mean)
        img = tf.math.divide(img, self.std)
        return img

    def call(self, input, **kwargs):
        with tf.device("/cpu:0"):
            imgs_map = tf.map_fn(self.byte_to_img, input, dtype=tf.float32)
        return imgs_map


def build_clip_img(model_path):
    model = tf.keras.models.load_model(
        model_path, compile=False)
    print(model.inputs)
    print(model.outputs)

    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
    x = Base64DecoderLayer([224, 224])(inputs)

    x=model(x)
    clip_img = tf.keras.Model(inputs=inputs, outputs=x)
    return clip_img

model=build_clip_img('/Volumes/Data/oysterqaq/Desktop/clip_img')
import base64


pic = open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())


##转成base64后的字符串格式为 b'图片base64字符串'，前面多了 b'，末尾多了 '，所以需要截取一下

print(model(tf.stack([tf.convert_to_tensor(pic_base64)])))

image = preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg")).unsqueeze(0).to(device)


with torch.no_grad():
    image_features = clip_model.encode_image(image)
    print(image_features)

