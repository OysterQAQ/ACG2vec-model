import json
import sys
import time

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


def transform_to_base64(image_path):
    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    preprocess = _transform(224)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    image = preprocess(Image.open(image_path))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    image = image.unsqueeze(0).permute(0, 2, 3, 1)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    return json.dumps(image.tolist())


if __name__ == "__main__":
    transform_to_base64(sys.argv[1])
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    sys.exit(0)

#print(transform_to_base64("/Volumes/Data/oysterqaq/Desktop/108557696_p0_master1200.jpg"))
