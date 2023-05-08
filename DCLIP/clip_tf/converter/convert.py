import hashlib
import os
import re
import sys
import urllib
import warnings
from typing import List
import logging

import numpy as np
import requests
import torch.hub

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import clip
from PIL import Image

from clip_tf.model import build_model

LOGGER = logging.Logger(__name__)

MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
}

# model input for verification
image_url = "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fc-ssl.duitang.com%2Fuploads%2Fblog%2F202108%2F04%2F20210804120908_96d67.thumb.1000_0.jpg&refer=http%3A%2F%2Fc-ssl.duitang.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1686040729&t=779493f29cb3adc4fe84511d33e61d93"
text_options = ["a diagram", "a dog", "a cat", "a neural network"]

from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import image_utils
from keras.utils import tf_utils
H_AXIS = -3
W_AXIS = -2
def smart_resize(x, size, interpolation="bilinear"):
    """Resize images to a target size without aspect ratio distortion.

    Warning: `tf.keras.preprocessing.image.smart_resize` is not recommended for
    new code. Prefer `tf.keras.layers.Resizing`, which provides the same
    functionality as a preprocessing layer and adds `tf.RaggedTensor` support.
    See the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers)
    for an overview of preprocessing layers.

    TensorFlow image datasets typically yield images that have each a different
    size. However, these images need to be batched before they can be
    processed by Keras layers. To be batched, images need to share the same
    height and width.

    You could simply do:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: tf.image.resize(img, size))
    ```

    However, if you do this, you distort the aspect ratio of your images, since
    in general they do not all have the same aspect ratio as `size`. This is
    fine in many cases, but not always (e.g. for GANs this can be a problem).

    Note that passing the argument `preserve_aspect_ratio=True` to `resize`
    will preserve the aspect ratio, but at the cost of no longer respecting the
    provided target size. Because `tf.image.resize` doesn't crop images,
    your output images will still have different sizes.

    This calls for:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: smart_resize(img, size))
    ```

    Your output images will actually be `(200, 200)`, and will not be distorted.
    Instead, the parts of the image that do not fit within the target size
    get cropped out.

    The resizing process is:

    1. Take the largest centered crop of the image that has the same aspect
    ratio as the target size. For instance, if `size=(200, 200)` and the input
    image has size `(340, 500)`, we take a crop of `(340, 340)` centered along
    the width.
    2. Resize the cropped image to the target size. In the example above,
    we resize the `(340, 340)` crop to `(200, 200)`.

    Args:
      x: Input image or batch of images (as a tensor or NumPy array). Must be in
        format `(height, width, channels)` or `(batch_size, height, width,
        channels)`.
      size: Tuple of `(height, width)` integer. Target size.
      interpolation: String, interpolation to use for resizing. Defaults to
        `'bilinear'`. Supports `bilinear`, `nearest`, `bicubic`, `area`,
        `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.

    Returns:
      Array with shape `(size[0], size[1], channels)`. If the input image was a
      NumPy array, the output is a NumPy array, and if it was a TF tensor,
      the output is a TF tensor.
    """
    if len(size) != 2:
        raise ValueError(
            f"Expected `size` to be a tuple of 2 integers, but got: {size}."
        )
    img = tf.convert_to_tensor(x)
    if img.shape.rank is not None:
        if img.shape.rank < 3 or img.shape.rank > 4:
            raise ValueError(
                "Expected an image array with shape `(height, width, "
                "channels)`, or `(batch_size, height, width, channels)`, but "
                f"got input with incorrect rank, of shape {img.shape}."
            )
    shape = tf.shape(img)
    height, width = shape[-3], shape[-2]
    target_height, target_width = size
    if img.shape.rank is not None:
        static_num_channels = img.shape[-1]
    else:
        static_num_channels = None

    crop_height = tf.cast(
        tf.cast(width * target_height, "float32") / target_width, "int32"
    )
    crop_width = tf.cast(
        tf.cast(height * target_width, "float32") / target_height, "int32"
    )

    # Set back to input height / width if crop_height / crop_width is not
    # smaller.
    crop_height = tf.minimum(height, crop_height)
    crop_width = tf.minimum(width, crop_width)

    crop_box_hstart = tf.cast(
        tf.cast(height - crop_height, "float32") / 2, "int32"
    )
    crop_box_wstart = tf.cast(
        tf.cast(width - crop_width, "float32") / 2, "int32"
    )

    if img.shape.rank == 4:
        crop_box_start = tf.stack([0, crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([-1, crop_height, crop_width, -1])
    else:
        crop_box_start = tf.stack([crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([crop_height, crop_width, -1])

    img = tf.slice(img, crop_box_start, crop_box_size)
    img = tf.image.resize(images=img, size=size, method=interpolation,antialias=True)
    # Apparent bug in resize_images_v2 may cause shape to be lost
    if img.shape.rank is not None:
        if img.shape.rank == 4:
            img.set_shape((None, None, None, static_num_channels))
        if img.shape.rank == 3:
            img.set_shape((None, None, static_num_channels))
    if isinstance(x, np.ndarray):
        return img.numpy()
    return img
def convert_inputs(inputs, dtype=None):
    if isinstance(inputs, dict):
        raise ValueError(
            "This layer can only process a tensor representing an image or "
            f"a batch of images. Received: type(inputs)={type(inputs)}."
            "If you need to pass a dict containing "
            "images, labels, and bounding boxes, you should "
            "instead use the preprocessing and augmentation layers "
            "from `keras_cv.layers`. See docs at "
            "https://keras.io/api/keras_cv/layers/"
        )
    inputs = utils.ensure_tensor(inputs, dtype=dtype)
    return inputs
class Resizing(base_layer.Layer):
    """A preprocessing layer which resizes images.

    This layer resizes an image input to a target height and width. The input
    should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
    format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype.
    By default, the layer will output floats.

    This layer can be called on tf.RaggedTensor batches of input images of
    distinct sizes, and will resize the outputs to dense tensors of uniform
    size.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        interpolation: String, the interpolation method.
            Defaults to `"bilinear"`.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
        crop_to_aspect_ratio: If True, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
    """

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self._interpolation_method = image_utils.get_interpolation(
            interpolation
        )
        super().__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("Resizing").set(True)

    def call(self, inputs):
        # tf.image.resize will always output float32
        # and operate more efficiently on float32
        # unless interpolation is nearest, in which case ouput type matches
        # input type.
        if self.interpolation == "nearest":
            input_dtype = self.compute_dtype
        else:
            input_dtype = tf.float32
        inputs = convert_inputs(inputs, dtype=input_dtype)
        size = [self.height, self.width]
        if self.crop_to_aspect_ratio:

            def resize_to_aspect(x):
                if tf_utils.is_ragged(inputs):
                    x = x.to_tensor()
                return smart_resize(
                    x, size=size, interpolation=self._interpolation_method
                )

            if tf_utils.is_ragged(inputs):
                size_as_shape = tf.TensorShape(size)
                shape = size_as_shape + inputs.shape[-1:]
                spec = tf.TensorSpec(shape, input_dtype)
                outputs = tf.map_fn(
                    resize_to_aspect, inputs, fn_output_signature=spec
                )
            else:
                outputs = resize_to_aspect(inputs)
        else:
            outputs = tf.image.resize(
                inputs, size=size, method=self._interpolation_method,antialias=True
            )
        return tf.cast(outputs, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "interpolation": self.interpolation,
            "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Base64DecoderLayer(tf.keras.layers.Layer):
    """
    Convert a incoming base 64 string into an bitmap with rgb values between 0 and 1
    target_size e.g. [width,height]
    """

    def __init__(self, target_size):
        self.target_size = target_size
        self.mean = tf.constant([0.48145466, 0.4578275, 0.40821073])
        self.std = tf.constant([0.26862954, 0.26130258, 0.27577711])
        self.resize_layer=Resizing(target_size[0],target_size[0],    interpolation='bicubic',
crop_to_aspect_ratio=True)

        super(Base64DecoderLayer, self).__init__()


    def byte_to_img(self, byte_tensor):
        # base64 decoding id done by tensorflow serve, when using b64 json
        byte_tensor = tf.io.decode_base64(byte_tensor)
        imgs_map = tf.io.decode_image(byte_tensor, channels=3)

        imgs_map.set_shape((None, None, 3))
        #imgs_map=tf.image.resize(imgs_map, [224, 224],method='bicubic',antialias=True,)
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
    model = keras.models.load_model(
        model_path, compile=False)
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
    x = Base64DecoderLayer([224, 224])(inputs)
    x=model(x)
    clip_img = keras.Model(inputs=inputs, outputs=x)
    clip_img.save(model_path+"_base64input")
    return clip_img

def build_clip_text(model):
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
    x = Base64DecoderLayer([224, 224])(inputs)
    model.visual(x)
    clip_img = keras.Model(inputs=inputs, outputs=x)
    return clip_img

def download_statedict(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    is_downloaded = False
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            is_downloaded = True
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    if not is_downloaded:
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    state_dict = torch.jit.load(
        download_target,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    ).state_dict()
    return state_dict


def load_pytorch_weights(model: keras.Model, state_dict: dict, verbose=False):
    tf_dict = {v.name.replace('.', '/'): v for v in model.weights}

    # (from, to) replacement pairs to convert tensor names from tf to pytorch
    def prepare_key(key):
        repl = [
            ('clip/', ''),
            (':0', ''),
            ('token_embedding', 'token_embedding/weight'),
            # batch norm
            ('gamma', 'weight'), ('beta', 'bias'),
            ('moving_mean', 'running_mean'), ('moving_variance', 'running_var'),
            # conv
            ('kernel', 'weight'),
            # attention (resnet)
            ('mha/key', 'k_proj'), ('mha/query', 'q_proj'), ('mha/value', 'v_proj'), ('mha/attention_output', 'c_proj'),
            # attention (transformer)
            ('attn/key', 'attn/k_proj'), ('attn/query', 'attn/q_proj'), ('attn/value', 'attn/v_proj'),
            ('attn/attention_output', 'attn/out_proj'),
            ('/', '.'),
        ]
        for rep in repl:
            key = key.replace(*rep)
        return key

    # convert existing keys in state_dict, e.g. splitting tensor into multiple
    initial_converters = {
        "in_proj_weight": lambda key, source: dict(
            zip(
                [
                    key.replace('in_proj_weight', 'q_proj.weight'),
                    key.replace('in_proj_weight', 'k_proj.weight'),
                    key.replace('in_proj_weight', 'v_proj.weight')
                ],
                torch.split(source, source.shape[0] // 3, dim=0)
            )
        ),
        "in_proj_bias": lambda key, source: dict(
            zip(
                [
                    key.replace('in_proj_bias', 'q_proj.bias'),
                    key.replace('in_proj_bias', 'k_proj.bias'),
                    key.replace('in_proj_bias', 'v_proj.bias')
                ],
                torch.split(source, source.shape[0] // 3, dim=0)
            )
        ),
    }

    def apply_initial_converters():
        state_dict_keys = list(state_dict.keys())
        for k, fn in initial_converters.items():
            r = re.compile(k)
            matched_keys = filter(r.search, state_dict_keys)
            for matched_key in matched_keys:
                res = fn(matched_key, state_dict[matched_key])
                for key, val in res.items():
                    state_dict[key] = val
                del state_dict[matched_key]

    apply_initial_converters()

    # convert keys when their destination is known, maps to tensor
    def multi_head_attention_weight_conversion(source, dest):
        res = source.T.reshape(tuple(np.array(dest.shape)))
        return res

    contextual_converters = {
        '_proj.weight': multi_head_attention_weight_conversion,
        '_proj.bias': multi_head_attention_weight_conversion,
    }

    def apply_contextual_converters(state_dict_key, source_weights, dest):
        for k, fn in contextual_converters.items():
            r = re.compile(k)
            if r.search(state_dict_key):
                return fn(source_weights, dest), k

        return source_weights, ''

    mapped_keys = set()

    for tf_key in tqdm(tf_dict.keys(), desc="Copying weights"):
        state_dict_key = prepare_key(tf_key)

        dest = tf_dict[tf_key]

        if state_dict_key not in state_dict:
            candidates = [key for key, src in state_dict.items()
                          if key not in mapped_keys and tf.reduce_prod(dest.shape) == np.prod(src.shape)]
            raise ValueError(
                f"'{tf_key}': Missing var {state_dict_key} in state_dict. shape={dest.shape}. candidates={candidates}")

        if state_dict_key in mapped_keys:
            raise ValueError(
                f"'{tf_key}': Duplicate var {state_dict_key} has already been assigned.")
        mapped_keys.add(state_dict_key)

        source_weights = state_dict[state_dict_key].cpu().detach()

        source_weights, converter_name = apply_contextual_converters(state_dict_key, source_weights, dest)

        compatible_weights_default = tf.reduce_all(dest.shape == source_weights.shape)
        compatible_weights_transposed = False
        if len(source_weights.shape) == 4:
            compatible_weights_transposed = tf.reduce_all(dest.shape == source_weights.permute(2, 3, 1, 0).shape)
            if compatible_weights_default and compatible_weights_transposed and len(source_weights.shape) > 1:
                print(
                    f"'{state_dict_key}' -> '{tf_key}': unclear whether shape {source_weights.shape} should be transposed to {dest.shape}",
                    file=sys.stderr)
            if compatible_weights_transposed and not compatible_weights_default:
                source_weights = source_weights.permute(2, 3, 1, 0)
                pass
        if len(source_weights.shape) == 2:
            compatible_weights_transposed = tf.reduce_all(dest.shape == source_weights.permute(1, 0).shape)
            if compatible_weights_default and compatible_weights_transposed and len(source_weights.shape) > 1:
                print(
                    f"'{state_dict_key}' -> '{tf_key}': unclear whether shape {source_weights.shape} should be transposed to {dest.shape}",
                    file=sys.stderr)
            if compatible_weights_transposed and not compatible_weights_default:
                source_weights = source_weights.permute(1, 0)
                pass

        if not (compatible_weights_default or compatible_weights_transposed):
            print(
                f"'{state_dict_key}' -> '{tf_key}': source shape {source_weights.shape} has to be equal to dest shape {dest.shape}",
                file=sys.stderr)

        # assert compatible_weights, f"'{key}': source shape {source_weights.shape} has to be equal to dest shape {dest.shape}"
        if verbose:
            print(
                f"convert '{state_dict_key}' -> '{tf_key}' {source_weights.shape} converter='{converter_name}' {'transposed' if compatible_weights_transposed and not compatible_weights_default else ''}")
        assert isinstance(dest, object)
        dest.assign(source_weights.numpy().astype(np.float32))

    # unmapped_keys = set(state_dict.keys()).difference(mapped_keys)
    # if len(unmapped_keys) > 0:
    #     print("Unmapped keys in state_dict:")
    #     for k in unmapped_keys:
    #         print(f"missing '{k}' -> '?'")
    #
    #     exit(0)


def verify(model_name: str, keras_model: keras.Model, image_url: str, text_options: List[str], verbose: bool = False):
    # load pytorch clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    image = preprocess(
        Image.open(requests.get(image_url, stream=True).raw)
    ).unsqueeze(0)
    # img = tf.cast(img, dtype=tf.float32) / 255

    text = clip.tokenize(text_options)

    with torch.no_grad():
        a=model.encode_image(image)
        logits_per_image, logits_per_text = model(
            image.to(device),
            text.to(device)
        )
        torch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # tf2
    image = image.permute(0, 2, 3, 1).detach().numpy()
    print(image.shape)
    text = text.unsqueeze(0)  # grml... keras doesnt like different cardinality in batch dim
    text = text.detach().numpy().astype(np.int32)
    logits_per_image, logits_per_text = keras_model.predict((image, text))
    tf_probs = tf.nn.softmax(logits_per_image, axis=1)
    tf_probs = np.array(tf_probs)
    b=keras_model.visual(image)
    print(np.isclose(b.numpy(),
                    a,
                     atol=1e-5))


    if verbose:
        print(f"Classify image: {image_url}")
        print(f"Text options: {text_options}")
        print(f"Pytorch: {torch_probs}")
        print(f"Tensorflow: {tf_probs}")

    assert np.abs(
        torch_probs - tf_probs).sum() < 1e-3, f"PyTorch and Tensorflow results should be almost equal: torch_probs={torch_probs}, tf_probs={tf_probs}"


def get_cache_path(model: str, cache_path: str, type: str = None) -> str:
    sanitized_model_name = model.replace("/", "_")
    if type is not None:
        sanitized_model_name = f"{type}_{sanitized_model_name}"
    return cache_path.format(model=sanitized_model_name)


def convert(model_name: str, output: str, image_output: str = None, text_output: str = None, all: bool = False,
            should_verify: bool = True,model_path: str = None,full_output_path: str = None,img_output_path: str = None,text_output_path: str = None,img_base64: bool= False):
    model_url = MODELS[model_name]
    if model_path is not None:
        state_dict=torch.load(model_path)
    else:
        state_dict = download_statedict(model_url)
    model = build_model(state_dict)

    # predict to build shapes (model.build doesnt work, as it only supports float inputs)
    model.predict((
        np.ones((1, model.image_resolution, model.image_resolution, 3), np.float32),
        np.ones((1, 4, 77), np.int32)
    ))
    load_pytorch_weights(model, state_dict, verbose=False)

    if should_verify:
        LOGGER.info("Verifying converted model...")
        verify(model_name, model, image_url, text_options, verbose=True)

    # create SavedModel
    output_filename = get_cache_path(model_name, output)
    LOGGER.info(f"Saving model: {output_filename}")
    model.save(full_output_path)

    # load and test model
    if should_verify:
        LOGGER.info("Verifying saved model...")
        saved_model = tf.keras.models.load_model(full_output_path)
        saved_model.summary()
        verify(model_name, saved_model, image_url, text_options, verbose=True)

    # Dedicated export of image or text encoder
    if image_output is not None or all:
        image_output_filename = get_cache_path(model_name, image_output) if image_output else get_cache_path(model_name,
                                                                                                             output,
                                                                                                             "image")
        LOGGER.info(f"Saving image encoder model: {image_output_filename}")
        #是否转为base64输入
        model.visual.save(img_output_path)
        if img_base64:
            build_clip_img(img_output_path)





    text_output = text_output or (output.format(model="text_{model}") if all else None)
    if text_output is not None:
        text_output_filename = get_cache_path(model_name, text_output) if text_output else get_cache_path(model_name,
                                                                                                          output,
                                                                                                          "text")
        LOGGER.info(f"Saving text encoder model: {text_output_filename}")
        inputs = keras.Input(shape=(None,), name="text", dtype=tf.int32)

        # we have to create a layer to capture all variables used inside of encode_text as well. TODO: more elegant solution
        class TextEncoder(tf.keras.layers.Layer):
            def __init__(self, model: tf.keras.models.Model):
                super().__init__()
                model.visual=None
                self.model = model

            def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
                return model.encode_text(inputs)

        outputs = TextEncoder(model)(inputs)

        text_encoder = keras.models.Model(inputs=inputs, outputs=outputs)
        text_encoder.summary()
        text_encoder.save(text_output_path)

