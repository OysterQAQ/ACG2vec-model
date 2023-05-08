import requests
import torch
import urllib
import clip
from io import BytesIO
import keras_cv

from PIL import Image
from clip.model import build_model
import  tensorflow as tf
import  numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
import base64
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")
from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import image_utils
from keras.utils import tf_utils
H_AXIS = -3
W_AXIS = -2
def smart_resize(x, size, interpolation='bicubic'):
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
        #imgs_map=tf.image.resize(imgs_map, [224, 224],method='bicubic',antialias=True,preserve_aspect_ratio=True,)
        imgs_map = self.resize_layer(imgs_map)
        #imgs_map =keras_cv.layers.Resizing( 224, 224,interpolation='bicubic',crop_to_aspect_ratio=True)(imgs_map)
        img = tf.cast(imgs_map, dtype=tf.float32) / 255
        img = tf.math.subtract(img, self.mean)
        img = tf.math.divide(img, self.std)
        return img

    def call(self, input, **kwargs):
        with tf.device("/cpu:0"):
            imgs_map = tf.map_fn(self.byte_to_img, input, dtype=tf.float32)
        return imgs_map
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

preprocess= _transform(224)
image = preprocess(
  # Image.open(requests.get(image_url).content)
   Image.open('/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg')
).unsqueeze(0)

print(image.shape)
image = image.permute(0, 2, 3, 1).detach().numpy()
plt.imshow(image[0])
plt.show()
print(image)


pic = open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())

img = Base64DecoderLayer([224, 224])(tf.stack([tf.convert_to_tensor(pic_base64)]))

plt.imshow(img[0])
plt.show()
plt.imshow(img[0]-image[0])
plt.show()
print(img.shape)
print((img[0]-image[0])[0])

from PIL import Image
im = Image.open(r"/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg")


min_width = 224

# 计算新高度
width, height = im.size
new_height = int(height * (min_width / width))

# 缩小图像并保持原始比例
im = im.resize((min_width, new_height),Image.BICUBIC)
width, height = im.size
left = (width - min_width) // 2
top = (height - min_width) // 2
right = (width + min_width) // 2
bottom = (height + min_width) // 2

# 裁剪图像
im = im.crop((left, top, right, bottom))

im=tf.keras.utils.img_to_array(im)
mean = tf.constant([0.48145466, 0.4578275, 0.40821073])
std = tf.constant([0.26862954, 0.26130258, 0.27577711])
im = tf.cast(im, dtype=tf.float32) / 255
im = tf.math.subtract(im, mean)
im = tf.math.divide(im, std)
plt.imshow(im-image[0])
plt.show()


print(np.isclose(image[0][0],
                    img[0].numpy()[0],
                     atol=1e-5))
import base64