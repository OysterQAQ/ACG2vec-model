import os


import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose


# tf.compat.v1.disable_eager_execution()


class SEBlock(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, in_channels, reduction=8, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters=(in_channels // reduction), kernel_size=1, strides=1, padding="valid",
                            use_bias=bias, input_shape=(None, None, in_channels), name=self.name + ".conv1")
        self.conv2 = Conv2D(filters=in_channels, kernel_size=1, strides=1, padding="valid", use_bias=bias,
                            input_shape=(None, None, (in_channels // reduction)), name=self.name + ".conv2")

    def call(self, inputs):  # Defines the computation from inputs to outputs
        x0 = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True)
        x0 = self.conv1(x0)
        x0 = tf.keras.layers.ReLU()(x0)
        x0 = self.conv2(x0)
        x0 = tf.keras.activations.sigmoid(x0)
        x = inputs * x0
        return x

    def mean_call(self, x, x0):
        x0 = self.conv1(x0)
        x0 = tf.keras.layers.ReLU()(x0)
        x0 = self.conv2(x0)
        x0 = tf.keras.activations.sigmoid(x0)
        x = x * x0
        return x


class UNetConv(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, in_channels, mid_channels, out_channels, se, **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.Sequential(
            [
                Conv2D(filters=mid_channels, kernel_size=3, strides=1, padding="valid",
                       input_shape=(None, None, in_channels), name=self.name + ".conv." + "0"),
                LeakyReLU(alpha=0.1),
                Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="valid",
                       input_shape=(None, None, mid_channels), name=self.name + ".conv." + "2"),
                LeakyReLU(alpha=0.1),
            ]
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True, name=self.name + "." + "seblock")
        else:
            self.seblock = None

    def call(self, inputs):
        x = self.conv(inputs)
        if self.seblock is not None:
            x = self.seblock(x)
        return x


class UNet1(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, in_channels, out_channels, deconv, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = UNetConv(in_channels, 32, 64, se=False, name=self.name + ".conv1")
        self.conv1_down = Conv2D(filters=64, kernel_size=2, strides=2, padding="valid", input_shape=(None, None, 64),
                                 name=self.name + ".conv1_down")
        self.conv2 = UNetConv(64, 128, 64, se=True, name="unet.conv2")
        self.conv2_up = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 64), name=self.name + ".conv2_up")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", input_shape=(None, None, 64),
                            name=self.name + ".conv3")
        if deconv:
            self.conv_bottom = Conv2DTranspose(filters=out_channels, kernel_size=4, strides=2, padding="same",
                                               input_shape=(None, None, 64), name=self.name + ".conv_bottom")
        else:
            self.conv_bottom = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="valid",
                                      input_shape=(None, None, 64), name=self.name + ".conv_bottom")
        # for m in self.submodules:
        #     if isinstance(m, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
        #         tf.keras.initializers.HeNormal()(m.weights)
        #     elif isinstance(m, tf.keras.layers.Dense):
        #         tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)(m.weights)
        #         if m.bias is not None:
        #             tf.keras.initializers.Zeros()(m.bias)

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        # x1 = tf.pad(x1, (-4, -4, -4, -4))
        # tf.pad无法完成负padding
        x1 = x1[:, 4:-4, 4:-4, :]
        # x1 = tf.pad(x1, tf.constant([[-4, -4], [-4, -4],[0,0]]))
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)
        x3 = self.conv3(x1 + x2)
        x3 = LeakyReLU(alpha=0.1)(x3)
        z = self.conv_bottom(x3)
        return z

    def call_a(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        # x1 = tf.pad(x1, (-4, -4, -4, -4))
        x1 = x1[:, 4:-4, 4:-4, :]
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def call_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)
        x3 = self.conv3(x1 + x2)
        x3 = LeakyReLU(alpha=0.1)(x3)
        z = self.conv_bottom(x3)
        return z


class UNet2(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, in_channels, out_channels, deconv, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = UNetConv(in_channels, 32, 64, se=False, name=self.name + ".conv1")
        self.conv1_down = Conv2D(filters=64, kernel_size=2, strides=2, padding="valid", input_shape=(None, None, 64),
                                 name=self.name + ".conv1_down")
        self.conv2 = UNetConv(64, 64, 128, se=True, name=self.name + ".conv2")
        self.conv2_down = Conv2D(filters=128, kernel_size=2, strides=2, padding="valid", input_shape=(None, None, 128),
                                 name=self.name + ".conv2_down")
        self.conv3 = UNetConv(128, 256, 128, se=True, name=self.name + ".conv3")
        self.conv3_up = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 128), name=self.name + ".conv3_up")
        self.conv4 = UNetConv(128, 64, 64, se=True, name=self.name + ".conv4")
        self.conv4_up = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 64), name=self.name + ".conv4_up")
        self.conv5 = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", input_shape=(None, None, 64),
                            name=self.name + ".conv5")

        if deconv:
            self.conv_bottom = Conv2DTranspose(filters=out_channels, kernel_size=4, strides=2, padding="valid",
                                               input_shape=(None, None, 64), name=self.name + ".conv_bottom")
        else:
            self.conv_bottom = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="valid",
                                      input_shape=(None, None, 64), name=self.name + ".conv_bottom")
        # for m in self.submodules:
        #     if isinstance(m, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
        #         tf.keras.initializers.HeNormal()(m.weights)
        #     elif isinstance(m, tf.keras.layers.Dense):
        #         tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)(m.weights)
        #         if m.bias is not None:
        #             tf.keras.initializers.Zeros()(m.bias)

    def call(self, inputs, alpha=1):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        # x1 = tf.pad(x1, (-16, -16, -16, -16))
        x1 = x1[:, 16:-16, 16:-16, :]
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        # x2 = tf.pad(x2, (-4, -4, -4, -4))
        x2 = x2[:, 4:-4, 4:-4, :]
        x3 = LeakyReLU(alpha=0.1)(x3)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = LeakyReLU(alpha=0.1)(x3)
        x4 = self.conv4(x2 + x3)
        x4 *= alpha
        x4 = self.conv4_up(x4)
        x4 = LeakyReLU(alpha=0.1)(x4)
        x5 = self.conv5(x1 + x4)
        x5 = LeakyReLU(alpha=0.1)(x5)
        z = self.conv_bottom(x5)
        z = z[:, 3:-3, 3:-3, :]
        return z

    def call_a(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        # x1 = tf.pad(x1, (-16, -16, -16, -16))
        x1 = x1[:, 16:-16, 16:-16, :]
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def call_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        # x2 = tf.pad(x2, (-4, -4, -4, -4))
        x2 = x2[:, 4:-4, 4:-4, :]
        x3 = LeakyReLU(alpha=0.1)(x3)
        x3 = self.conv3.conv(x3)
        return x2, x3

    def call_c(self, x2, x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = LeakyReLU(alpha=0.1)(x3)
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def call_d(self, x1, x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = LeakyReLU(alpha=0.1)(x4)
        x5 = self.conv5(x1 + x4)
        x5 = LeakyReLU(alpha=0.1)(x5)
        z = self.conv_bottom(x5)
        return z


class UpCunet2x(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, alpha=0.7, pro=True, half=False, **kwargs):
        super().__init__(**kwargs)
        self.unet1 = UNet1(in_channels, out_channels, deconv=True, name=self.name + ".unet1")
        self.unet2 = UNet2(in_channels, out_channels, deconv=False, name=self.name + ".unet2")
        self.alpha = alpha
        self.pro = pro
        self.half = half

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32) / (
                255 / 0.7) + 0.15

        # 修改输入 tensorflow默认为 nhwc 修改为nchw
        # x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        #x = inputs
        # n, c, h0, w0 = x.shape
        n, h0, w0, c = x.shape
        # 根据shape自动设置tile_mode
        if h0 > w0:
            tile_mode = h0 // 1080
        else:
            tile_mode = w0 // 1080

        if (tf.dtypes.float16 == x.dtype):
            if_half = True
        else:
            if_half = False
        if (tile_mode == 0):  # 不tile
            # 将奇数变为下一个最近的偶数
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            # 填充h w 维度 （h前填充，h后填充，w前填充，w后填充）
            x = tf.pad(x, [[0, 0], [18, 18 + ph - h0],  [18, 18 + pw - w0],[0, 0]], 'REFLECT')  # 需要保证被2整除
            x = self.unet1(x)
            x0 = self.unet2(x, self.alpha)
            # x = tf.pad(x, (-20, -20, -20, -20))
            x = x[:, 20:-20, 20:-20, :]

            x = tf.math.add(x0, x)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
            if (self.pro):
                return tf.cast(tf.clip_by_value(tf.math.round(((x - 0.15) * (255 / 0.7))), clip_value_min=0, clip_value_max=255),dtype=tf.dtypes.uint8)
            else:
                return tf.cast(tf.clip_by_value(tf.math.round(x * 255), clip_value_min=0, clip_value_max=255),dtype=tf.dtypes.uint8)
        elif (tile_mode == 1):  # 对长边减半
            if (w0 >= h0):
                crop_size_w = ((w0 - 1) // 4 * 4 + 4) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_h = (h0 - 1) // 2 * 2 + 2  # 能被2整除
            else:
                crop_size_h = ((h0 - 1) // 4 * 4 + 4) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_w = (w0 - 1) // 2 * 2 + 2  # 能被2整除
            crop_size = (crop_size_h, crop_size_w)
        elif (tile_mode >= 2):
            tile_mode = min(min(h0, w0) // 128, int(tile_mode))  # 最小短边为128*128
            t2 = tile_mode * 2
            crop_size = (((h0 - 1) // t2 * t2 + t2) // tile_mode, ((w0 - 1) // t2 * t2 + t2) // tile_mode)
        else:
            print("tile_mode config error")
            os._exit(233)

        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x = tf.pad(x, [[0, 0], [18, 18 + pw - w0], [18, 18 + ph - h0], [0, 0]], 'REFLECT')
        n, h, w, c = x.shape
        if (if_half):
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)
        n_patch = 0
        tmp_dict = {}
        for i in range(0, h - 36, crop_size[0]):
            tmp_dict[i] = {}
            for j in range(0, w - 36, crop_size[1]):
                x_crop = x[:, i:i + crop_size[0] + 36, j:j + crop_size[1] + 36, :]
                n, h1, w1, c1 = x_crop.shape
                tmp0, x_crop = self.unet1.call_a(x_crop)
                # if (if_half):  # torch.HalfTensor/torch.cuda.HalfTensor
                #     tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3), keepdim=True).half()
                # else:
                #     tmp_se_mean = torch.mean(x_crop, dim=(2, 3), keepdim=True)
                tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
                tmp_dict[i][j] = (tmp0, x_crop)
        se_mean0 /= n_patch
        if (if_half):
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)
        for i in range(0, h - 36, crop_size[0]):
            for j in range(0, w - 36, crop_size[1]):
                tmp0, x_crop = tmp_dict[i][j]
                x_crop = self.unet1.conv2.seblock.mean_call(x_crop, se_mean0)
                opt_unet1 = self.unet1.call_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
                # opt_unet1 = tf.pad(opt_unet1, (-20, -20, -20, -20))

                opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
                tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2)
        se_mean1 /= n_patch
        if (if_half):
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)
        for i in range(0, h - 36, crop_size[0]):
            for j in range(0, w - 36, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x2 = tmp_dict[i][j]
                tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
                tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True)
                se_mean0 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2, tmp_x3)
        se_mean0 /= n_patch
        if (if_half):
            se_mean1 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)
        for i in range(0, h - 36, crop_size[0]):
            for j in range(0, w - 36, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tmp_dict[i][j]
                tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3, se_mean0)
                tmp_x4 = self.unet2.call_c(tmp_x2, tmp_x3) * self.alpha
                # if (if_half):  # torch.HalfTensor/torch.cuda.HalfTensor
                #     tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3), keepdim=True).half()
                # else:
                #     tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3), keepdim=True)
                tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x4)
        se_mean1 /= n_patch
        res = []
        for i in range(0, h - 36, crop_size[0]):
            temp = []
            for j in range(0, w - 36, crop_size[1]):
                x, tmp_x1, tmp_x4 = tmp_dict[i][j]
                tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1)
                x0 = self.unet2.call_d(tmp_x1, tmp_x4)
                del tmp_dict[i][j]
                x = tf.math.add(x0, x)  # x0是unet2的最终输出

                if (self.pro):
                    # tensorflow的tensor不支持修改
                    # res[:, i * 2:i * 2 + h1 * 2 - 72, j * 2:j * 2 + w1 * 2 - 72,:].assign(tf.clip_by_value(tf.math.round(
                    #         (x - 0.15) * (255 / 0.7)),clip_value_min=0, clip_value_max=255))
                    temp.append(
                        tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)), clip_value_min=0, clip_value_max=255),dtype=tf.dtypes.uint8))
                else:
                    temp.append(tf.cast(tf.clip_by_value(tf.math.round((x * 255)), clip_value_min=0, clip_value_max=255),dtype=tf.dtypes.uint8))
                    # res[:,  i * 2:i * 2 + h1 * 2 - 72, j * 2:j * 2 + w1 * 2 - 72,:].assign(tf.clip_by_value(tf.math.round((x * 255)),clip_value_min=0, clip_value_max=255))
            # stack
            temp = tf.concat(
                temp, axis=2
            )
            res.append(temp)
        res = tf.concat(
            res, axis=1
        )

        del tmp_dict
        # torch.cuda.empty_cache()
        if (w0 != pw or h0 != ph): res = res[:, :h0 * 2, :w0 * 2, :]
        return res

    def np2tensor(self, np_frame):
        if (self.pro):
            # half半精度
            if (self.half == False):
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float32)),
                                      axis=0) / (
                        255 / 0.7) + 0.15
            else:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float16)),
                                      axis=0) / (
                        255 / 0.7) + 0.15
        else:
            if (self.half == False):
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float32)),
                                      axis=0) / 255
            else:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float16)),
                                      axis=0) / 255


class Base64DecoderLayer(tf.keras.layers.Layer):
    """
    Convert a incoming base 64 string into an bitmap with rgb values between 0 and 1
    target_size e.g. [width,height]
    """

    def __init__(self):
        # self.target_size = target_size
        super(Base64DecoderLayer, self).__init__()

    def byte_to_img(self, byte_tensor):
        # base64 decoding id done by tensorflow serve, when using b64 json
        byte_tensor = tf.io.decode_base64(byte_tensor)
        imgs_map = tf.io.decode_image(byte_tensor, channels=3)
        #imgs_map.set_shape((None, None, 3))
        # img = tf.image.resize(imgs_map, self.target_size)
        img = tf.cast(imgs_map, dtype=tf.float32)/ (
        255 / 0.7) + 0.15
        return img

    def call(self, input, **kwargs):
        with tf.device("/cpu:0"):
            imgs_map = tf.map_fn(self.byte_to_img, input, dtype=tf.float32)
        return imgs_map


def load_pt_weight_to_tf(weight_path, tf_model, map_json_file_path):
    import torch
    import json

    with open(map_json_file_path) as json_file:

        map_json = json.load(json_file)

    pt_weights = torch.load(weight_path, map_location="cpu")
    tf_new_weights = []
    for tf_weight in tf_model.weights:
        pt_weight = pt_weights[map_json[tf_weight.name]]
        if "kernel" in tf_weight.name:
            pt_weight = pt_weight.numpy()
            pt_weight = tf.transpose(pt_weight, perm=[3, 2, 1, 0])
        tf_new_weights.append(pt_weight.numpy())

    tf_model.set_weights(tf_new_weights)


# inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
# x=Base64DecoderLayer()(inputs)
# cunet = UpCunet2x(name="UpCunet2x")
# y=cunet(x)
# model = tf.keras.Model(inputs=inputs, outputs=y)
# model.save("/Volumes/Data/oysterqaq/Desktop/cugan")
# cunet.build(input_shape=(None,None,3))
# input = tf.keras.Input(shape=(None, None, 3), name="input")
# y=cunet(input)
# model = keras.Model(inputs=input, outputs=y)

input_shape = (1, 550, 550, 3)
x = tf.random.normal(input_shape)
cunet = UpCunet2x(name="UpCunet2x")
y = cunet(x)
load_pt_weight_to_tf("weights_pro/pro-denoise3x-up2x.pth", cunet, "weight_map.json")

imgs_map = tf.io.decode_image(tf.io.read_file("inputs/31726597.png"), channels=3)


y = cunet(tf.expand_dims(imgs_map, axis=0))

tf.keras.utils.save_img(
    "output/1-1.png", tf.squeeze(y)
)
# cunet.save("/Volumes/Home/oysterqaq/Desktop/cugan_pro-no-denoise-up2x")
# load_pt_weight_to_tf("weights_pro/pro-conservative-up2x.pth", cunet, "weight_map.json")
# cunet.save("/Volumes/Home/oysterqaq/Desktop/cugan_pro-conservative-up2x")
# load_pt_weight_to_tf("weights_pro/pro-denoise3x-up2x.pth", cunet, "weight_map.json")
# cunet.save("/Volumes/Home/oysterqaq/Desktop/cugan_pro-denoise3x-up2x")
#
# model = keras.models.load_model('/Volumes/Home/oysterqaq/Desktop/cugan', compile=False)