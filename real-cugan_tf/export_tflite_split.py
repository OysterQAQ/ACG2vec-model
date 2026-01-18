import os
import json
import torch
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose


# =========================
# 原始模型代码（未改）
# =========================

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction=8, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters=(in_channels // reduction), kernel_size=1, strides=1, padding="valid",
                            use_bias=bias, input_shape=(None, None, in_channels), name=self.name + ".conv1")
        self.conv2 = Conv2D(filters=in_channels, kernel_size=1, strides=1, padding="valid", use_bias=bias,
                            input_shape=(None, None, (in_channels // reduction)), name=self.name + ".conv2")

    def call(self, inputs):
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
    def __init__(self, in_channels, mid_channels, out_channels, se, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.Sequential(
            [
                Conv2D(filters=mid_channels, kernel_size=3, strides=1, padding="valid",
                       input_shape=(None, None, in_channels), name=self.name + ".conv.0"),
                LeakyReLU(alpha=0.1),
                Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="valid",
                       input_shape=(None, None, mid_channels), name=self.name + ".conv.2"),
                LeakyReLU(alpha=0.1),
            ]
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True, name=self.name + ".seblock")
        else:
            self.seblock = None

    def call(self, inputs):
        x = self.conv(inputs)
        if self.seblock is not None:
            x = self.seblock(x)
        return x


class UNet1(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, deconv, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = UNetConv(in_channels, 32, 64, se=False, name=self.name + ".conv1")
        self.conv1_down = Conv2D(filters=64, kernel_size=2, strides=2, padding="valid",
                                 input_shape=(None, None, 64), name=self.name + ".conv1_down")
        self.conv2 = UNetConv(64, 128, 64, se=True, name=self.name + ".conv2")
        self.conv2_up = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 64), name=self.name + ".conv2_up")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid",
                            input_shape=(None, None, 64), name=self.name + ".conv3")
        if deconv:
            self.conv_bottom = Conv2DTranspose(filters=out_channels, kernel_size=4, strides=2, padding="valid",
                                               input_shape=(None, None, 64), name=self.name + ".conv_bottom")
        else:
            self.conv_bottom = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="valid",
                                      input_shape=(None, None, 64), name=self.name + ".conv_bottom")

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        x1 = x1[:, 4:-4, 4:-4, :]
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)
        x3 = self.conv3(x1 + x2)
        x3 = LeakyReLU(alpha=0.1)(x3)
        z = self.conv_bottom(x3)
        z = z[:, 3:-3, 3:-3, :]
        return z

    def call_a(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
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
        z = z[:, 3:-3, 3:-3, :]
        return z


class UNet2(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, deconv, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = UNetConv(in_channels, 32, 64, se=False, name=self.name + ".conv1")
        self.conv1_down = Conv2D(filters=64, kernel_size=2, strides=2, padding="valid",
                                 input_shape=(None, None, 64), name=self.name + ".conv1_down")
        self.conv2 = UNetConv(64, 64, 128, se=True, name=self.name + ".conv2")
        self.conv2_down = Conv2D(filters=128, kernel_size=2, strides=2, padding="valid",
                                 input_shape=(None, None, 128), name=self.name + ".conv2_down")
        self.conv3 = UNetConv(128, 256, 128, se=True, name=self.name + ".conv3")
        self.conv3_up = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 128), name=self.name + ".conv3_up")
        self.conv4 = UNetConv(128, 64, 64, se=True, name=self.name + ".conv4")
        self.conv4_up = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 64), name=self.name + ".conv4_up")
        self.conv5 = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid",
                            input_shape=(None, None, 64), name=self.name + ".conv5")

        if deconv:
            self.conv_bottom = Conv2DTranspose(filters=out_channels, kernel_size=4, strides=2, padding="same",
                                               input_shape=(None, None, 64), name=self.name + ".conv_bottom")
        else:
            self.conv_bottom = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="valid",
                                      input_shape=(None, None, 64), name=self.name + ".conv_bottom")

    def call(self, inputs, alpha=1):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        x1 = x1[:, 16:-16, 16:-16, :]
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
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
        return z

    def call_a(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv1_down(x1)
        x1 = x1[:, 16:-16, 16:-16, :]
        x2 = LeakyReLU(alpha=0.1)(x2)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def call_b(self, x2):
        x3 = self.conv2_down(x2)
        x2 = x2[:, 4:-4, 4:-4, :]
        x3 = LeakyReLU(alpha=0.1)(x3)
        x3 = self.conv3.conv(x3)
        return x2, x3

    def call_c(self, x2, x3):
        x3 = self.conv3_up(x3)
        x3 = LeakyReLU(alpha=0.1)(x3)
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def call_d(self, x1, x4):
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
        raise RuntimeError("Full call not used for TFLite export")


# =========================
# 权重加载函数（原样保留）
# =========================

def load_pt_weight_to_tf(weight_path, tf_model, map_json_file_path):
    import torch
    import json

    with open(map_json_file_path) as json_file:
        map_json = json.load(json_file)

    pt_weights = torch.load(weight_path, map_location="cpu")
    tf_new_weights = []
    for tf_weight, pt_weight in zip(tf_model.weights, pt_weights.values()):
        if "kernel" in tf_weight.name:
            pt_weight = pt_weight.numpy()
            pt_weight = tf.transpose(pt_weight, perm=[3, 2, 1, 0])
        tf_new_weights.append(pt_weight.numpy())
    tf_model.set_weights(tf_new_weights)


# =========================
# Stage 模型（导出专用）
# =========================

# =========================
# Stage 模型（导出专用）
# =========================

class StageA(tf.keras.Model):
    def __init__(self, unet1):
        super().__init__()
        self.unet1 = unet1

    # StageA 没问题，可以保持原样
    def call(self, x):
        x1, x2 = self.unet1.call_a(x)
        return x1, x2


class StageB(tf.keras.Model):
    def __init__(self, unet1, unet2):
        super().__init__()
        self.unet1 = unet1
        self.unet2 = unet2

    # 每个 tensor 单独写参数
    def call(self, x1, x2, se_mean0):
        x2 = self.unet1.conv2.seblock.mean_call(x2, se_mean0)
        out1 = self.unet1.call_b(x1, x2)
        u2_x1, u2_x2 = self.unet2.call_a(out1)
        return out1, u2_x1, u2_x2


class StageC(tf.keras.Model):
    def __init__(self, unet2):
        super().__init__()
        self.unet2 = unet2

    # 每个 tensor 单独写参数
    def call(self, u2_x2, se_mean1):
        u2_x2 = self.unet2.conv2.seblock.mean_call(u2_x2, se_mean1)
        x2, x3 = self.unet2.call_b(u2_x2)
        return x2, x3


class StageD(tf.keras.Model):
    def __init__(self, unet2, alpha):
        super().__init__()
        self.unet2 = unet2
        self.alpha = alpha

    # 每个 tensor 单独写参数
    def call(self, x2, x3, se_mean2):
        x3 = self.unet2.conv3.seblock.mean_call(x3, se_mean2)
        x4 = self.unet2.call_c(x2, x3) * self.alpha
        return x4


class StageE(tf.keras.Model):
    def __init__(self, unet2):
        super().__init__()
        self.unet2 = unet2

    # 每个 tensor 单独写参数
    def call(self, x1, x4, se_mean3):
        x4 = self.unet2.conv4.seblock.mean_call(x4, se_mean3)
        out = self.unet2.call_d(x1, x4)
        return out



# =========================
# TFLite 导出函数
# =========================

def export_tflite(model, name, input_specs):
    concrete = tf.function(model).get_concrete_function(*input_specs)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = []
    tflite = converter.convert()
    with open(f"{name}.tflite", "wb") as f:
        f.write(tflite)
    print(f"[OK] exported {name}.tflite")


# =========================
# 主导出逻辑
# =========================

def main():
    tf.keras.backend.clear_session()

    cunet = UpCunet2x(name="UpCunet2x")

    dummy = tf.random.normal((1, 256, 256, 3))
    _ = cunet.unet1(dummy)
    _ = cunet.unet2(dummy)

    load_pt_weight_to_tf("weights_pro/pro-denoise3x-up2x.pth", cunet, "weight_map.json")

    stageA = StageA(cunet.unet1)
    stageB = StageB(cunet.unet1, cunet.unet2)
    stageC = StageC(cunet.unet2)
    stageD = StageD(cunet.unet2, cunet.alpha)
    stageE = StageE(cunet.unet2)

    export_tflite(stageA, "stageA", [
        tf.TensorSpec([1, None, None, 3], tf.float32),
    ])

    export_tflite(stageB, "stageB", [
        tf.TensorSpec([1, None, None, 64], tf.float32),
        tf.TensorSpec([1, None, None, 64], tf.float32),
        tf.TensorSpec([1, 1, 1, 64], tf.float32),
    ])

    export_tflite(stageC, "stageC", [
        tf.TensorSpec([1, None, None, 128], tf.float32),
        tf.TensorSpec([1, 1, 1, 128], tf.float32),
    ])

    export_tflite(stageD, "stageD", [
        tf.TensorSpec([1, None, None, 128], tf.float32),
        tf.TensorSpec([1, None, None, 128], tf.float32),
        tf.TensorSpec([1, 1, 1, 128], tf.float32),
    ])

    export_tflite(stageE, "stageE", [
        tf.TensorSpec([1, None, None, 64], tf.float32),
        tf.TensorSpec([1, None, None, 64], tf.float32),
        tf.TensorSpec([1, 1, 1, 64], tf.float32),
    ])


if __name__ == "__main__":
    main()
