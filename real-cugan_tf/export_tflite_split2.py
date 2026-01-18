import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Input
from tensorflow.keras.models import Model
import json


# tf.compat.v1.disable_eager_execution()


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
    def __init__(self, in_channels, out_channels, deconv, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = UNetConv(in_channels, 32, 64, se=False, name=self.name + ".conv1")
        self.conv1_down = Conv2D(filters=64, kernel_size=2, strides=2, padding="valid", input_shape=(None, None, 64),
                                 name=self.name + ".conv1_down")
        self.conv2 = UNetConv(64, 128, 64, se=True, name=self.name + ".conv2")
        self.conv2_up = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="valid",
                                        input_shape=(None, None, 64), name=self.name + ".conv2_up")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", input_shape=(None, None, 64),
                            name=self.name + ".conv3")
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


# ====================== 第一阶段：UNet1调用A部分 ======================
class Stage1_UNet1_A(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.unet1 = UNet1(in_channels, out_channels, deconv=True, name="unet1")

    def call(self, inputs):
        # 输入: [batch, 164, 164, 3]  (128+36)
        x1, x2 = self.unet1.call_a(inputs)
        return x1, x2


# ====================== 第二阶段：UNet1调用B部分 ======================
class Stage2_UNet1_B(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.unet1 = UNet1(in_channels, out_channels, deconv=True, name="unet1")

    def call(self, x1, x2, se_mean0):
        # 输入: x1: [batch, 128, 128, 64], x2: [batch, 64, 64, 64], se_mean0: [batch, 1, 1, 64]
        if se_mean0 is not None:
            x2 = self.unet1.conv2.seblock.mean_call(x2, se_mean0)
        opt_unet1 = self.unet1.call_b(x1, x2)
        return opt_unet1


# ====================== 第三阶段：UNet2调用A部分 ======================
class Stage3_UNet2_A(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False, name="unet2")

    def call(self, inputs):
        # 输入: [batch, 128, 128, 3]  (来自stage2的输出)
        tmp_x1, tmp_x2 = self.unet2.call_a(inputs)
        return tmp_x1, tmp_x2


# ====================== 第四阶段：UNet2调用B和C部分 ======================
class Stage4_UNet2_BC(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False, name="unet2")
        self.alpha = alpha

    def call(self, x2, se_mean1, se_mean0):
        # 输入: x2: [batch, 64, 64, 128], se_mean1: [batch, 1, 1, 128], se_mean0: [batch, 1, 1, 128]
        if se_mean1 is not None:
            tmp_x2 = self.unet2.conv2.seblock.mean_call(x2, se_mean1)
        else:
            tmp_x2 = x2

        tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)

        if se_mean0 is not None:
            tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3, se_mean0)

        tmp_x4 = self.unet2.call_c(tmp_x2, tmp_x3) * self.alpha
        return tmp_x4


# ====================== 第五阶段：UNet2调用D部分 ======================
class Stage5_UNet2_D(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, pro=True, **kwargs):
        super().__init__(**kwargs)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False, name="unet2")
        self.pro = pro

    def call(self, tmp_x1, tmp_x4, se_mean1, opt_unet1):
        # 输入: tmp_x1: [batch, 64, 64, 64], tmp_x4: [batch, 64, 64, 64],
        #       se_mean1: [batch, 1, 1, 64], opt_unet1: [batch, 88, 88, 3]
        if se_mean1 is not None:
            tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1)

        x0 = self.unet2.call_d(tmp_x1, tmp_x4)
        x = tf.math.add(x0, opt_unet1)

        if self.pro:
            return tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)),
                                            clip_value_min=0, clip_value_max=255), dtype=tf.uint8)
        else:
            return tf.cast(tf.clip_by_value(tf.math.round(x * 255),
                                            clip_value_min=0, clip_value_max=255), dtype=tf.uint8)


# ====================== 辅助函数 ======================
def create_models():
    """创建五个阶段的模型"""
    # 阶段1: 处理输入块 (164x164 -> 输出x1, x2)
    input_stage1 = Input(shape=(164, 164, 3), name='input_stage1')
    stage1 = Stage1_UNet1_A(name="stage1")
    x1, x2 = stage1(input_stage1)
    model_stage1 = Model(inputs=input_stage1, outputs=[x1, x2], name="Stage1_UNet1_A")

    # 阶段2: 处理UNet1的B部分
    input_x1 = Input(shape=(128, 128, 64), name='input_x1')
    input_x2 = Input(shape=(64, 64, 64), name='input_x2')
    input_se_mean0 = Input(shape=(1, 1, 64), name='input_se_mean0')
    stage2 = Stage2_UNet1_B(name="stage2")
    opt_unet1 = stage2(input_x1, input_x2, input_se_mean0)
    model_stage2 = Model(inputs=[input_x1, input_x2, input_se_mean0],
                         outputs=opt_unet1, name="Stage2_UNet1_B")

    # 阶段3: UNet2的A部分
    input_stage3 = Input(shape=(128, 128, 3), name='input_stage3')
    stage3 = Stage3_UNet2_A(name="stage3")
    tmp_x1, tmp_x2 = stage3(input_stage3)
    model_stage3 = Model(inputs=input_stage3, outputs=[tmp_x1, tmp_x2], name="Stage3_UNet2_A")

    # 阶段4: UNet2的B和C部分
    input_x2_stage4 = Input(shape=(64, 64, 128), name='input_x2_stage4')
    input_se_mean1 = Input(shape=(1, 1, 128), name='input_se_mean1_stage4')
    input_se_mean0_stage4 = Input(shape=(1, 1, 128), name='input_se_mean0_stage4')
    stage4 = Stage4_UNet2_BC(alpha=0.7, name="stage4")
    tmp_x4 = stage4(input_x2_stage4, input_se_mean1, input_se_mean0_stage4)
    model_stage4 = Model(inputs=[input_x2_stage4, input_se_mean1, input_se_mean0_stage4],
                         outputs=tmp_x4, name="Stage4_UNet2_BC")

    # 阶段5: UNet2的D部分和最终输出
    input_tmp_x1 = Input(shape=(64, 64, 64), name='input_tmp_x1')
    input_tmp_x4 = Input(shape=(64, 64, 64), name='input_tmp_x4')
    input_se_mean1_stage5 = Input(shape=(1, 1, 64), name='input_se_mean1_stage5')
    input_opt_unet1 = Input(shape=(88, 88, 3), name='input_opt_unet1')
    stage5 = Stage5_UNet2_D(pro=True, name="stage5")
    output = stage5(input_tmp_x1, input_tmp_x4, input_se_mean1_stage5, input_opt_unet1)
    model_stage5 = Model(inputs=[input_tmp_x1, input_tmp_x4, input_se_mean1_stage5, input_opt_unet1],
                         outputs=output, name="Stage5_UNet2_D")

    return {
        'stage1': model_stage1,
        'stage2': model_stage2,
        'stage3': model_stage3,
        'stage4': model_stage4,
        'stage5': model_stage5
    }


def export_tflite_models(models_dict, output_dir="./tflite_models"):
    """导出TFLite模型"""
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models_dict.items():
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 尝试量化
        try:
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用TensorFlow Lite内置操作
                tf.lite.OpsSet.SELECT_TF_OPS  # 启用TensorFlow操作
            ]
        except:
            pass

        tflite_model = converter.convert()

        output_path = os.path.join(output_dir, f"{name}.tflite")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Exported {output_path}")


def load_pt_weight_to_tf(weight_path, tf_models, map_json_file_path):
    """加载PyTorch权重到TensorFlow模型"""
    import torch
    import json

    with open(map_json_file_path) as json_file:
        map_json = json.load(json_file)

    pt_weights = torch.load(weight_path, map_location="cpu")

    # 为每个模型加载权重
    for model_name, model in tf_models.items():
        print(f"Loading weights for {model_name}")
        tf_new_weights = []

        for tf_weight in model.weights:
            weight_name = tf_weight.name
            # 移除模型名前缀
            if weight_name.startswith(model_name + "/"):
                weight_name = weight_name[len(model_name) + 1:]

            # 查找对应的PyTorch权重
            if weight_name in map_json:
                pt_key = map_json[weight_name]
                if pt_key in pt_weights:
                    pt_weight = pt_weights[pt_key]
                    if "kernel" in tf_weight.name:
                        # 转换卷积核权重格式: [out_channels, in_channels, height, width] -> [height, width, in_channels, out_channels]
                        pt_weight = pt_weight.numpy()
                        if len(pt_weight.shape) == 4:  # 卷积层
                            pt_weight = np.transpose(pt_weight, (2, 3, 1, 0))
                        elif len(pt_weight.shape) == 2:  # 全连接层
                            pt_weight = np.transpose(pt_weight, (1, 0))
                    else:
                        pt_weight = pt_weight.numpy()

                    tf_new_weights.append(pt_weight)
                else:
                    print(f"Weight not found: {pt_key}")
                    tf_new_weights.append(tf_weight.numpy())
            else:
                print(f"Mapping not found for: {weight_name}")
                tf_new_weights.append(tf_weight.numpy())

        if tf_new_weights:
            model.set_weights(tf_new_weights)
        else:
            print(f"No weights loaded for {model_name}")


# ====================== 测试代码 ======================
def test_stages():
    """测试每个阶段的模型"""
    models = create_models()

    # 测试数据
    batch_size = 1
    test_input_stage1 = tf.random.normal((batch_size, 164, 164, 3))  # 128+36

    # 阶段1测试
    print("Testing Stage 1...")
    x1, x2 = models['stage1'](test_input_stage1)
    print(f"Stage1 outputs: x1 shape={x1.shape}, x2 shape={x2.shape}")

    # 阶段2测试
    print("Testing Stage 2...")
    se_mean0 = tf.zeros((batch_size, 1, 1, 64), dtype=tf.float32)
    opt_unet1 = models['stage2']([x1, x2, se_mean0])
    print(f"Stage2 output shape: {opt_unet1.shape}")

    # 阶段3测试
    print("Testing Stage 3...")
    tmp_x1, tmp_x2 = models['stage3'](opt_unet1)
    print(f"Stage3 outputs: tmp_x1 shape={tmp_x1.shape}, tmp_x2 shape={tmp_x2.shape}")

    # 阶段4测试
    print("Testing Stage 4...")
    se_mean1 = tf.zeros((batch_size, 1, 1, 128), dtype=tf.float32)
    se_mean0_stage4 = tf.zeros((batch_size, 1, 1, 128), dtype=tf.float32)
    tmp_x4 = models['stage4']([tmp_x2, se_mean1, se_mean0_stage4])
    print(f"Stage4 output shape: {tmp_x4.shape}")

    # 阶段5测试
    print("Testing Stage 5...")
    se_mean1_stage5 = tf.zeros((batch_size, 1, 1, 64), dtype=tf.float32)
    output = models['stage5']([tmp_x1, tmp_x4, se_mean1_stage5, opt_unet1])
    print(f"Stage5 output shape: {output.shape}")
    print(f"Stage5 output dtype: {output.dtype}")

    return models


if __name__ == "__main__":
    # 创建并测试所有模型
    models = test_stages()

    # 加载权重
    print("\nLoading weights...")
    load_pt_weight_to_tf("weights_pro/pro-denoise3x-up2x.pth", models, "weight_map.json")

    # 导出TFLite模型
    print("\nExporting TFLite models...")
    export_tflite_models(models, "./tflite_models")

    # 保存Keras模型
    print("\nSaving Keras models...")
    for name, model in models.items():
        model.save(f"./saved_models/{name}", save_format='tf')
        print(f"Saved {name}")