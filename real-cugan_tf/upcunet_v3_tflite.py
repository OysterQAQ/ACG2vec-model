import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose


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
        self.conv2 = UNetConv(64, 128, 64, se=True, name="unet.conv2")
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


class UpCunet2x(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels=3, alpha=0.7, pro=True, half=False, crop_size=128, **kwargs):
        """
        初始化 UpCunet2x
        Args:
            tile_mode: 分块模式
                0: 不tile
                1: 固定128x128分块
                2: 根据图像大小自适应分块
        """
        super().__init__(**kwargs)
        self.unet1 = UNet1(in_channels, out_channels, deconv=True, name=self.name + ".unet1")
        self.unet2 = UNet2(in_channels, out_channels, deconv=False, name=self.name + ".unet2")
        self.alpha = alpha
        self.pro = pro
        self.half = half
        self.crop_size = crop_size
        self.stage_model_signatures = {}  # 存储每个stage模型的输入输出签名信息

    def _record_stage_signature(self, stage_name, inputs_dict, outputs_dict):
        """
        记录stage模型的输入输出签名信息

        Args:
            stage_name: stage名称
            inputs_dict: 输入字典，key为输入名称，value为(tensor, shape)
            outputs_dict: 输出字典，key为输出名称，value为(tensor, shape)
        """
        input_signatures = {}
        output_signatures = {}

        # 记录输入签名
        for input_name, (input_tensor, input_shape) in inputs_dict.items():
            # 将TensorShape转换为列表
            shape_list = list(input_shape)
            input_signatures[input_name] = {
                'shape': shape_list  # 只保存形状，不保存dtype
            }

        # 记录输出签名
        for output_name, (output_tensor, output_shape) in outputs_dict.items():
            # 将TensorShape转换为列表
            shape_list = list(output_shape)
            output_signatures[output_name] = {
                'shape': shape_list  # 只保存形状，不保存dtype
            }

        self.stage_model_signatures[stage_name] = {
            'inputs': input_signatures,
            'outputs': output_signatures
        }

    def save_stage_signatures(self, filepath):
        """保存stage模型签名到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.stage_model_signatures, f, indent=2, ensure_ascii=False)

        print(f"Stage模型签名已保存到: {filepath}")

    def load_stage_signatures(self, filepath):
        """从JSON文件加载stage模型签名"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.stage_model_signatures = json.load(f)
        print(f"从 {filepath} 加载了Stage模型签名")

    def _create_stage_models(self):
        """根据网络结构创建每个stage的Keras模型，使用记录的签名形状"""
        print("\n创建stage模型（使用记录的签名形状）...")

        # 检查是否有保存的签名
        if not self.stage_model_signatures:
            print("警告: stage_model_signatures为空，需要先运行call方法生成签名信息")
            return {}

        stage_models = {}

        # Stage 1 模型
        if "stage_1" in self.stage_model_signatures:
            print("创建 stage_1 模型...")
            stage1_sig = self.stage_model_signatures["stage_1"]

            # 从签名获取输入形状
            input_shape = stage1_sig['inputs']['input']['shape'][1:]  # 去掉batch维度
            input_tensor = tf.keras.Input(shape=input_shape, name="input")

            # 构建计算图
            tmp0, x_crop = self.unet1.call_a(input_tensor)
            tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True, name="tmp_se_mean")

            stage_1_model = tf.keras.Model(
                inputs=input_tensor,
                outputs={"tmp0": tmp0, "x_crop": x_crop, "tmp_se_mean": tmp_se_mean},
                name="stage_1"
            )
            stage_1_model.summary()
            stage_models["stage_1"] = stage_1_model
        else:
            print("警告: stage_1签名未找到")

        # Stage 2 模型
        if "stage_2" in self.stage_model_signatures:
            print("创建 stage_2 模型...")
            stage2_sig = self.stage_model_signatures["stage_2"]

            # 从签名获取输入形状
            tmp0_shape = stage2_sig['inputs']['tmp0']['shape'][1:]  # 去掉batch维度
            x_crop_shape = stage2_sig['inputs']['x_crop']['shape'][1:]
            se_mean0_shape = stage2_sig['inputs']['se_mean0']['shape'][1:]

            # 创建输入层
            tmp0_input = tf.keras.Input(shape=tmp0_shape, name="tmp0")
            x_crop_input = tf.keras.Input(shape=x_crop_shape, name="x_crop")
            se_mean0_input = tf.keras.Input(shape=se_mean0_shape, name="se_mean0")

            # 构建计算图
            x_crop = self.unet1.conv2.seblock.mean_call(x_crop_input, se_mean0_input)
            opt_unet1 = self.unet1.call_b(tmp0_input, x_crop)
            tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
            opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
            tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True, name="tmp_se_mean")

            stage_2_model = tf.keras.Model(
                inputs={"tmp0": tmp0_input, "x_crop": x_crop_input, "se_mean0": se_mean0_input},
                outputs={"opt_unet1": opt_unet1, "tmp_x1": tmp_x1, "tmp_x2": tmp_x2, "tmp_se_mean": tmp_se_mean},
                name="stage_2"
            )
            stage_2_model.summary()
            stage_models["stage_2"] = stage_2_model
        else:
            print("警告: stage_2签名未找到")

        # Stage 3 模型
        if "stage_3" in self.stage_model_signatures:
            print("创建 stage_3 模型...")
            stage3_sig = self.stage_model_signatures["stage_3"]

            # 从签名获取输入形状
            tmp_x2_shape = stage3_sig['inputs']['tmp_x2']['shape'][1:]  # 去掉batch维度
            se_mean1_shape = stage3_sig['inputs']['se_mean1']['shape'][1:]

            # 创建输入层
            tmp_x2_input = tf.keras.Input(shape=tmp_x2_shape, name="tmp_x2")
            se_mean1_input = tf.keras.Input(shape=se_mean1_shape, name="se_mean1")

            # 构建计算图
            tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2_input, se_mean1_input)
            tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
            tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True, name="tmp_se_mean")

            stage_3_model = tf.keras.Model(
                inputs={"tmp_x2": tmp_x2_input, "se_mean1": se_mean1_input},
                outputs={"tmp_x2": tmp_x2, "tmp_x3": tmp_x3, "tmp_se_mean": tmp_se_mean},
                name="stage_3"
            )
            stage_3_model.summary()
            stage_models["stage_3"] = stage_3_model
        else:
            print("警告: stage_3签名未找到")

        # Stage 4 模型
        if "stage_4" in self.stage_model_signatures:
            print("创建 stage_4 模型...")
            stage4_sig = self.stage_model_signatures["stage_4"]

            # 从签名获取输入形状
            tmp_x2_shape = stage4_sig['inputs']['tmp_x2']['shape'][1:]  # 去掉batch维度
            tmp_x3_shape = stage4_sig['inputs']['tmp_x3']['shape'][1:]
            se_mean0_shape = stage4_sig['inputs']['se_mean0']['shape'][1:]

            # 创建输入层
            tmp_x2_input = tf.keras.Input(shape=tmp_x2_shape, name="tmp_x2")
            tmp_x3_input = tf.keras.Input(shape=tmp_x3_shape, name="tmp_x3")
            se_mean0_input = tf.keras.Input(shape=se_mean0_shape, name="se_mean0")

            # 构建计算图
            tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3_input, se_mean0_input)
            tmp_x4 = self.unet2.call_c(tmp_x2_input, tmp_x3) * self.alpha
            tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True, name="tmp_se_mean")

            stage_4_model = tf.keras.Model(
                inputs={"tmp_x2": tmp_x2_input, "tmp_x3": tmp_x3_input, "se_mean0": se_mean0_input},
                outputs={"tmp_x4": tmp_x4, "tmp_se_mean": tmp_se_mean},
                name="stage_4"
            )
            stage_4_model.summary()
            stage_models["stage_4"] = stage_4_model
        else:
            print("警告: stage_4签名未找到")

        # Stage 5 模型
        if "stage_5" in self.stage_model_signatures:
            print("创建 stage_5 模型...")
            stage5_sig = self.stage_model_signatures["stage_5"]

            # 从签名获取输入形状
            x_shape = stage5_sig['inputs']['x']['shape'][1:]  # 去掉batch维度
            tmp_x1_shape = stage5_sig['inputs']['tmp_x1']['shape'][1:]
            tmp_x4_shape = stage5_sig['inputs']['tmp_x4']['shape'][1:]
            se_mean1_shape = stage5_sig['inputs']['se_mean1']['shape'][1:]

            # 创建输入层
            x_input = tf.keras.Input(shape=x_shape, name="x")
            tmp_x1_input = tf.keras.Input(shape=tmp_x1_shape, name="tmp_x1")
            tmp_x4_input = tf.keras.Input(shape=tmp_x4_shape, name="tmp_x4")
            se_mean1_input = tf.keras.Input(shape=se_mean1_shape, name="se_mean1")

            # 构建计算图
            tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4_input, se_mean1_input)
            x0 = self.unet2.call_d(tmp_x1_input, tmp_x4)
            x_output = tf.math.add(x0, x_input, name="output")

            stage_5_model = tf.keras.Model(
                inputs={"x": x_input, "tmp_x1": tmp_x1_input, "tmp_x4": tmp_x4_input, "se_mean1": se_mean1_input},
                outputs={"output": x_output},
                name="stage_5"
            )
            stage_5_model.summary()
            stage_models["stage_5"] = stage_5_model
        else:
            print("警告: stage_5签名未找到")

        return stage_models
    def _export_tflite_model(self, model, model_name):
        """导出单个模型为TFLite格式"""
        print(f"\n导出 {model_name} 为TFLite格式...")

        # 创建保存目录
        # os.makedirs("tflite/"+model_name, exist_ok=True)

        # 保存Keras模型
        keras_model_path = f"{model_name}/{model_name}_model"
        # model.save(keras_model_path)
        print(f"Keras模型保存到: {keras_model_path}")

        # 转换为TFLite
        # converter = tf.lite.TFLiteConverter.from_saved_model("tflite/"+model_name)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # 配置转换器
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True

        # 转换
        tflite_model = converter.convert()
        tflite_filename = f"tflite/{model_name}.tflite"

        with open(tflite_filename, "wb") as f:
            f.write(tflite_model)

        print(f"TFLite模型保存到: {tflite_filename}")
        return tflite_filename

    def export_tf_lite(self, signatures_save_path="stage_model_signatures.json"):
        """导出所有stage的TFLite模型"""
        print("=" * 60)
        print("开始导出TFLite模型")
        print("=" * 60)

        # 首先检查是否有保存的stage_model_signatures
        if not self.stage_model_signatures:
            print("警告: stage_model_signatures为空，需要先运行call方法生成签名信息")
            print("正在运行前向传播生成签名...")
            # 使用一个测试输入来生成签名
            test_input = tf.random.normal((1, self.crop_size, self.crop_size, 3))
            _ = self.call(test_input)

        # 保存签名信息到JSON文件
        self.save_stage_signatures(signatures_save_path)

        # 创建所有stage模型
        stage_models = self._create_stage_models()

        # 导出所有stage模型为TFLite
        exported_files = []
        for stage_name, model in stage_models.items():
            tflite_file = self._export_tflite_model(model, stage_name)
            exported_files.append(tflite_file)

        # 同时导出完整模型
        # print("\n" + "=" * 60)
        # print("导出完整模型")
        # print("=" * 60)
        #
        # # 创建完整模型的输入
        # input_shape = (None, None, 3)
        # full_input = tf.keras.Input(shape=input_shape, name="input")
        #
        # # 构建完整模型的前向传播
        # full_output = self.call(full_input)
        #
        # # 创建完整Keras模型
        # full_model = tf.keras.Model(
        #     inputs=full_input,
        #     outputs=full_output,
        #     name="UpCunet2x_full"
        # )
        # full_model.summary()

        # 导出完整模型
        # full_tflite = self._export_tflite_model(full_model, "UpCunet2x_full")
        # exported_files.append(full_tflite)

        print("\n" + "=" * 60)
        print("导出完成!")
        print("=" * 60)
        print(f"导出了 {len(exported_files)} 个模型:")
        for file in exported_files:
            print(f"  - {file}")

        return exported_files

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32) / (255 / 0.7) + 0.15
        n, h0, w0, c = x.shape

        if (tf.dtypes.float16 == x.dtype):
            if_half = True
        else:
            if_half = False

        ph = ((h0 - 1) // self.crop_size + 1) * self.crop_size
        pw = ((w0 - 1) // self.crop_size + 1) * self.crop_size
        x = tf.pad(x, [[0, 0], [18, 18 + pw - w0], [18, 18 + ph - h0], [0, 0]], 'REFLECT')
        n, h, w, c = x.shape

        if if_half:
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)

        n_patch = 0
        tmp_dict = {}

        # Stage 1: 记录签名
        stage1_inputs_dict = {}
        stage1_outputs_dict = {}

        for i in range(0, h - 36, self.crop_size):
            tmp_dict[i] = {}
            for j in range(0, w - 36, self.crop_size):
                x_crop = x[:, i:i + self.crop_size + 36, j:j + self.crop_size + 36, :]

                # 记录stage_1的输入输出
                if not stage1_inputs_dict:
                    stage1_inputs_dict['input'] = (x_crop, x_crop.shape)

                tmp0, x_crop = self.unet1.call_a(x_crop)
                tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
                tmp_dict[i][j] = (tmp0, x_crop)

                # 记录stage_1的输出
                if not stage1_outputs_dict:
                    stage1_outputs_dict['tmp0'] = (tmp0, tmp0.shape)
                    stage1_outputs_dict['x_crop'] = (x_crop, x_crop.shape)
                    stage1_outputs_dict['tmp_se_mean'] = (tmp_se_mean, tmp_se_mean.shape)

        se_mean0 /= n_patch

        # 保存stage_1签名
        self._record_stage_signature("stage_1", stage1_inputs_dict, stage1_outputs_dict)

        if if_half:
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)

        # Stage 2: 记录签名
        stage2_inputs_dict = {}
        stage2_outputs_dict = {}

        for i in range(0, h - 36, self.crop_size):
            for j in range(0, w - 36, self.crop_size):
                tmp0, x_crop = tmp_dict[i][j]

                # 记录stage_2的输入
                if not stage2_inputs_dict:
                    stage2_inputs_dict['tmp0'] = (tmp0, tmp0.shape)
                    stage2_inputs_dict['x_crop'] = (x_crop, x_crop.shape)
                    stage2_inputs_dict['se_mean0'] = (se_mean0, se_mean0.shape)

                x_crop = self.unet1.conv2.seblock.mean_call(x_crop, se_mean0)
                opt_unet1 = self.unet1.call_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
                opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
                tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2)

                # 记录stage_2的输出
                if not stage2_outputs_dict:
                    stage2_outputs_dict['opt_unet1'] = (opt_unet1, opt_unet1.shape)
                    stage2_outputs_dict['tmp_x1'] = (tmp_x1, tmp_x1.shape)
                    stage2_outputs_dict['tmp_x2'] = (tmp_x2, tmp_x2.shape)
                    stage2_outputs_dict['tmp_se_mean'] = (tmp_se_mean, tmp_se_mean.shape)

        se_mean1 /= n_patch

        # 保存stage_2签名
        self._record_stage_signature("stage_2", stage2_inputs_dict, stage2_outputs_dict)

        if if_half:
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)

        # Stage 3: 记录签名
        stage3_inputs_dict = {}
        stage3_outputs_dict = {}

        for i in range(0, h - 36, self.crop_size):
            for j in range(0, w - 36, self.crop_size):
                opt_unet1, tmp_x1, tmp_x2 = tmp_dict[i][j]

                # 记录stage_3的输入
                if not stage3_inputs_dict:
                    stage3_inputs_dict['tmp_x2'] = (tmp_x2, tmp_x2.shape)
                    stage3_inputs_dict['se_mean1'] = (se_mean1, se_mean1.shape)

                tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
                tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True)
                se_mean0 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2, tmp_x3)

                # 记录stage_3的输出
                if not stage3_outputs_dict:
                    stage3_outputs_dict['tmp_x2'] = (tmp_x2, tmp_x2.shape)
                    stage3_outputs_dict['tmp_x3'] = (tmp_x3, tmp_x3.shape)
                    stage3_outputs_dict['tmp_se_mean'] = (tmp_se_mean, tmp_se_mean.shape)

        se_mean0 /= n_patch

        # 保存stage_3签名
        self._record_stage_signature("stage_3", stage3_inputs_dict, stage3_outputs_dict)

        if if_half:
            se_mean1 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)

        # Stage 4: 记录签名
        stage4_inputs_dict = {}
        stage4_outputs_dict = {}

        for i in range(0, h - 36, self.crop_size):
            for j in range(0, w - 36, self.crop_size):
                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tmp_dict[i][j]

                # 记录stage_4的输入
                if not stage4_inputs_dict:
                    stage4_inputs_dict['tmp_x2'] = (tmp_x2, tmp_x2.shape)
                    stage4_inputs_dict['tmp_x3'] = (tmp_x3, tmp_x3.shape)
                    stage4_inputs_dict['se_mean0'] = (se_mean0, se_mean0.shape)

                tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3, se_mean0)
                tmp_x4 = self.unet2.call_c(tmp_x2, tmp_x3) * self.alpha
                tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x4)

                # 记录stage_4的输出
                if not stage4_outputs_dict:
                    stage4_outputs_dict['tmp_x4'] = (tmp_x4, tmp_x4.shape)
                    stage4_outputs_dict['tmp_se_mean'] = (tmp_se_mean, tmp_se_mean.shape)

        se_mean1 /= n_patch

        # 保存stage_4签名
        self._record_stage_signature("stage_4", stage4_inputs_dict, stage4_outputs_dict)

        res = []

        # Stage 5: 记录签名
        stage5_inputs_dict = {}
        stage5_outputs_dict = {}

        for i in range(0, h - 36, self.crop_size):
            temp = []
            for j in range(0, w - 36, self.crop_size):
                x, tmp_x1, tmp_x4 = tmp_dict[i][j]

                # 记录stage_5的输入
                if not stage5_inputs_dict:
                    stage5_inputs_dict['x'] = (x, x.shape)
                    stage5_inputs_dict['tmp_x1'] = (tmp_x1, tmp_x1.shape)
                    stage5_inputs_dict['tmp_x4'] = (tmp_x4, tmp_x4.shape)
                    stage5_inputs_dict['se_mean1'] = (se_mean1, se_mean1.shape)

                tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1)
                x0 = self.unet2.call_d(tmp_x1, tmp_x4)
                x = tf.math.add(x0, x)

                if self.pro:
                    temp.append(tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)),
                                                         clip_value_min=0, clip_value_max=255), dtype=tf.dtypes.uint8))
                else:
                    temp.append(tf.cast(tf.clip_by_value(tf.math.round(x * 255),
                                                         clip_value_min=0, clip_value_max=255), dtype=tf.dtypes.uint8))

                # 记录stage_5的输出
                if not stage5_outputs_dict:
                    stage5_outputs_dict['output'] = (x, x.shape)  # 注意：这里使用'output'作为输出名称

            temp = tf.concat(temp, axis=2)
            res.append(temp)

        res = tf.concat(res, axis=1)
        del tmp_dict

        if w0 != pw or h0 != ph:
            res = res[:, :h0 * 2, :w0 * 2, :]

        # 保存stage_5签名
        self._record_stage_signature("stage_5", stage5_inputs_dict, stage5_outputs_dict)

        return res

    def np2tensor(self, np_frame):
        if self.pro:
            if not self.half:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float32)),
                                      axis=0) / (255 / 0.7) + 0.15
            else:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float16)),
                                      axis=0) / (255 / 0.7) + 0.15
        else:
            if not self.half:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float32)),
                                      axis=0) / 255
            else:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float16)),
                                      axis=0) / 255


# 使用示例
if __name__ == "__main__":
    # 创建模型
    crop_size = 128
    cunet = UpCunet2x(name="UpCunet2x", crop_size=crop_size)


    # 加载权重（如果需要的话）
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


    # 测试
    input_shape = (1, crop_size, crop_size, 3)
    x = tf.random.normal(input_shape)

    # 运行一次call以生成stage_model_signatures
    print("运行前向传播以生成stage模型签名...")
    y = cunet(x)

    # 打印生成的签名信息
    print(f"\n生成的stage_model_signatures: {list(cunet.stage_model_signatures.keys())}")
    for stage_name, signatures in cunet.stage_model_signatures.items():
        print(f"\n{stage_name}:")
        print(f"  输入:")
        for input_name, input_info in signatures['inputs'].items():
            print(f"    {input_name}: shape={input_info['shape']}")
        print(f"  输出:")
        for output_name, output_info in signatures['outputs'].items():
            print(f"    {output_name}: shape={output_info['shape']}")

    # 保存签名到JSON文件
    cunet.save_stage_signatures("tflite/stage_model_signatures.json")

    # 导出所有TFLite模型
    exported_files = cunet.export_tf_lite()

    print("\n完成!")