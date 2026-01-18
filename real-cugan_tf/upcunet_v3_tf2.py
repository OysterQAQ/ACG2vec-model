import os
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
        self.crop_size = crop_size  # 添加 tile_mode 参数

    def export_tf_lite(self):
        crop_size=512
        def export_stage(model, stage_name):
            model.save(stage_name)
            """将模型转换为支持动态形状的TFLite格式"""
            # 转换为 TFLite
            tf.saved_model.save(model, f"{stage_name}_saved")
            converter = tf.lite.TFLiteConverter.from_saved_model(
                f"{stage_name}_saved",
                signature_keys=['serving_default']
            )
            # converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 启用优化
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # 支持动态形状
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用 TFLite 内置操作
                tf.lite.OpsSet.SELECT_TF_OPS,  # 启用部分 TensorFlow 操作
            ]
            converter.allow_custom_ops = True


            # 允许动态形状
            converter.experimental_new_converter = True
            converter.target_spec.supported_types = [tf.float32]

            # 转换并保存
            tflite_model = converter.convert()
            filename = f"{stage_name}.tflite"
            with open(filename, "wb") as f:
                f.write(tflite_model)
            print(f"{stage_name} TFLite model saved")
            return tflite_model
        # 分阶段导出图
        # stage_1
        def stage_1():
            input = tf.keras.Input(shape=(164, 164, 3), name="input")
            tmp0, x_crop = self.unet1.call_a(input)
            tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
            tmp0 = tf.identity(tmp0, name='tmp0')
            x_crop = tf.identity(x_crop, name='x_crop')
            tmp_se_mean = tf.identity(tmp_se_mean, name='tmp_se_mean')
            # 打印输出尺寸
            print(f"stage_1 tmp0 shape: {tmp0.shape}, x_crop shape: {x_crop.shape}, tmp_se_mean shape: {tmp_se_mean.shape}")
            stage_1 = tf.keras.Model(inputs={"input":input}, outputs={"tmp0":tmp0, "x_crop":x_crop, "tmp_se_mean":tmp_se_mean}, name="stage_1")
            # stage_1 保存
            stage_1.save("stage_1")
            export_stage(stage_1,"stage_1")
        # stage_2
        def stage_2():
            se_mean0_input = tf.keras.Input(shape=(1, 1, 64), name="se_mean0")
            tmp0_input = tf.keras.Input(shape=(152, 152, 64), name="tmp0")
            x_crop_input = tf.keras.Input(shape=(76, 76, 64), name="x_crop")

            x_crop = self.unet1.conv2.seblock.mean_call(x_crop_input, se_mean0_input)
            opt_unet1 = self.unet1.call_b(tmp0_input, x_crop)
            tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
            opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
            tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)
            # 打印输出尺寸
            print(f"stage_2 opt_unet1 shape: {opt_unet1.shape}, tmp_x1 shape: {tmp_x1.shape}, tmp_x2 shape: {tmp_x2.shape}, tmp_se_mean shape: {tmp_se_mean.shape}")
            stage_2 = tf.keras.Model(inputs={"tmp0":tmp0_input, "x_crop":x_crop_input, "se_mean0":se_mean0_input}, outputs={"opt_unet1":opt_unet1, "tmp_x1":tmp_x1, "tmp_x2":tmp_x2, "tmp_se_mean":tmp_se_mean}, name="stage_2")
            # stage_2 保存
            stage_2.save("stage_2")
            export_stage(stage_2,"stage_2")
        # stage_3
        def stage_3():
            se_mean1_input = tf.keras.Input(shape=(1, 1, 128), name="se_mean1")
            tmp_x2_input = tf.keras.Input(shape=(142, 142, 128), name="tmp_x2")
            tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2_input, se_mean1_input)
            tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
            tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True)
            # 打印输出尺寸
            print(f"stage_3 tmp_x2 shape: {tmp_x2.shape}, tmp_x3 shape: {tmp_x3.shape}, tmp_se_mean shape: {tmp_se_mean.shape}")
            stage_3 = tf.keras.Model(inputs={"tmp_x2":tmp_x2_input, "se_mean1":se_mean1_input}, outputs={"tmp_x2":tmp_x2, "tmp_x3":tmp_x3, "tmp_se_mean":tmp_se_mean}, name="stage_3")
            # stage_3 保存
            stage_3.save("stage_3")
            export_stage(stage_3,"stage_3")
        # stage_4
        def stage_4():
            tmp_x2_input = tf.keras.Input(shape=(134, 134, 128), name="tmp_x2")
            tmp_x3_input = tf.keras.Input(shape=(67, 67, 128), name="tmp_x3")
            se_mean0_input = tf.keras.Input(shape=(1, 1, 128), name="se_mean0")
            tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3_input, se_mean0_input)
            tmp_x4 = self.unet2.call_c(tmp_x2_input, tmp_x3) * self.alpha
            tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)
            # 打印输出尺寸
            print(f"stage_4 tmp_x4 shape: {tmp_x4.shape}, tmp_se_mean shape: {tmp_se_mean.shape}")
            stage_4 = tf.keras.Model(inputs={"tmp_x2":tmp_x2_input, "tmp_x3":tmp_x3_input, "se_mean0":se_mean0_input}, outputs={"tmp_x4":tmp_x4, "tmp_se_mean":tmp_se_mean}, name="stage_4")
            # stage_4 保存
            stage_4.save("stage_4")
            export_stage(stage_4,"stage_4")
        # stage_5
        def stage_5():
            x_input = tf.keras.Input(shape=(256, 256, 3), name="x")
            tmp_x1_input = tf.keras.Input(shape=(260, 260, 64), name="tmp_x1")
            tmp_x4_input = tf.keras.Input(shape=(130, 130, 64), name="tmp_x4")
            se_mean1_input = tf.keras.Input(shape=(1, 1, 64), name="se_mean1")
            tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4_input, se_mean1_input)
            x0 = self.unet2.call_d(tmp_x1_input, tmp_x4)
            x = tf.math.add(x0, x_input)
            # 打印输出尺寸
            print(f"stage_5 x shape: {x.shape}")
            stage_5 = tf.keras.Model(inputs={"x":x_input, "tmp_x1":tmp_x1_input, "tmp_x4":tmp_x4_input, "se_mean1":se_mean1_input}, outputs={"x":x}, name="stage_5")
            # stage_5 保存
            stage_5.save("stage_5")
            export_stage(stage_5,"stage_5")
        stage_1()
        stage_2()
        stage_3()
        stage_4()
        stage_5()

    def export_tf_lite3(self):
        def export_stage(model, stage_name):
            model.save(stage_name)
            """将模型转换为支持动态形状的TFLite格式"""
            # 转换为 TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 启用优化
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # 支持动态形状
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用 TFLite 内置操作
                tf.lite.OpsSet.SELECT_TF_OPS,  # 启用部分 TensorFlow 操作
            ]
            converter.allow_custom_ops = True

            # 允许动态形状
            # converter.experimental_new_converter = True
            # converter.target_spec.supported_types = [tf.float32]

            # 转换并保存
            tflite_model = converter.convert()
            filename = f"{stage_name}.tflite"
            with open(filename, "wb") as f:
                f.write(tflite_model)
            print(f"{stage_name} TFLite model saved")
            return tflite_model

        def export_stage_with_quantization(model, stage_name, representative_dataset=None):
            """将模型转换为支持动态形状的TFLite格式，可选的量化支持"""
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 如果提供代表性数据集，则进行完全量化
            if representative_dataset:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]

            converter.experimental_new_converter = True

            # 转换并保存
            tflite_model = converter.convert()
            filename = f"{stage_name}.tflite"
            with open(filename, "wb") as f:
                f.write(tflite_model)
            print(f"{stage_name} TFLite model saved")
            return tflite_model

        # stage_1
        def stage_1():
            input = tf.keras.Input(shape=(None, None, 3), name="input",batch_size=1, )
            tmp0, x_crop = self.unet1.call_a(input)
            tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
            stage_1_model = tf.keras.Model(
                inputs={"input": input},
                outputs={"tmp0": tmp0, "x_crop": x_crop, "tmp_se_mean": tmp_se_mean},
                name="stage_1"
            )
            export_stage(stage_1_model, "stage_1")

        # stage_2
        def stage_2():
            se_mean0_input = tf.keras.Input(shape=(1, 1, 64), name="se_mean0",batch_size=1, )
            tmp0_input = tf.keras.Input(shape=(None, None, 64), name="tmp0",batch_size=1, )
            x_crop_input = tf.keras.Input(shape=(None, None, 64), name="x_crop",batch_size=1, )

            x_crop = self.unet1.conv2.seblock.mean_call(x_crop_input, se_mean0_input)
            opt_unet1 = self.unet1.call_b(tmp0_input, x_crop)
            tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
            opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
            tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)

            stage_2_model = tf.keras.Model(
                inputs={"tmp0": tmp0_input, "x_crop": x_crop_input, "se_mean0": se_mean0_input},
                outputs={"opt_unet1": opt_unet1, "tmp_x1": tmp_x1, "tmp_x2": tmp_x2, "tmp_se_mean": tmp_se_mean},
                name="stage_2"
            )
            export_stage(stage_2_model, "stage_2")

        # stage_3
        def stage_3():
            se_mean1_input = tf.keras.Input(shape=(1, 1, 128), name="se_mean1",batch_size=1, )
            tmp_x2_input = tf.keras.Input(shape=(None, None, 128), name="tmp_x2",batch_size=1, )
            tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2_input, se_mean1_input)
            tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
            tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True)

            stage_3_model = tf.keras.Model(
                inputs={"tmp_x2": tmp_x2_input, "se_mean1": se_mean1_input},
                outputs={"tmp_x2": tmp_x2, "tmp_x3": tmp_x3, "tmp_se_mean": tmp_se_mean},
                name="stage_3"
            )
            export_stage(stage_3_model, "stage_3")

        # stage_4
        def stage_4():
            tmp_x2_input = tf.keras.Input(shape=(None, None, 128), name="tmp_x2",batch_size=1, )
            tmp_x3_input = tf.keras.Input(shape=(None, None, 128), name="tmp_x3",batch_size=1, )
            se_mean0_input = tf.keras.Input(shape=(1, 1, 128), name="se_mean0",batch_size=1, )
            tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3_input, se_mean0_input)
            tmp_x4 = self.unet2.call_c(tmp_x2_input, tmp_x3) * self.alpha
            tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)

            stage_4_model = tf.keras.Model(
                inputs={"tmp_x2": tmp_x2_input, "tmp_x3": tmp_x3_input, "se_mean0": se_mean0_input},
                outputs={"tmp_x4": tmp_x4, "tmp_se_mean": tmp_se_mean},
                name="stage_4"
            )
            export_stage(stage_4_model, "stage_4")

        # stage_5
        def stage_5():
            x_input = tf.keras.Input(shape=(None, None, 3), name="x",batch_size=1, )
            tmp_x1_input = tf.keras.Input(shape=(None, None, 64), name="tmp_x1",batch_size=1, )
            tmp_x4_input = tf.keras.Input(shape=(None, None, 64), name="tmp_x4",batch_size=1, )
            se_mean1_input = tf.keras.Input(shape=(1, 1, 64), name="se_mean1",batch_size=1, )
            tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4_input, se_mean1_input)
            x0 = self.unet2.call_d(tmp_x1_input, tmp_x4)
            x = tf.math.add(x0, x_input)

            stage_5_model = tf.keras.Model(
                inputs={"x": x_input, "tmp_x1": tmp_x1_input, "tmp_x4": tmp_x4_input, "se_mean1": se_mean1_input},
                outputs={"x": x},
                name="stage_5"
            )
            export_stage(stage_5_model, "stage_5")

        # 批量导出所有阶段
        def export_all_stages():
            """批量导出所有阶段的TFLite模型"""
            stages = [
                (stage_1, "Stage 1"),
                (stage_2, "Stage 2"),
                (stage_3, "Stage 3"),
                (stage_4, "Stage 4"),
                (stage_5, "Stage 5")
            ]

            for stage_func, stage_name in stages:
                try:
                    print(f"Exporting {stage_name}...")
                    stage_func()
                    print(f"{stage_name} exported successfully")
                except Exception as e:
                    print(f"Error exporting {stage_name}: {e}")
                    raise

        # 执行导出
        export_all_stages()
        print("All stages exported to TFLite format successfully!")


    def export_tf_lite2(self):
        def export_stage(model, stage_name):
            model.save(stage_name)
            """将模型转换为支持动态形状的TFLite格式"""
            # 转换为 TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 启用优化
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # 支持动态形状
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用 TFLite 内置操作
                tf.lite.OpsSet.SELECT_TF_OPS,  # 启用部分 TensorFlow 操作
            ]
            converter.allow_custom_ops = True

            # 允许动态形状
            # converter.experimental_new_converter = True
            # converter.target_spec.supported_types = [tf.float32]

            # 转换并保存
            tflite_model = converter.convert()
            filename = f"{stage_name}.tflite"
            with open(filename, "wb") as f:
                f.write(tflite_model)
            print(f"{stage_name} TFLite model saved")
            return tflite_model

        def export_stage_with_quantization(model, stage_name, representative_dataset=None):
            """将模型转换为支持动态形状的TFLite格式，可选的量化支持"""
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 如果提供代表性数据集，则进行完全量化
            if representative_dataset:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]

            converter.experimental_new_converter = True

            # 转换并保存
            tflite_model = converter.convert()
            filename = f"{stage_name}.tflite"
            with open(filename, "wb") as f:
                f.write(tflite_model)
            print(f"{stage_name} TFLite model saved")
            return tflite_model

        # stage_1
        def stage_1():
            input = tf.keras.Input(shape=(None, None, 3), name="input",batch_size=1, )
            tmp0, x_crop = self.unet1.call_a(input)
            tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
            stage_1_model = tf.keras.Model(
                inputs={"input": input},
                outputs={"tmp0": tmp0, "x_crop": x_crop, "tmp_se_mean": tmp_se_mean},
                name="stage_1"
            )
            export_stage(stage_1_model, "stage_1")

        # stage_2
        def stage_2():
            se_mean0_input = tf.keras.Input(shape=(1, 1, 64), name="se_mean0",batch_size=1, )
            tmp0_input = tf.keras.Input(shape=(None, None, 64), name="tmp0",batch_size=1, )
            x_crop_input = tf.keras.Input(shape=(None, None, 64), name="x_crop",batch_size=1, )

            x_crop = self.unet1.conv2.seblock.mean_call(x_crop_input, se_mean0_input)
            opt_unet1 = self.unet1.call_b(tmp0_input, x_crop)
            tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
            opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
            tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)

            stage_2_model = tf.keras.Model(
                inputs={"tmp0": tmp0_input, "x_crop": x_crop_input, "se_mean0": se_mean0_input},
                outputs={"opt_unet1": opt_unet1, "tmp_x1": tmp_x1, "tmp_x2": tmp_x2, "tmp_se_mean": tmp_se_mean},
                name="stage_2"
            )
            export_stage(stage_2_model, "stage_2")

        # stage_3
        def stage_3():
            se_mean1_input = tf.keras.Input(shape=(1, 1, 128), name="se_mean1",batch_size=1, )
            tmp_x2_input = tf.keras.Input(shape=(None, None, 128), name="tmp_x2",batch_size=1, )
            tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2_input, se_mean1_input)
            tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
            tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True)

            stage_3_model = tf.keras.Model(
                inputs={"tmp_x2": tmp_x2_input, "se_mean1": se_mean1_input},
                outputs={"tmp_x2": tmp_x2, "tmp_x3": tmp_x3, "tmp_se_mean": tmp_se_mean},
                name="stage_3"
            )
            export_stage(stage_3_model, "stage_3")

        # stage_4
        def stage_4():
            tmp_x2_input = tf.keras.Input(shape=(None, None, 128), name="tmp_x2",batch_size=1, )
            tmp_x3_input = tf.keras.Input(shape=(None, None, 128), name="tmp_x3",batch_size=1, )
            se_mean0_input = tf.keras.Input(shape=(1, 1, 128), name="se_mean0",batch_size=1, )
            tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3_input, se_mean0_input)
            tmp_x4 = self.unet2.call_c(tmp_x2_input, tmp_x3) * self.alpha
            tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)

            stage_4_model = tf.keras.Model(
                inputs={"tmp_x2": tmp_x2_input, "tmp_x3": tmp_x3_input, "se_mean0": se_mean0_input},
                outputs={"tmp_x4": tmp_x4, "tmp_se_mean": tmp_se_mean},
                name="stage_4"
            )
            export_stage(stage_4_model, "stage_4")

        # stage_5
        def stage_5():
            x_input = tf.keras.Input(shape=(None, None, 3), name="x",batch_size=1, )
            tmp_x1_input = tf.keras.Input(shape=(None, None, 64), name="tmp_x1",batch_size=1, )
            tmp_x4_input = tf.keras.Input(shape=(None, None, 64), name="tmp_x4",batch_size=1, )
            se_mean1_input = tf.keras.Input(shape=(1, 1, 64), name="se_mean1",batch_size=1, )
            tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4_input, se_mean1_input)
            x0 = self.unet2.call_d(tmp_x1_input, tmp_x4)
            x = tf.math.add(x0, x_input)

            stage_5_model = tf.keras.Model(
                inputs={"x": x_input, "tmp_x1": tmp_x1_input, "tmp_x4": tmp_x4_input, "se_mean1": se_mean1_input},
                outputs={"x": x},
                name="stage_5"
            )
            export_stage(stage_5_model, "stage_5")

        # 批量导出所有阶段
        def export_all_stages():
            """批量导出所有阶段的TFLite模型"""
            stages = [
                (stage_1, "Stage 1"),
                (stage_2, "Stage 2"),
                (stage_3, "Stage 3"),
                (stage_4, "Stage 4"),
                (stage_5, "Stage 5")
            ]

            for stage_func, stage_name in stages:
                try:
                    print(f"Exporting {stage_name}...")
                    stage_func()
                    print(f"{stage_name} exported successfully")
                except Exception as e:
                    print(f"Error exporting {stage_name}: {e}")
                    raise

        # 执行导出
        export_all_stages()
        print("All stages exported to TFLite format successfully!")


    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32) / (255 / 0.7) + 0.15

        n, h0, w0, c = x.shape


        if (tf.dtypes.float16 == x.dtype):
            if_half = True
        else:
            if_half = False





        ph = ((h0 - 1) // self.crop_size  + 1) * self.crop_size
        pw = ((w0 - 1) // self.crop_size  + 1) * self.crop_size
        x = tf.pad(x, [[0, 0], [18, 18 + pw - w0], [18, 18 + ph - h0], [0, 0]], 'REFLECT')
        n, h, w, c = x.shape

        if (if_half):
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)

        n_patch = 0
        tmp_dict = {}

        for i in range(0, h - 36, self.crop_size ):
            tmp_dict[i] = {}
            for j in range(0, w - 36, self.crop_size ):
                x_crop = x[:, i:i + self.crop_size  + 36, j:j + self.crop_size  + 36, :]
                tf.print("1")
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(x_crop))
                tmp0, x_crop = self.unet1.call_a(x_crop)
                tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
                tmp_dict[i][j] = (tmp0, x_crop)

        se_mean0 /= n_patch

        if (if_half):
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)

        for i in range(0, h - 36, self.crop_size ):
            for j in range(0, w - 36, self.crop_size ):
                tmp0, x_crop = tmp_dict[i][j]
                tf.print("2")
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(se_mean0))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp0))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(x_crop))
                x_crop = self.unet1.conv2.seblock.mean_call(x_crop, se_mean0)
                opt_unet1 = self.unet1.call_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
                opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
                tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2)

        se_mean1 /= n_patch

        if (if_half):
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)

        for i in range(0, h - 36, self.crop_size ):
            for j in range(0, w - 36, self.crop_size ):
                opt_unet1, tmp_x1, tmp_x2 = tmp_dict[i][j]
                # 打印一次 块形状
                tf.print("3")
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(se_mean1))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp_x1))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp_x2))


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

        for i in range(0, h - 36, self.crop_size ):
            for j in range(0, w - 36, self.crop_size ):

                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tmp_dict[i][j]
                # 打印一次 块形状
                tf.print("4")
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp_x2))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp_x3))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(se_mean0))

                tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3, se_mean0)
                tmp_x4 = self.unet2.call_c(tmp_x2, tmp_x3) * self.alpha
                tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x4)

        se_mean1 /= n_patch

        res = []
        for i in range(0, h - 36, self.crop_size ):
            temp = []
            for j in range(0, w - 36, self.crop_size ):
                x, tmp_x1, tmp_x4 = tmp_dict[i][j]
                # 打印一次 块形状
                tf.print("5")
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(x))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp_x1))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(tmp_x4))
                tf.print("Processing block at (", i, ",", j, ") with shape:", tf.shape(se_mean1))

                tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1)
                x0 = self.unet2.call_d(tmp_x1, tmp_x4)
                x = tf.math.add(x0, x)

                if (self.pro):
                    temp.append(tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)),
                                                         clip_value_min=0, clip_value_max=255), dtype=tf.dtypes.uint8))
                else:
                    temp.append(tf.cast(tf.clip_by_value(tf.math.round(x * 255),
                                                         clip_value_min=0, clip_value_max=255), dtype=tf.dtypes.uint8))
            temp = tf.concat(temp, axis=2)
            res.append(temp)

        res = tf.concat(res, axis=1)
        del tmp_dict

        if (w0 != pw or h0 != ph):
            res = res[:, :h0 * 2, :w0 * 2, :]
        return res

    def np2tensor(self, np_frame):
        if (self.pro):
            if (self.half == False):
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float32)),
                                      axis=0) / (255 / 0.7) + 0.15
            else:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float16)),
                                      axis=0) / (255 / 0.7) + 0.15
        else:
            if (self.half == False):
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float32)),
                                      axis=0) / 255
            else:
                return tf.expand_dims(tf.convert_to_tensor(np.transpose(np_frame, (2, 0, 1), dtype=tf.float16)),
                                      axis=0) / 255
def load_pt_weight_to_tf(weight_path, tf_model, map_json_file_path):
    import torch
    import json

    with open(map_json_file_path) as json_file:
        map_json = json.load(json_file)

    pt_weights = torch.load(weight_path, map_location="cpu")
    tf_new_weights = []
    #print(tf_model.weights)
    for tf_weight,pt_weight in zip(tf_model.weights,pt_weights.values()):
        #print(str(tf_weight.name) + ":" + str(tf_weight.shape))

        #pt_weight = pt_weights[map_json[tf_model.name]]
        if "kernel" in tf_weight.name:
            pt_weight = pt_weight.numpy()
            pt_weight = tf.transpose(pt_weight, perm=[3, 2, 1, 0])
        tf_new_weights.append(pt_weight.numpy())

    tf_model.set_weights(tf_new_weights)

import cv2
from PIL import Image
def load_image(image_path):
    """加载图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 使用OpenCV加载图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图片: {image_path}")

    # OpenCV默认是BGR，转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image_array, save_path):
    """保存图片"""
    if len(image_array.shape) == 4:  # 如果是批量数据，取第一个
        image_array = image_array[0]

    # 确保数据类型正确
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # 保存图片
    img = Image.fromarray(image_array)
    img.save(save_path)
    print(f"图片已保存到: {save_path}")
# 使用示例
if __name__ == "__main__":
    # 创建模型，设置 tile_mode=1 使用固定128x128分块
    cunet = UpCunet2x(name="UpCunet2x", crop_size=128)  # tile_mode=1 表示固定128x128分块

    # 加载权重

    # 测试
    input_shape = (1, 128, 128, 3)
    x = tf.random.normal(input_shape)
    y = cunet(x)
    load_pt_weight_to_tf("weights_pro/pro-denoise3x-up2x.pth", cunet, "weight_map.json")
    y = cunet(x)

    print(f"Output shape: {y.shape}")
    original_img = load_image("/Volumes/Home/oysterqaq/Desktop/D9589AE7E41A10C5C989A37A74511DFC.png")
    input_tensor = tf.expand_dims(tf.convert_to_tensor(original_img, dtype=tf.float32), axis=0)
    output_tensor = cunet(input_tensor)
    output_img = output_tensor.numpy()
    save_image(output_img, "output_path.png")
    cunet.export_tf_lite()




