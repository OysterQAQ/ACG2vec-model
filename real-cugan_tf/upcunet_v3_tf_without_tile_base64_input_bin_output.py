import base64

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
        # x3=tf.keras.layers.ZeroPadding2D(padding=3, data_format=None,)(x3)
        z = self.conv_bottom(x3)
        # 由于tf转置卷积层不支持自定义padding 需要手动对输出crop
        z = z[:, 3:-3, 3:-3, :]
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
        z = z[:, 3:-3, 3:-3, :]
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


class UpCunet2x(tf.keras.layers.Layer):
    def __init__(self, in_channels=3, out_channels=3, alpha=1, pro=True, half=False, **kwargs):
        super().__init__(**kwargs)
        self.unet1 = UNet1(in_channels, out_channels, deconv=True, name=self.name + ".unet1")
        self.unet2 = UNet2(in_channels, out_channels, deconv=False, name=self.name + ".unet2")
        self.alpha = alpha
        self.pro = pro
        self.half = half

    #@tf.function
    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32) / (
                255 / 0.7) + 0.15
        input_tensor_shape = tf.shape(x)
        n, h0, w0, c = input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]
        #tile_mode = tf.cond(h0 > w0, lambda:  h0 // 1080, lambda: w0 // 1080)
        # if tf.math.greater(h0,w0):
        #     tile_mode = h0 // 1080
        # else:
        #     tile_mode = w0 // 1080
        if (tf.dtypes.float16 == x.dtype):
            if_half = True
        else:
            if_half = False

        print(n)
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        # 填充h w 维度 （h前填充，h后填充，w前填充，w后填充）
        tile_mode = tf.cond(h0 > w0, lambda: h0 // 1080, lambda: w0 // 1080)
        return  tf.cond(tile_mode ==0, lambda: self.forward_tile_0(x,ph,h0,pw,w0), lambda: self.forward_tile_not_0(x, ph, h0, pw, w0, tile_mode, if_half))
        #return self.forward_tile_0(x,ph,h0,pw,w0)
        # if (tile_mode == tf.constant(0)):
        #     return self.forward_tile_0(x,ph,h0,pw,w0)
        # # elif(tile_mode == tf.constant(1)):
        # #     if tf.math.greater_equal(w0,h0):
        # #         crop_size_w = ((w0 - 1) // 4 * 4 + 4) // 2  # 减半后能被2整除，所以要先被4整除
        # #         crop_size_h = (h0 - 1) // 2 * 2 + 2  # 能被2整除
        # #     else:
        # #         crop_size_h = ((h0 - 1) // 4 * 4 + 4) // 2  # 减半后能被2整除，所以要先被4整除
        # #         crop_size_w = (w0 - 1) // 2 * 2 + 2  # 能被2整除
        # #     crop_size = (crop_size_h, crop_size_w)
        # else:
        #     return self.forward_tile_not_0(x, ph, h0, pw, w0, tile_mode, if_half)

    #@tf.function
    def forward_tile_0(self,x,ph,h0,pw,w0):
        x = tf.pad(x, [[0, 0], [18, 18 + ph - h0], [18, 18 + pw - w0], [0, 0]], 'REFLECT')  # 需要保证被2整除
        x = self.unet1(x)
        x0 = self.unet2(x, self.alpha)
        x = x[:, 20:-20, 20:-20, :]

        x = tf.math.add(x0, x)
        x = self.check(w0, pw, h0, ph, x)
        output = tf.cast(
            tf.clip_by_value(tf.math.round(((x - 0.15) * (255 / 0.7))), clip_value_min=0, clip_value_max=255),
            dtype=tf.dtypes.uint8)
        output_image = tf.io.encode_png(
            tf.squeeze(output)

        )
        return output_image

    #@tf.function
    def forward_tile_not_0(self,x,ph,h0,pw,w0,tile_mode,if_half):
        tile_mode = tf.math.minimum(tf.math.minimum(h0, w0) // 128, tile_mode)  # 最小短边为128*128
        t2 = tile_mode * 2
        crop_size = (((h0 - 1) // t2 * t2 + t2) // tile_mode, ((w0 - 1) // t2 * t2 + t2) // tile_mode)
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x = tf.pad(x, [[0, 0], [18, 18 + ph - h0], [18, 18 + pw - w0], [0, 0]], 'REFLECT')
        x_tensor_shape = tf.shape(x)
        n, h, w, c = x_tensor_shape[0], x_tensor_shape[1], x_tensor_shape[2], x_tensor_shape[3]
        if (if_half):
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)
        n_patch = 0
        # 构建四个tensorArray tmp_array_0 tmp_array_1 tmp_array_2 tmp_array_3
        # tensorArray的索引为i*i_length+j
        tmp_array_0 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True,
                                     infer_shape=False)
        tmp_array_1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True,
                                     infer_shape=False)
        tmp_array_2 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True,
                                     infer_shape=False)

        i_length = (h - 36) // crop_size[0]
        j_length = (w - 36) // crop_size[1]
        for i in tf.range(i_length*j_length):
            j=i%i_length
            x_crop = x[:, i//i_length * crop_size[0]:(i//i_length + 1) * crop_size[0] + 36,
                     j * (crop_size[1]):(j + 1) * crop_size[1] + 36, :]
            x_crop_tensor_shape = tf.shape(x_crop)
            n, h1, w1, c1 = x_crop_tensor_shape[0], x_crop_tensor_shape[1], x_crop_tensor_shape[2], \
                x_crop_tensor_shape[3]
            tmp0, x_crop = self.unet1.call_a(x_crop)
            tmp_se_mean = tf.math.reduce_mean(x_crop, axis=(1, 2), keepdims=True)
            se_mean0 += tmp_se_mean
            n_patch += 1
            tmp_array_0=tmp_array_0.write(i, tmp0)
            tmp_array_1=tmp_array_1.write(i, x_crop)
            #for j in tf.range(j_length):

        se_mean0 /= tf.cast(n_patch, dtype=tf.dtypes.float32)
        if (if_half):
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)
        for i in tf.range(i_length*j_length):
            tmp0 = tmp_array_0.read(i)
            x_crop = tmp_array_1.read(i)
            x_crop = self.unet1.conv2.seblock.mean_call(x_crop, se_mean0)
            opt_unet1 = self.unet1.call_b(tmp0, x_crop)
            tmp_x1, tmp_x2 = self.unet2.call_a(opt_unet1)
            opt_unet1 = opt_unet1[:, 20:-20, 20:-20, :]
            tmp_se_mean = tf.math.reduce_mean(tmp_x2, axis=(1, 2), keepdims=True)
            se_mean1 += tmp_se_mean
            tmp_array_0=tmp_array_0.write(i, opt_unet1)
            tmp_array_1=tmp_array_1.write(i, tmp_x1)
            tmp_array_2=tmp_array_2.write(i, tmp_x2)
        se_mean1 /= tf.cast(n_patch, dtype=tf.dtypes.float32)
        if (if_half):
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float16)
        else:
            se_mean0 = tf.zeros((n, 1, 1, 128), dtype=tf.dtypes.float32)
        tmp_array_3 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True,
                                     infer_shape=False)
        for i in tf.range(i_length*j_length):
            #for j in tf.range():
            opt_unet1= tmp_array_0.read(i)
            tmp_x1= tmp_array_1.read(i)
            tmp_x2= tmp_array_2.read(i)
            tmp_x2 = self.unet2.conv2.seblock.mean_call(tmp_x2, se_mean1)
            tmp_x2, tmp_x3 = self.unet2.call_b(tmp_x2)
            tmp_se_mean = tf.math.reduce_mean(tmp_x3, axis=(1, 2), keepdims=True)
            se_mean0 += tmp_se_mean
            tmp_array_0=tmp_array_0.write(i, opt_unet1)
            tmp_array_1=tmp_array_1.write(i, tmp_x1)
            tmp_array_2=tmp_array_2.write(i, tmp_x2)
            tmp_array_3=tmp_array_3.write(i, tmp_x3)
        se_mean0 /= tf.cast(n_patch, dtype=tf.dtypes.float32)
        if (if_half):
            se_mean1 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float16)
        else:
            se_mean1 = tf.zeros((n, 1, 1, 64), dtype=tf.dtypes.float32)
        for i in tf.range(i_length*j_length):
            #for j in tf.range(j_length):
            opt_unet1=tmp_array_0.read(i)
            tmp_x1=tmp_array_1.read(i)
            tmp_x2=tmp_array_2.read(i )
            tmp_x3=tmp_array_3.read(i )
            tmp_x3 = self.unet2.conv3.seblock.mean_call(tmp_x3, se_mean0)
            tmp_x4 = self.unet2.call_c(tmp_x2, tmp_x3) * self.alpha
            tmp_se_mean = tf.math.reduce_mean(tmp_x4, axis=(1, 2), keepdims=True)
            se_mean1 += tmp_se_mean
            tmp_array_0=tmp_array_0.write(i , opt_unet1)
            tmp_array_1=tmp_array_1.write(i, tmp_x1)
            tmp_array_2=tmp_array_2.write(i, tmp_x4)
        se_mean1 /= tf.cast(n_patch, dtype=tf.dtypes.float32)
        res = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True,
                                     infer_shape=False)
        for i in tf.range(i_length):
            temp = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True,
                                     infer_shape=False)
            for j in tf.range(j_length):
                x = tmp_array_0.read(i * j_length + j)
                tmp_x1 = tmp_array_1.read(i * j_length + j)
                tmp_x4 = tmp_array_2.read(i * j_length + j)
                tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1)
                x0 = self.unet2.call_d(tmp_x1, tmp_x4)
                x = tf.math.add(x0, x)  # x0是unet2的最终输出
                temp=temp.write(j,tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)), clip_value_min=0,
                                                    clip_value_max=255), dtype=tf.dtypes.uint8))

            #收集0-j-1个
            row=temp.gather(tf.range(j_length-1))
            # tf.ensure_shape(
            #     row, shape=(j_length-1,None,None,3), name=None
            # )
            tf.print(j_length-1)
            row = tf.unstack(row,num=j_length-1)
            row = tf.concat(row, axis=2)
            row = tf.concat([row, temp.read(j_length-1)], axis=2)
            temp.close()
            res.write(i,row)
        output=res.gather(tf.range(i_length-1))
        output = tf.unstack(output)
        output = tf.concat(output, axis=1)
        output = tf.concat([output, res.read(i_length - 1)], axis=1)
        res.close()




        # res = None
        # for i in tf.range(i_length):
        #     temp = None
        #     for j in tf.range(j_length):
        #         x = tmp_array_0.read(i * j_length + j)
        #         tmp_x1=tmp_array_1.read(i * j_length + j)
        #         tmp_x4=tmp_array_2.read(i * j_length + j)
        #         tmp_x4 = self.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1)
        #         x0 = self.unet2.call_d(tmp_x1, tmp_x4)
        #         x = tf.math.add(x0, x)  # x0是unet2的最终输出
        #         if (temp is None):
        #             temp = tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)), clip_value_min=0,
        #                                             clip_value_max=255), dtype=tf.dtypes.uint8)
        #         else:
        #             tf.print(temp)
        #             temp = tf.concat(
        #                 [temp, tf.cast(tf.clip_by_value(tf.math.round((x - 0.15) * (255 / 0.7)), clip_value_min=0,
        #                                                 clip_value_max=255), dtype=tf.dtypes.uint8)], axis=2
        #             )
        #     if (res is None):
        #         res = temp
        #     else:
        #         res = tf.concat(
        #             [res, temp], axis=1
        #         )
        tmp_array_0.close()
        tmp_array_1.close()
        tmp_array_2.close()
        tmp_array_3.close()
        output = self.check(w0, pw, h0, ph, output)
        output_image = tf.io.encode_png(
            tf.squeeze(output)
        )
        return output_image



    @tf.function
    def check(self, w0, pw, h0, ph, x):
        if (w0 != pw or h0 != ph):
            return x[:, :h0 * 2, :w0 * 2, :]
        return x

    @tf.function
    def check2(self, w, pw, h, ph, x):
        if (w != pw * 2 or h != ph * 2):
            return x[:, (h - ph * 2) // 2:-((h - ph * 2) // 2), (w - pw * 2) // 2:-((w - pw * 2) // 2), :]
        return x


class Base64DecoderLayer(tf.keras.layers.Layer):
    """
    Convert a incoming base 64 string into an bitmap with rgb values between 0 and 1
    target_size e.g. [width,height]
    """

    def __init__(self, **kwargs):
        # self.target_size = target_size
        super(Base64DecoderLayer, self).__init__(**kwargs)

    def byte_to_img(self, byte_tensor):
        # base64 decoding id done by tensorflow serve, when using b64 json
        byte_tensor = tf.io.decode_base64(byte_tensor)
        imgs_map = tf.io.decode_image(byte_tensor, channels=3)
        imgs_map.set_shape((None, None, 3))
        # img = tf.image.resize(imgs_map, self.target_size)
        # img = tf.cast(imgs_map, dtype=tf.float32)/ (
        # 255 / 0.7) + 0.15
        return imgs_map

    def call(self, input, **kwargs):
        with tf.device("/cpu:0"):
            imgs_map = tf.map_fn(self.byte_to_img, input, dtype=tf.uint8)
        return imgs_map


def load_pt_weight_to_tf(weight_path, tf_model, map_json_file_path):
    import torch
    import json

    with open(map_json_file_path) as json_file:

        map_json = json.load(json_file)

    pt_weights = torch.load(weight_path, map_location="cpu")
    tf_new_weights = []
    for tf_weight,pt_weight in zip(tf_model.weights,pt_weights.values()):
        #print(tf_weight.name)
        #pt_weight = pt_weights[map_json[tf_weight.name]]
        if "kernel" in tf_weight.name:
            pt_weight = pt_weight.numpy()
            pt_weight = tf.transpose(pt_weight, perm=[3, 2, 1, 0])
        tf_new_weights.append(pt_weight.numpy())

    tf_model.set_weights(tf_new_weights)


def build_model(weight_path):
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
    x = Base64DecoderLayer( trainable=False)(inputs)
    cunet = UpCunet2x(name="UpCunet2x",trainable=False)
    # input_shape = (1,1224, 1224, 3)
    # x1 = tf.random.normal(input_shape)
    # y = cunet(x1)
    y = cunet(x)
    load_pt_weight_to_tf(weight_path, cunet, "weight_map.json")
    model = tf.keras.Model(inputs=inputs, outputs=y, trainable=False)
    return model


def export(model, path):
    model.save(path)


model = build_model("weights_pro/pro-no-denoise-up2x.pth")
#model.export("/Volumes/Home/oysterqaq/Desktop/cugan")
#export(model,"tf_weight/pro-no-denoise-up2x_without_tile_b64_in_bin_out")
# model=build_model("weights_pro/pro-denoise3x-up2x.pth")
# export(model,"tf_weight/pro-denoise3x-up2x_without_tile_b64_in_bin_out")
# model=build_model("weights_pro/pro-conservative-up2x.pth")
# export(model,"tf_weight/pro-conservative-up2x_without_tile_b64_in_bin_out")
pic = open("inputs/test3.jpeg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())
file_path = "output/test4.png"
with open(file_path, "wb") as file:

    binary_data = model(tf.stack([tf.convert_to_tensor(pic_base64)])).numpy()
    # Example binary data
    file.write(binary_data)
