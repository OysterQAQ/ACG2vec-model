import * as tf from '@tensorflow/tfjs';

class SEBlock extends tf.layers.Layer {
    constructor(inChannels, reduction = 8, bias = false, kwargs) {
        super(kwargs);


        // Define the layers in the constructor
        this.conv1 = tf.layers.conv2d({
            filters: inChannels / reduction,
            kernelSize: 1,
            strides: 1,
            padding: 'valid',
            useBias: bias,
            inputShape: [null, null, inChannels],
            name: `${this.name}.conv1`
        });

        this.conv2 = tf.layers.conv2d({
            filters: inChannels,
            kernelSize: 1,
            strides: 1,
            padding: 'valid',
            useBias: bias,
            inputShape: [null, null, inChannels / reduction],
            name: `${this.name}.conv2`
        });
    }

    // Define the call method for forward pass
    call(inputs) {

        return tf.tidy(() => {

            const x0 = tf.layers.activation({activation: 'sigmoid'}).apply(this.conv2.apply(tf.layers.activation({activation: 'relu'}).apply(tf.mean(inputs, [1, 2], true))));
            return tf.mul(inputs, x0);
        });
        // let x0 = tf.mean(inputs, [1, 2], true);
        // x0 = this.conv1.apply(x0);
        // x0 = tf.layers.activation({ activation: 'relu' }).apply(x0);
        // x0 = this.conv2.apply(x0);
        // x0 = tf.layers.activation({ activation: 'sigmoid' }).apply(x0);
        // const x = tf.mul(inputs, x0);
        // return x;
    }

    // Define the mean_call method for mean computation
    mean_call(x, x0) {


        return tf.tidy(() => {
            const x0_ = tf.layers.activation({activation: 'sigmoid'}).apply(this.conv2.apply(tf.layers.activation({activation: 'relu'}).apply(this.conv1.apply(x0))));

            return tf.mul(x, x0_);
        });
        // x0 = this.conv1.apply(x0);
        // x0 = tf.layers.activation({ activation: 'relu' }).apply(x0);
        // x0 = this.conv2.apply(x0);
        // x0 = tf.layers.activation({ activation: 'sigmoid' }).apply(x0);
        // const result = tf.mul(x, x0);
        // return result;
    }
}

class UNetConv extends tf.layers.Layer {
    constructor(inChannels, midChannels, outChannels, se, kwargs) {
        super(kwargs);

        // Define the layers in the constructor
        this.conv = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    filters: midChannels,
                    kernelSize: 3,
                    strides: 1,
                    padding: 'valid',
                    inputShape: [null, null, inChannels],
                    name: `${this.name}.conv.0`
                }),
                tf.layers.leakyReLU({alpha: 0.1}),
                tf.layers.conv2d({
                    filters: outChannels,
                    kernelSize: 3,
                    strides: 1,
                    padding: 'valid',
                    inputShape: [null, null, midChannels],
                    name: `${this.name}.conv.2`
                }),
                tf.layers.leakyReLU({alpha: 0.1}),
            ]
        });

        if (se) {
            this.seblock = new SEBlock(outChannels, 8, true, {name: `${this.name}.seblock`});
        } else {
            this.seblock = null;
        }
    }

    // Define the call method for forward pass
    call(inputs) {
        return tf.tidy(() => {
            const x = this.conv.apply(inputs);
            if (this.seblock !== null) {
                return this.seblock.call(x);
            }

            return x;
        });

        // let x = this.conv.apply(inputs);
        // if (this.seblock !== null) {
        //   x = this.seblock.call(x);
        // }
        // return x;
    }
}

class UNet1 extends tf.layers.Layer {
    constructor(inChannels, outChannels, deconv, kwargs) {
        super(kwargs);

        this.conv1 = new UNetConv(inChannels, 32, 64, false, {name: `${this.name}.conv1`});
        this.conv1_down = tf.layers.conv2d({
            filters: 64,
            kernelSize: 2,
            strides: 2,
            padding: 'valid',
            inputShape: [null, null, 64],
            name: `${this.name}.conv1_down`
        });
        this.conv2 = new UNetConv(64, 128, 64, true, {name: 'unet.conv2'});
        this.conv2_up = tf.layers.conv2dTranspose({
            filters: 64,
            kernelSize: 2,
            strides: 2,
            padding: 'valid',
            inputShape: [null, null, 64],
            name: `${this.name}.conv2_up`
        });
        this.conv3 = tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 1,
            padding: 'valid',
            inputShape: [null, null, 64],
            name: `${this.name}.conv3`
        });

        if (deconv) {
            this.conv_bottom = tf.layers.conv2dTranspose({
                filters: outChannels,
                kernelSize: 4,
                strides: 2,
                padding: 'valid',
                inputShape: [null, null, 64],
                name: `${this.name}.conv_bottom`
            });
        } else {
            this.conv_bottom = tf.layers.conv2d({
                filters: outChannels,
                kernelSize: 3,
                strides: 1,
                padding: 'valid',
                inputShape: [null, null, 64],
                name: `${this.name}.conv_bottom`
            });
        }
    }

    // Define the call method for forward pass
    call(inputs) {
        return tf.tidy(() => {
            const x1 = this.conv1.apply(inputs);

            const x2 = this.conv1_down.apply(x1);
            const x1_ = x1.slice([0, 4, 4, 0], [-1, x1.shape[1] - 8, x1.shape[2] - 8, -1]);
            const x2_ = tf.layers.leakyReLU({alpha: 0.1}).apply(x2);
            const x2__ = this.conv2.apply(x2_);
            const x2___ = this.conv2_up.apply(x2__);
            const x2____ = tf.layers.leakyReLU({alpha: 0.1}).apply(x2___);
            const x3 = this.conv3.apply(tf.add(x1_, x2____));
            const x3_ = tf.layers.leakyReLU({alpha: 0.1}).apply(x3);

            const z = this.conv_bottom.apply(x3_);
            return z.slice([0, 3, 3, 0], [-1, z.shape[1] - 6, z.shape[2] - 6, -1]);
        });

        // let x1 = this.conv1.apply(inputs);
        // let x2 = this.conv1_down.apply(x1);
        // x1 = x1.slice([0, 4, 4, 0], [-1, x1.shape[1] - 8, x1.shape[2] - 8, -1]);
        // x2 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x2);
        // x2 = this.conv2.apply(x2);
        // x2 = this.conv2_up.apply(x2);
        // x2 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x2);
        // let x3 = this.conv3.apply(tf.add(x1, x2));
        // x3 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x3);
        // let z = this.conv_bottom.apply(x3);
        // z = z.slice([0, 3, 3, 0], [-1, z.shape[1] - 6, z.shape[2] - 6, -1]);
        // return z;
    }

    // Define the auxiliary call_a method for partial forward pass
    call_a(inputs) {
        return tf.tidy(() => {
            const x1 = this.conv1.apply(inputs);
            const x2 = this.conv1_down.apply(x1);
            const x1Cropped = x1.slice([0, 4, 4, 0], [-1, x1.shape[1] - 8, x1.shape[2] - 8, -1]);
            const x2LeakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x2);
            const x2Conv = this.conv2.conv.apply(x2LeakyReLU);
            return [x1Cropped, x2Conv];
        });
    }

    // Define the auxiliary call_b method for the remaining forward pass
    call_b(x1, x2) {
        return tf.tidy(() => {
            const x2Up = this.conv2_up.apply(x2);
            const x2UpleakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x2Up);
            const x3 = this.conv3.apply(tf.add(x1, x2UpleakyReLU));
            const x3LeakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x3);
            const z = this.conv_bottom.apply(x3LeakyReLU);
            return z.slice([0, 3, 3, 0], [-1, z.shape[1] - 6, z.shape[2] - 6, -1]);
        });

    }
}

class UNet2 extends tf.layers.Layer {
    constructor(inChannels, outChannels, deconv, kwargs) {
        super(kwargs);

        this.conv1 = new UNetConv(inChannels, 32, 64, false, {name: `${this.name}.conv1`});
        this.conv1_down = tf.layers.conv2d({
            filters: 64,
            kernelSize: 2,
            strides: 2,
            padding: 'valid',
            inputShape: [null, null, 64],
            name: `${this.name}.conv1_down`
        });
        this.conv2 = new UNetConv(64, 64, 128, true, {name: `${this.name}.conv2`});
        this.conv2_down = tf.layers.conv2d({
            filters: 128,
            kernelSize: 2,
            strides: 2,
            padding: 'valid',
            inputShape: [null, null, 128],
            name: `${this.name}.conv2_down`
        });
        this.conv3 = new UNetConv(128, 256, 128, true, {name: `${this.name}.conv3`});
        this.conv3_up = tf.layers.conv2dTranspose({
            filters: 128,
            kernelSize: 2,
            strides: 2,
            padding: 'valid',
            inputShape: [null, null, 128],
            name: `${this.name}.conv3_up`
        });
        this.conv4 = new UNetConv(128, 64, 64, true, {name: `${this.name}.conv4`});
        this.conv4_up = tf.layers.conv2dTranspose({
            filters: 64,
            kernelSize: 2,
            strides: 2,
            padding: 'valid',
            inputShape: [null, null, 64],
            name: `${this.name}.conv4_up`
        });
        this.conv5 = tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 1,
            padding: 'valid',
            inputShape: [null, null, 64],
            name: `${this.name}.conv5`
        });

        if (deconv) {
            this.conv_bottom = tf.layers.conv2dTranspose({
                filters: outChannels,
                kernelSize: 4,
                strides: 2,
                padding: 'same',
                inputShape: [null, null, 64],
                name: `${this.name}.conv_bottom`
            });
        } else {
            this.conv_bottom = tf.layers.conv2d({
                filters: outChannels,
                kernelSize: 3,
                strides: 1,
                padding: 'valid',
                inputShape: [null, null, 64],
                name: `${this.name}.conv_bottom`
            });
        }
    }

    // Define the call method for forward pass
    call(inputs, alpha = 1) {
        return tf.tidy(() => {
            const x1 = this.conv1.apply(inputs);
            const x2 = this.conv1_down.apply(x1);
            const x1_ = x1.slice([0, 16, 16, 0], [-1, x1.shape[1] - 32, x1.shape[2] - 32, -1]);
            const x2_ = tf.layers.leakyReLU({alpha: 0.1}).apply(x2);
            const x2__ = this.conv2.apply(x2_);
            const x3 = this.conv2_down.apply(x2__);
            const x2___ = x2__.slice([0, 4, 4, 0], [-1, x2__.shape[1] - 8, x2__.shape[2] - 8, -1]);
            const x3_ = tf.layers.leakyReLU({alpha: 0.1}).apply(x3);

            const x3__ = this.conv3.apply(x3_);
            const x3___ = this.conv3_up.apply(x3__);
            const x3____ = tf.layers.leakyReLU({alpha: 0.1}).apply(x3___);
            const x4 = this.conv4.apply(tf.add(x2___, x3____));
            const x4_ = tf.mul(x4, alpha);
            const x4__ = this.conv4_up.apply(x4_);
            const x4___ = tf.layers.leakyReLU({alpha: 0.1}).apply(x4__);
            const x5 = this.conv5.apply(tf.add(x1_, x4___));
            const x5_ = tf.layers.leakyReLU({alpha: 0.1}).apply(x5);
            return this.conv_bottom.apply(x5_);

        });
        // console.log('inputs'+inputs.shape)
        // let x1 = this.conv1.apply(inputs);
        // let x2 = this.conv1_down.apply(x1);
        // x1 = x1.slice([0, 16, 16, 0], [-1, x1.shape[1] - 32, x1.shape[2] - 32, -1]);
        // x2 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x2);
        // x2 = this.conv2.apply(x2);
        // let x3 = this.conv2_down.apply(x2);
        // x2 = x2.slice([0, 4, 4, 0], [-1, x2.shape[1] - 8, x2.shape[2] - 8, -1]);
        // x3 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x3);
        // x3 = this.conv3.apply(x3);
        // x3 = this.conv3_up.apply(x3);
        // x3 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x3);
        // console.log('x2'+x2.shape)
        // console.log('x3'+x3.shape)
        // let x4 = this.conv4.apply(tf.add(x2, x3));
        // x4 = tf.mul(x4, alpha);
        // x4 = this.conv4_up.apply(x4);
        // x4 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x4);
        // let x5 = this.conv5.apply(tf.add(x1, x4));
        // x5 = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x5);
        // let z = this.conv_bottom.apply(x5);

        // return z;
    }

    // Define the auxiliary call_a method for partial forward pass
    call_a(inputs) {
        return tf.tidy(() => {
            const x1 = this.conv1.apply(inputs);
            const x2 = this.conv1_down.apply(x1);
            const x1Cropped = x1.slice([0, 16, 16, 0], [-1, x1.shape[1] - 32, x1.shape[2] - 32, -1]);
            const x2LeakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x2);
            const x2Conv = this.conv2.conv.apply(x2LeakyReLU);
            return [x1Cropped, x2Conv];
        });
    }

    // Define the auxiliary call_b method for the second part of forward pass
    call_b(x2) {
        return tf.tidy(() => {
            const x3 = this.conv2_down.apply(x2);
            const x2Slice = x2.slice([0, 4, 4, 0], [-1, x2.shape[1] - 8, x2.shape[2] - 8, -1]);
            const x3leakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x3);
            const x3leakyReLUConv = this.conv3.conv.apply(x3leakyReLU);
            return [x2Slice, x3leakyReLUConv];
        });
    }

    // Define the auxiliary call_c method for the third part of forward pass
    call_c(x2, x3) {
        return tf.tidy(() => {
            const x3Conv3_up = this.conv3_up.apply(x3);
            const x3Conv3_upLeakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x3Conv3_up);
            return this.conv4.apply(tf.add(x2, x3Conv3_upLeakyReLU));
        });
    }

    call_d(x1, x4) {
        return tf.tidy(() => {
            const x4Conv4_up = this.conv4_up.apply(x4);
            const x4Conv4_upLeakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x4Conv4_up);
            const x5 = this.conv5.apply(tf.add(x1, x4Conv4_upLeakyReLU));
            const x5LeakyReLU = tf.layers.leakyReLU({alpha: 0.1}).apply(x5);
            return this.conv_bottom.apply(x5LeakyReLU);
        });
    }
}

class UpCunet2x extends tf.layers.Layer {
    constructor(inChannels, outChannels, alpha, pro, half, kwargs) {
        super(kwargs);
        this.unet1 = new UNet1(inChannels, outChannels, true, {name: `${this.name}.unet1`});
        this.unet2 = new UNet2(inChannels, outChannels, false, {name: `${this.name}.unet2`});
        this.alpha = alpha;
        this.pro = pro;
        this.half = half;
    }

    call(inputs) {
        const x = tf.cast(inputs.div(tf.scalar(255 / 0.7)).add(tf.scalar(0.15)), 'float32');
        inputs.dispose()
        const [n, h0, w0, c] = x.shape;

        let tile_mode;
        if (h0 > w0) {
            tile_mode = Math.floor(h0 / 128);
        } else {
            tile_mode = Math.floor(w0 / 128);
        }

        let if_half;
        if (x.dtype === 'float16') {
            if_half = true;
        } else {
            if_half = false;
        }
        let crop_size;
        if (tile_mode === 0) {
            const ph = (Math.floor((h0 - 1) / 2) + 1) * 2;
            const pw = (Math.floor((w0 - 1) / 2) + 1) * 2;
            console.log(ph)
            console.log(pw)
            const x_ = x.pad([[0, 0], [18, 18 + ph - h0], [18, 18 + pw - w0], [0, 0]], 'reflect');
            x.dispose()
            console.log('x' + x_.shape)
            const x__ = this.unet1.apply(x_);
            x_.dispose()
            console.log('unaet1' + x__.shape)
            const x0 = this.unet2.apply(x__, this.alpha);
            console.log('unaet2' + x0.shape)


            const x___ = x__.slice([0, 20, 20, 0], [-1, x__.shape[1] - 40, x__.shape[2] - 40, -1]);

            const x____ = tf.add(x0, x___);

            if (w0 !== pw || h0 !== ph) {
                const x_____ = x____.slice([0, 0, 0, 0], [-1, h0 * 2, w0 * 2, -1]);
                if (this.pro) {
                    return tf.cast(tf.clipByValue(tf.round(tf.mul(tf.sub(x_____, 0.15), tf.scalar(255 / 0.7))), 0, 255), 'int32');
                } else {
                    return tf.cast(tf.clipByValue(tf.round(tf.mul(x_____, tf.scalar(255))), 0, 255), 'int32');
                }
            } else {
                const x_____ = x____
                if (this.pro) {
                    return tf.cast(tf.clipByValue(tf.round(tf.mul(tf.sub(x_____, 0.15), tf.scalar(255 / 0.7))), 0, 255), 'int32');
                } else {
                    return tf.cast(tf.clipByValue(tf.round(tf.mul(x_____, tf.scalar(255))), 0, 255), 'int32');
                }
            }

        } else /*if (tile_mode === 1) {
            let crop_size_w, crop_size_h;
            if (w0 >= h0) {
                crop_size_w = Math.floor((w0 - 1) / 4 * 4 + 4) / 2;
                crop_size_h = Math.floor((h0 - 1) / 2) * 2 + 2;
            } else {
                crop_size_h = Math.floor((h0 - 1) / 4 * 4 + 4) / 2;
                crop_size_w = Math.floor((w0 - 1) / 2) * 2 + 2;
            }
            crop_size = [crop_size_h, crop_size_w];
        } else if (tile_mode >= 2) */{
            tile_mode = Math.min(Math.floor(Math.min(h0, w0) / 128), tile_mode);
            const t2 = tile_mode * 2;
            crop_size = [
                Math.floor(Math.floor((h0 - 1) / t2) * t2 + t2 / tile_mode),
                Math.floor(Math.floor((w0 - 1) / t2) * t2 + t2 / tile_mode)
            ];
        }/* else {
            console.log("tile_mode config error");
            process.exit(233);
        }*/

        const ph = (Math.floor((h0 - 1) / crop_size[0]) + 1) * crop_size[0];
        const pw = (Math.floor((w0 - 1) / crop_size[1]) + 1) * crop_size[1];
        const xCrop = x.pad([[0, 0], [18, 18 + pw - w0], [18, 18 + ph - h0], [0, 0]], 'reflect');
        x.dispose()
        const [n1, h1, w1, c1] = xCrop.shape;
        let se_mean0;
        if (if_half) {
            se_mean0 = tf.zeros([n1, 1, 1, 64], 'float16');
        } else {
            se_mean0 = tf.zeros([n1, 1, 1, 64], 'float32');
        }
        let n_patch = 0;
        let tmp_dict = {};
        for (let i = 0; i < h1 - 36; i += crop_size[0]) {
            tmp_dict[i] = {};
            for (let j = 0; j < w1 - 36; j += crop_size[1]) {
                const x_crop = xCrop.slice([0, i, j, 0], [-1, crop_size[0] + 36, crop_size[1] + 36, -1]);

                //const [n2, h2, w2, c2] = x_crop.shape;
                const [tmp0, x_crop_tmp] = this.unet1.call_a(x_crop);
                const tmp_se_mean = tf.mean(x_crop_tmp, [1, 2], true);
                se_mean0 = tf.add(se_mean0, tmp_se_mean);
                n_patch += 1;
                tmp_dict[i][j] = [tmp0, x_crop_tmp];
                // tmp0.dispose()
                // x_crop_tmp.dispose()

                x_crop.dispose()
                tmp_se_mean.dispose()
            }
        }
        xCrop.dispose()
        se_mean0 = se_mean0.div(tf.scalar(n_patch));

        let se_mean1;
        if (if_half) {
            se_mean1 = tf.zeros([n1, 1, 1, 128], 'float16');
        } else {
            se_mean1 = tf.zeros([n1, 1, 1, 128], 'float32');
        }

        for (let i = 0; i < h1 - 36; i += crop_size[0]) {
            for (let j = 0; j < w1 - 36; j += crop_size[1]) {
                const [tmp0, x_crop] = tmp_dict[i][j];
                const x_crop_tmp = this.unet1.conv2.seblock.mean_call(x_crop, se_mean0);
                const opt_unet1 = this.unet1.call_b(tmp0, x_crop_tmp);
                x_crop_tmp.dispose()
                const [tmp_x1, tmp_x2] = this.unet2.call_a(opt_unet1);
                const opt_unet1Crop = opt_unet1.slice([0, 20, 20, 0], [-1, h1 - 40, w1 - 40, -1]);
                opt_unet1.dispose()
                const tmp_se_mean = tf.mean(tmp_x2, [1, 2], true);
                se_mean1 = se_mean1.add(tmp_se_mean);
                tmp_se_mean.dispose()
                tmp_dict[i][j][0].dispose()
                tmp_dict[i][j][1].dispose()
                tmp_dict[i][j] = [opt_unet1Crop, tmp_x1, tmp_x2];
            }
        }
        se_mean1 = se_mean1.div(tf.scalar(n_patch));
        se_mean0.dispose()
        if (if_half) {
            se_mean0 = tf.zeros([n1, 1, 1, 128], 'float16');
        } else {
            se_mean0 = tf.zeros([n1, 1, 1, 128], 'float32');
        }

        for (let i = 0; i < h1 - 36; i += crop_size[0]) {
            for (let j = 0; j < w1 - 36; j += crop_size[1]) {
                const [opt_unet1, tmp_x1, tmp_x2] = tmp_dict[i][j];
                const tmp_x2_tmp = this.unet2.conv2.seblock.mean_call(tmp_x2, se_mean1);
                tmp_x2.dispose()
                const [tmp_x2_, tmp_x3] = this.unet2.call_b(tmp_x2_tmp);
                const tmp_se_mean = tf.mean(tmp_x3, [1, 2], true);
                se_mean0 = se_mean0.add(tmp_se_mean);
                tmp_se_mean.dispose()
                tmp_dict[i][j] = [opt_unet1, tmp_x1, tmp_x2_, tmp_x3];
            }
        }
        se_mean0 = se_mean0.div(tf.scalar(n_patch));

        se_mean1.dispose();
        if (if_half) {
            se_mean1 = tf.zeros([n1, 1, 1, 64], 'float16');
        } else {
            se_mean1 = tf.zeros([n1, 1, 1, 64], 'float32');
        }

        for (let i = 0; i < h1 - 36; i += crop_size[0]) {
            for (let j = 0; j < w1 - 36; j += crop_size[1]) {
                const [opt_unet1, tmp_x1, tmp_x2, tmp_x3] = tmp_dict[i][j];
                const tmp_x3_tmp = this.unet2.conv3.seblock.mean_call(tmp_x3, se_mean0);
                tmp_x3.dispose()
                const tmp_x4 = this.unet2.call_c(tmp_x2, tmp_x3_tmp).mul(tf.scalar(this.alpha));
                tmp_x3_tmp.dispose()
                tmp_x2.dispose()
                const tmp_se_mean = tf.mean(tmp_x4, [1, 2], true);
                se_mean1 = se_mean1.add(tmp_se_mean);
                tmp_se_mean.dispose()
                tmp_dict[i][j] = [opt_unet1, tmp_x1, tmp_x4];
            }
        }
        se_mean1 = se_mean1.div(tf.scalar(n_patch));

        const res = [];
        for (let i = 0; i < h1 - 36; i += crop_size[0]) {
            const temp = [];
            for (let j = 0; j < w1 - 36; j += crop_size[1]) {
                const [x_, tmp_x1, tmp_x4] = tmp_dict[i][j];
                console.log(se_mean1)
                console.log(tmp_x4)
                const tmp_x4_tmp = this.unet2.conv4.seblock.mean_call(tmp_x4, se_mean1);
                console.log(tmp_x4_tmp)
                const x0 = this.unet2.call_d(tmp_x1, tmp_x4_tmp);
                const x__ = x_.add(x0);
                if (this.pro) {
                    temp.push(tf.cast(tf.clipByValue(tf.round(x__.sub(0.15).mul(tf.scalar(255 / 0.7))), 0, 255), 'int32'));
                } else {
                    temp.push(tf.cast(tf.clipByValue(tf.round(x__.mul(tf.scalar(255))), 0, 255), 'int32'));
                }
                x__.dispose();
                x0.dispose();
                tmp_dict[i][j][0].dispose()
                tmp_dict[i][j][1].dispose()
                tmp_dict[i][j][2].dispose()
            }
            res.push(tf.concat(temp, 2));
            temp.dispose()
        }
        const result = tf.concat(res, 1);
        res.dispose()
        if (w0 !== pw || h0 !== ph) return result.slice([0, 0, 0, 0], [-1, h0 * 2, w0 * 2, -1]);
        return result;
    }
}

export {
    UpCunet2x,
    SEBlock,
    UNetConv,
    UNet1
}