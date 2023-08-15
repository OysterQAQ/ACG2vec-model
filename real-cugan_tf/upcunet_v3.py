'''
cache_mode:
0:使用cache缓存必要参数
1:使用cache缓存必要参数，对cache进行8bit量化节省显存，带来小许延时增长
2:不使用cache，耗时约为mode0的2倍，但是显存不受输入图像分辨率限制，tile_mode填得够大，1.5G显存可超任意比例
'''
import os
import sys

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

root_path = os.path.abspath('.')
sys.path.append(root_path)


def q(inp, cache_mode):
    maxx = inp.max()
    minn = inp.min()
    delta = maxx - minn
    if (cache_mode == 2):
        return ((inp - minn) / delta * 255).round().byte().cpu(), delta, minn, inp.device  # 大概3倍延时#太慢了，屏蔽该模式
    elif (cache_mode == 1):
        return ((inp - minn) / delta * 255).round().byte(), delta, minn, inp.device  # 不用CPU转移


def dq(inp, if_half, cache_mode, delta, minn, device):
    if (cache_mode == 2):
        if (if_half == True):
            return inp.to(device).half() / 255 * delta + minn
        else:
            return inp.to(device).float() / 255 * delta + minn
    elif (cache_mode == 1):
        if (if_half == True):
            return inp.half() / 255 * delta + minn  # 不用CPU转移
        else:
            return inp.float() / 255 * delta + minn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        print(in_channels)
        print(in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x


class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            #self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z



class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha=1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        x4 *= alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):  # conv234结尾有se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x2, x3

    def forward_c(self, x2, x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1, x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z


class UpCunet2x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x, tile_mode, cache_mode, alpha, pro):
        n, c, h0, w0 = x.shape
        if ("Half" in x.type()):
            if_half = True
        else:
            if_half = False
        if (tile_mode == 0):  # 不tile
            #将奇数变为下一个最近的偶数
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            #填充h w 维度 （h前填充，h后填充，w前填充，w后填充）
            x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x, alpha)
            x = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
            if (pro):
                return ((x - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
            else:
                return (x * 255).round().clamp_(0, 255).byte()
        elif (tile_mode == 1):  # 对长边减半
            if (w0 >= h0):
                crop_size_w = ((w0 - 1) // 4 * 4 + 4) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_h = (h0 - 1) // 2 * 2 + 2  # 能被2整除
            else:
                crop_size_h = ((h0 - 1) // 4 * 4 + 4) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_w = (w0 - 1) // 2 * 2 + 2  # 能被2整除
            crop_size = (crop_size_h, crop_size_w)
        elif (tile_mode >= 2):
            tile_mode=min(h0, w0) // 128
            #tile_mode = min(min(h0, w0) // 128, int(tile_mode))  # 最小短边为128*128
            t2 = tile_mode * 2
            crop_size = (((h0 - 1) // t2 * t2 + t2) // tile_mode, ((w0 - 1) // t2 * t2 + t2) // tile_mode)
        else:
            print("tile_mode config error")
            os._exit(233)

        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')
        n, c, h, w = x.shape
        if (if_half):
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        n_patch = 0
        tmp_list = []
        i_length=(h - 36)//crop_size[0]
        j_length=(w - 36) //crop_size[1]
        for i in range(i_length):
            tmp_list.append([])
            for j in range(j_length):
                print(f"i:{i},j:{j}")
                x_crop = x[:, :, i*crop_size[0]:(i +1)* crop_size[0] + 36, j*(crop_size[1]):(j+1)* crop_size[1] + 36]
                n, c1, h1, w1 = x_crop.shape
                tmp0, x_crop = self.unet1.forward_a(x_crop)
                if (if_half):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3), keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3), keepdim=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
                #tmp_dict[i][j] = (tmp0, x_crop)
                tmp_list[i].append([])
                # tmp_list[i][j].append(tmp0)
                # tmp_list[i][j].append(x_crop)
                tmp_list[i][j] = [tmp0, x_crop]


        se_mean0 /= n_patch
        if (if_half):
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)

        print(f"{'-'*10}")



        for i in range(i_length):
            for j in range(j_length):
                print(f"i:{i},j:{j}")
                tmp0, x_crop =  tmp_list[i][j][0],tmp_list[i][j][1]
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                opt_unet1 = F.pad(opt_unet1, (-20, -20, -20, -20))
                if (cache_mode): opt_unet1, tmp_x1 = q(opt_unet1, cache_mode), q(tmp_x1, cache_mode)
                if (if_half):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3), keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3), keepdim=True)
                if (cache_mode): tmp_x2 = q(tmp_x2, cache_mode)
                se_mean1 += tmp_se_mean
                tmp_list[i][j] = [opt_unet1, tmp_x1, tmp_x2]

        se_mean1 /= n_patch
        if (if_half):
            se_mean0 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean0 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)

        print(f"{'-' * 10}")
        for i in range(i_length):
            for j in range(j_length):
                print(f"i:{i},j:{j}")
                opt_unet1, tmp_x1, tmp_x2 = tmp_list[i][j][0],tmp_list[i][j][1],tmp_list[i][j][2]
                if (cache_mode): tmp_x2 = dq(tmp_x2[0], if_half, cache_mode, tmp_x2[1], tmp_x2[2], tmp_x2[3])
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                if (cache_mode): tmp_x2 = q(tmp_x2, cache_mode)
                if (if_half):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3), keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3), keepdim=True)
                if (cache_mode): tmp_x3 = q(tmp_x3, cache_mode)
                se_mean0 += tmp_se_mean
                tmp_list[i][j] = [opt_unet1, tmp_x1, tmp_x2, tmp_x3]
        se_mean0 /= n_patch
        if (if_half):
            se_mean1 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean1 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        print(f"{'-' * 10}")
        for i in range(i_length):
            for j in range(j_length):
                print(f"i:{i},j:{j}")
                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tmp_list[i][j][0],tmp_list[i][j][1],tmp_list[i][j][2],tmp_list[i][j][3]
                if (cache_mode): tmp_x3 = dq(tmp_x3[0], if_half, cache_mode, tmp_x3[1], tmp_x3[2], tmp_x3[3])
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(tmp_x3, se_mean0)
                if (cache_mode): tmp_x2 = dq(tmp_x2[0], if_half, cache_mode, tmp_x2[1], tmp_x2[2], tmp_x2[3])
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3) * alpha
                if (if_half):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3), keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3), keepdim=True)
                if (cache_mode): tmp_x4 = q(tmp_x4, cache_mode)
                se_mean1 += tmp_se_mean
                tmp_list[i][j] = [opt_unet1, tmp_x1, tmp_x4]
        se_mean1 /= n_patch
        res = torch.zeros((n, c, h * 2 - 72, w * 2 - 72), dtype=torch.uint8, device=x.device)
        print(f"{'-' * 10}")
        for i in range((h - 36) // crop_size[0]):
            for j in range((w - 36) // crop_size[1]):
                print(f"i:{i},j:{j}")
                x, tmp_x1, tmp_x4 = tmp_list[i][j][0],tmp_list[i][j][1],tmp_list[i][j][2]
                if (cache_mode): tmp_x4 = dq(tmp_x4[0], if_half, cache_mode, tmp_x4[1], tmp_x4[2], tmp_x4[3])
                tmp_x4 = self.unet2.conv4.seblock.forward_mean(tmp_x4, se_mean1)
                if (cache_mode): tmp_x1 = dq(tmp_x1[0], if_half, cache_mode, tmp_x1[1], tmp_x1[2], tmp_x1[3])
                x0 = self.unet2.forward_d(tmp_x1, tmp_x4)
                if (cache_mode): x = dq(x[0], if_half, cache_mode, x[1], x[2], x[3])
                #del tmp_list[i][j]
                x = torch.add(x0, x)  # x0是unet2的最终输出
                if (pro):
                    res[:, :, i*crop_size[0] * 2:i*crop_size[0] * 2 + h1 * 2 - 72, j*crop_size[1] * 2:j *crop_size[1]* 2 + w1 * 2 - 72] = (
                            (x - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
                else:
                    res[:, :, i*crop_size[0] * 2:i*crop_size[0] * 2 + h1 * 2 - 72, j*crop_size[1] * 2:j*crop_size[1] * 2 + w1 * 2 - 72] = (x * 255).round().clamp_(0,
                                                                                                               255).byte()
        del tmp_list
        # torch.cuda.empty_cache()
        if (w0 != pw or h0 != ph): res = res[:, :, :h0 * 2, :w0 * 2]
        return res



class RealWaifuUpScaler(object):
    def __init__(self, scale, weight_path, half, device):
        weight = torch.load(weight_path, map_location="cpu")
        self.pro = "pro" in weight
        if (self.pro): del weight["pro"]
        #
        self.model = eval("UpCunet%sx" % scale)()
        if (half == True):
            self.model = self.model.half().to(device)
        else:
            self.model = self.model.to(device)
        self.model.load_state_dict(weight, strict=True)
        self.model.eval()
        self.half = half
        self.device = device

    def np2tensor(self, np_frame):
        if (self.pro):
            # half半精度
            if (self.half == False):
                return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).float() / (
                        255 / 0.7) + 0.15
            else:
                return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).half() / (
                        255 / 0.7) + 0.15
        else:
            if (self.half == False):
                return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).float() / 255
            else:
                return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).half() / 255

    def tensor2np(self, tensor):
        return (np.transpose(tensor.squeeze().cpu().numpy(), (1, 2, 0)))

    def __call__(self, frame, tile_mode, cache_mode, alpha):
        with torch.no_grad():
            # numpy转为tensor 将HWC转为HCW
            tensor = self.np2tensor(frame)
            if (cache_mode == 3):
                result = self.tensor2np(self.model.forward_gap_sync(tensor, tile_mode, alpha, self.pro))
            elif (cache_mode == 2):
                result = self.tensor2np(self.model.forward_fast_rough(tensor, tile_mode, alpha, self.pro))
            else:
                result = self.tensor2np(self.model(tensor, tile_mode, cache_mode, alpha, self.pro))
        return result


if __name__ == "__main__":
    ###########inference_img
    import time, cv2, sys
    from time import time as ttime
    """
    mode: 在其中填写video或者image决定超视频还是超图像；
scale: 超分倍率；
model_path: 填写模型参数路径(目前3倍4倍超分只有3个模型，2倍有4个不同降噪强度模型和1个保守模型)；
device: 显卡设备号。如果有多卡超图片，建议手工将输入任务平分到不同文件夹，填写不同的卡号；
超图像，需要填写输入输出文件夹；超视频，需要指定输入输出视频的路径。
cache_mode:根据个人N卡显存大小调节缓存模式.mode2/3可超任意大小分辨率（瓶颈不在显存）图像
0: 默认使用cache缓存必要参数
1: 使用cache缓存必要参数，对缓存进行8bit量化节省显存，带来15%延时增长，肉眼完全无法感知的有损模式
tile: 数字越大显存需求越低，相对地可能会小幅降低推理速度 {0, 1, x, auto}
0: 直接使用整张图像进行推理，大显存用户或者低分辨率需求可使用
1: 对长边平分切成两块推理
x: 宽高分别平分切成x块推理
auto: 当输入图片文件夹图片分辨率不同时，填写auto自动调节不同图片tile模式，未来将支持该模式。

alpha: 该值越大AI修复程度、痕迹越小，越模糊；alpha越小处理越烈，越锐化，色偏（对比度、饱和度增强）越大；默认为1不调整，推荐调整区间(0.7,1.3)；
half: 半精度推理，>=20系显卡直接写True开着就好


tile 0 
cache_mode 1

    """
    tile_mode=3
    cache_mode=0
    alpha=1
    scale=2
    weight_name = "weights_pro/pro-no-denoise-up2x.pth".split("/")[-1].split(".")[0]
    # cpu下 half需要为False
    upscaler2x = RealWaifuUpScaler(scale, "weights_pro/pro-no-denoise-up2x.pth", half=False, device="cpu:0")
    frame = cv2.imread("inputs/test3.jpeg")[:, :, [2, 1, 0]]

    result = upscaler2x(frame, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)[:, :, ::-1]
    cv2.imwrite("output/pt_2x_3.png", result)

    # input_dir = "%s/inputs" % root_path
    # output_dir = "%s/output-dir-all-test" % root_path
    # os.makedirs(output_dir, exist_ok=True)
    # for name in os.listdir(input_dir):
    #     print(name)
    #     tmp = name.split(".")
    #     inp_path = os.path.join(input_dir, name)
    #     suffix = tmp[-1]
    #     prefix = ".".join(tmp[:-1])
    #     tmp_path = os.path.join(root_path, "tmp", "%s.%s" % (int(time.time() * 1000000), suffix))
    #     print(inp_path, tmp_path)
    #     # 支持中文路径
    #     # os.link(inp_path, tmp_path)#win用硬链接
    #     os.symlink(inp_path, tmp_path)  # linux用软链接
    #     # 加载图片为矩阵 H W C（BGR） (2, 1, 0指的是指读取rgb)
    #     frame = cv2.imread(tmp_path)[:, :, [2, 1, 0]]
    #     t0 = ttime()
    #     # 调用upscaler2x __call__
    #     result = upscaler2x(frame, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)[:, :, ::-1]
    #     t1 = ttime()
    #     print(prefix, "done", t1 - t0, "tile_mode=%s" % tile_mode, cache_mode)
    #     tmp_opt_path = os.path.join(root_path, "tmp", "%s.%s" % (int(time.time() * 1000000), suffix))
    #     cv2.imwrite(tmp_opt_path, result)
    #     n = 0
    #     while (1):
    #         if (n == 0):
    #             suffix = "_%sx_tile%s_cache%s_alpha%s_%s.png" % (
    #                 scale, tile_mode, cache_mode, alpha, weight_name)
    #         else:
    #             suffix = "_%sx_tile%s_cache%s_alpha%s_%s_%s.png" % (
    #                 scale, tile_mode, cache_mode, alpha, weight_name, n)
    #         if (os.path.exists(os.path.join(output_dir, prefix + suffix)) == False):
    #             break
    #         else:
    #             n += 1
    #     final_opt_path = os.path.join(output_dir, prefix + suffix)
    #     os.rename(tmp_opt_path, final_opt_path)
    #     os.remove(tmp_path)

    # for weight_path, scale in [("weights_v3/up2x-latest-denoise3x.pth", 2), ("weights_v3/up3x-latest-denoise3x.pth", 3),
    #                            ("weights_v3/up4x-latest-denoise3x.pth", 4), ("weights_pro/pro-denoise3x-up2x.pth", 2),
    #                            ("weights_pro/pro-denoise3x-up3x.pth", 3), ]:
    #
    #
    #     for tile_mode in [0, 5]:
    #         for cache_mode in [0, 1, 2, 3]:
    #             for alpha in [1]:
    #                 weight_name = weight_path.split("/")[-1].split(".")[0]
    #                 #cpu下 half需要为False
    #                 upscaler2x = RealWaifuUpScaler(scale, weight_path, half=False, device="cpu:0")
    #                 input_dir = "%s/inputs" % root_path
    #                 output_dir = "%s/output-dir-all-test" % root_path
    #                 os.makedirs(output_dir, exist_ok=True)
    #                 for name in os.listdir(input_dir):
    #                     print(name)
    #                     tmp = name.split(".")
    #                     inp_path = os.path.join(input_dir, name)
    #                     suffix = tmp[-1]
    #                     prefix = ".".join(tmp[:-1])
    #                     tmp_path = os.path.join(root_path, "tmp", "%s.%s" % (int(time.time() * 1000000), suffix))
    #                     print(inp_path, tmp_path)
    #                     # 支持中文路径
    #                     # os.link(inp_path, tmp_path)#win用硬链接
    #                     os.symlink(inp_path, tmp_path)  # linux用软链接
    #                     # 加载图片为矩阵 H W C（BGR） (2, 1, 0指的是指读取rgb)
    #                     frame = cv2.imread(tmp_path)[:, :, [2, 1, 0]]
    #                     t0 = ttime()
    #                     # 调用upscaler2x __call__
    #                     result = upscaler2x(frame, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)[:, :, ::-1]
    #                     t1 = ttime()
    #                     print(prefix, "done", t1 - t0, "tile_mode=%s" % tile_mode, cache_mode)
    #                     tmp_opt_path = os.path.join(root_path, "tmp", "%s.%s" % (int(time.time() * 1000000), suffix))
    #                     cv2.imwrite(tmp_opt_path, result)
    #                     n = 0
    #                     while (1):
    #                         if (n == 0):
    #                             suffix = "_%sx_tile%s_cache%s_alpha%s_%s.png" % (
    #                                 scale, tile_mode, cache_mode, alpha, weight_name)
    #                         else:
    #                             suffix = "_%sx_tile%s_cache%s_alpha%s_%s_%s.png" % (
    #                                 scale, tile_mode, cache_mode, alpha, weight_name, n)
    #                         if (os.path.exists(os.path.join(output_dir, prefix + suffix)) == False):
    #                             break
    #                         else:
    #                             n += 1
    #                     final_opt_path = os.path.join(output_dir, prefix + suffix)
    #                     os.rename(tmp_opt_path, final_opt_path)
    #                     os.remove(tmp_path)
