import numpy as np
import tensorflow as tf
from PIL import Image


# =========================
# 工具函数
# =========================

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32)
    return img_np


def save_image(img_np, path):
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def tflite_inference(interpreter, *inputs):
    """调用 tflite 模型，返回输出"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, inp in enumerate(inputs):
        interpreter.set_tensor(input_details[i]['index'], inp)

    interpreter.invoke()
    outputs = [interpreter.get_tensor(o['index']) for o in output_details]
    return outputs if len(outputs) > 1 else outputs[0]


def mean_hw(x):
    return np.mean(x, axis=(1, 2), keepdims=True)


def crop_image(x, i, j, h, w):
    return x[:, i:i + h, j:j + w, :]


def paste_image(out, tile, i, j):
    h, w = tile.shape[1:3]
    out[:, i:i + h, j:j + w, :] = tile


# =========================
# 加载 TFLite 模型
# =========================

interpreterA = tf.lite.Interpreter("stageA.tflite");
interpreterA.allocate_tensors()
interpreterB = tf.lite.Interpreter("stageB.tflite");
interpreterB.allocate_tensors()
interpreterC = tf.lite.Interpreter("stageC.tflite");
interpreterC.allocate_tensors()
interpreterD = tf.lite.Interpreter("stageD.tflite");
interpreterD.allocate_tensors()
interpreterE = tf.lite.Interpreter("stageE.tflite");
interpreterE.allocate_tensors()

# =========================
# 加载图片并归一化
# =========================

img_np = load_image("test.png")  # HWC
h0, w0, c = img_np.shape
x = img_np / (255 / 0.7) + 0.15
x = np.expand_dims(x, axis=0).astype(np.float32)  # NHWC

# =========================
# 设置 tile 大小和 padding
# =========================

tile_h, tile_w = 128, 128
pad = 36
stride_h, stride_w = tile_h, tile_w

# 输出图初始化
out_h, out_w = h0 * 2, w0 * 2
out = np.zeros((1, out_h, out_w, 3), dtype=np.float32)

# =========================
# 分块推理
# =========================

# 用字典保存每个 tile 的中间结果
tile_dict = {}

# StageA，先统计 se_mean0
se_mean0 = np.zeros((1, 1, 1, 64), dtype=np.float32)
n_patch = 0

for i in range(0, h0, stride_h):
    for j in range(0, w0, stride_w):
        # 切块并加 padding
        h_crop = min(tile_h, h0 - i)
        w_crop = min(tile_w, w0 - j)
        tile = x[:, i:i + h_crop + pad, j:j + w_crop + pad, :]

        # StageA
        x1, x2 = tflite_inference(interpreterA, tile)
        se_mean0 += mean_hw(x2)
        n_patch += 1
        tile_dict[(i, j)] = {'x1': x1, 'x2': x2}

se_mean0 /= n_patch

# StageB
se_mean1 = np.zeros((1, 1, 1, 128), dtype=np.float32)
for key, val in tile_dict.items():
    out1, u2_x1, u2_x2 = tflite_inference(interpreterB, val['x1'], val['x2'], se_mean0)
    se_mean1 += mean_hw(u2_x2)
    val.update({'out1': out1, 'u2_x1': u2_x1, 'u2_x2': u2_x2})
se_mean1 /= n_patch

# StageC
se_mean2 = np.zeros((1, 1, 1, 128), dtype=np.float32)
for val in tile_dict.values():
    x2_c, x3 = tflite_inference(interpreterC, val['u2_x2'], se_mean1)
    se_mean2 += mean_hw(x3)
    val.update({'x2_c': x2_c, 'x3': x3})
se_mean2 /= n_patch

# StageD
se_mean3 = np.zeros((1, 1, 1, 64), dtype=np.float32)
for val in tile_dict.values():
    x4 = tflite_inference(interpreterD, val['x2_c'], val['x3'], se_mean2)
    se_mean3 += mean_hw(x4)
    val['x4'] = x4
se_mean3 /= n_patch

# StageE + 拼接
for (i, j), val in tile_dict.items():
    out_tile = tflite_inference(interpreterE, val['u2_x1'], val['x4'], se_mean3)
    # 残差相加
    out_tile += val['out1']
    # 反归一化
    out_tile = (out_tile - 0.15) * (255 / 0.7)
    # paste 到输出图
    paste_image(out, out_tile, i * 2, j * 2)

# 保存输出
save_image(out[0], "/Volumes/Home/oysterqaq/PycharmProjects/ACG2vec-model/real-cugan_tf/tmp/1691504166649551.png")
print("✅ 推理完成，输出保存为 output_test.png")
