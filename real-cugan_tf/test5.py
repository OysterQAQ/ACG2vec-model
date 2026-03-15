import numpy as np
import tensorflow as tf
import os
from typing import Dict, Tuple, List
np.set_printoptions(suppress=True)  # 禁止科学计数法显示

# 输出形状 → 语义名字映射
STAGE_OUTPUTS_WITH_SHAPE = {
    "stage_1": {
        "tmp0": [1, 152, 152, 64],
        "x_crop": [1, 76, 76, 64],
        "tmp_se_mean": [1, 1, 1, 64]
    },
    "stage_2": {
        "tmp_se_mean": [1, 1, 1, 128],
        "tmp_x2": [1, 142, 142, 128],
        "tmp_x1": [1, 260, 260, 64],
        "opt_unet1": [1, 256, 256, 3]
    },
    "stage_3": {
        "tmp_x2": [1, 134, 134, 128],
        "tmp_x3": [1, 67, 67, 128],
        "tmp_se_mean": [1, 1, 1, 128]
    },
    "stage_4": {
        "tmp_se_mean": [1, 1, 1, 64],
        "tmp_x4": [1, 130, 130, 64]
    },
    "stage_5": {
        "x_out": [1, 256, 256, 3]
    },
}


class UpCunet2x_TFLite:
    """使用导出的5个阶段TFLite模型实现超分辨率"""

    def __init__(self, model_dir: str = ".", half: bool = False, pro: bool = True, alpha: float = 0.7, crop_size: int = 128):
        self.half = half
        self.pro = pro
        self.alpha = alpha

        self.interpreters = {}
        self.input_details = {}
        self.output_details = {}
        self.output_shapes = {}  # 保存每个阶段的输出形状信息
        self.load_tflite_models(model_dir)
        self.crop_size = crop_size

    def load_tflite_models(self, model_dir: str):
        """加载TFLite模型"""
        stage_names = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5"]
        for name in stage_names:
            tflite_path = os.path.join(model_dir, f"{name}.tflite")
            if not os.path.exists(tflite_path):
                raise FileNotFoundError(f"TFLite模型文件未找到: {tflite_path}")
            print(f"加载TFLite模型: {tflite_path}")

            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            self.interpreters[name] = interpreter
            self.input_details[name] = interpreter.get_input_details()
            self.output_details[name] = interpreter.get_output_details()

            # 保存输出形状信息
            self.output_shapes[name] = {}
            for out in self.output_details[name]:
                shape = tuple(out['shape'])
                self.output_shapes[name][shape] = out.get('name', f'output_{len(self.output_shapes[name])}')

            # 打印输入输出信息
            print(f"{name} 输入详情:")
            for i, inp in enumerate(self.input_details[name]):
                print(f"  输入 {i}: 名称={inp.get('name', 'unknown')}, 形状={inp['shape']}, dtype={inp['dtype']}")
            print(f"{name} 输出详情:")
            for i, out in enumerate(self.output_details[name]):
                print(f"  输出 {i}: 名称={out.get('name', 'unknown')}, 形状={out['shape']}, dtype={out['dtype']}")

        # 验证输出形状映射
        self.validate_output_shapes()
        print("所有TFLite模型加载完成")

    def validate_output_shapes(self):
        """验证输出形状映射是否正确"""
        print("\n验证输出形状映射...")
        for stage_name, shape_mapping in STAGE_OUTPUTS_WITH_SHAPE.items():
            print(f"\n验证 {stage_name}:")
            for semantic_name, expected_shape in shape_mapping.items():
                expected_shape_tuple = tuple(expected_shape)

                # 检查TFLite模型的实际输出
                actual_shapes = list(self.output_shapes[stage_name].keys())

                # 查找匹配的形状
                matched = False
                for actual_shape in actual_shapes:
                    if actual_shape == expected_shape_tuple:
                        matched = True
                        print(f"  ✓ {semantic_name}: 期望 {expected_shape}, 实际 {actual_shape}")
                        break

                if not matched:
                    print(f"  ✗ {semantic_name}: 期望 {expected_shape}, 实际输出形状: {actual_shapes}")

        print("\n验证完成")

    def match_output_by_shape(self, stage_name: str, output_tensor: np.ndarray) -> str:
        """
        根据形状匹配输出张量的语义名称

        Args:
            stage_name: 阶段名称
            output_tensor: 输出张量

        Returns:
            匹配的语义名称
        """
        tensor_shape = output_tensor.shape

        # 查找STAGE_OUTPUTS_WITH_SHAPE中匹配的形状
        shape_mapping = STAGE_OUTPUTS_WITH_SHAPE.get(stage_name, {})
        for semantic_name, expected_shape in shape_mapping.items():
            if tuple(expected_shape) == tensor_shape:
                return semantic_name

        # 如果没有精确匹配，尝试模糊匹配（考虑动态形状中的None）
        for semantic_name, expected_shape in shape_mapping.items():
            if len(expected_shape) != len(tensor_shape):
                continue

            matched = True
            for expected_dim, actual_dim in zip(expected_shape, tensor_shape):
                if expected_dim is not None and expected_dim != actual_dim:
                    matched = False
                    break

            if matched:
                return semantic_name

        # 如果还是没有匹配，使用默认名称
        return f"unknown_output_{tensor_shape}"

    def run_tflite_model(self, stage_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        运行TFLite模型推理
        输入按形状匹配，输出按形状匹配语义名字
        """
        if stage_name == "stage_2":
            for key, value in inputs.items():
                print(f"input {stage_name} Key: {key}, Value: {value}")


        interpreter = self.interpreters[stage_name]
        input_details = self.input_details[stage_name]
        output_details = self.output_details[stage_name]

        # 为每个模型输入匹配数据
        used_keys = set()
        for input_info in input_details:
            idx = input_info['index']
            input_shape = tuple(input_info['shape'])
            input_name = input_info.get('name', f'input_{idx}')

            # 首先尝试按名称匹配
            if input_name in inputs:
                arr = inputs[input_name]
                if arr.shape == input_shape or (None in input_shape and len(arr.shape) == len(input_shape)):
                    interpreter.set_tensor(idx, arr.astype(input_info['dtype']))
                    used_keys.add(input_name)
                    print(f"  {stage_name}: 按名称匹配输入 {input_name} -> 形状 {arr.shape}")
                    continue

            # 按形状匹配输入
            matched = False
            for key, arr in inputs.items():
                if key in used_keys:
                    continue

                # 检查形状是否匹配
                shape_matched = False
                if arr.shape == input_shape:
                    shape_matched = True
                elif None in input_shape and len(arr.shape) == len(input_shape):
                    # 动态形状匹配
                    shape_matched = True
                    for i, (expected_dim, actual_dim) in enumerate(zip(input_shape, arr.shape)):
                        if expected_dim is not None and expected_dim != actual_dim:
                            shape_matched = False
                            break

                if shape_matched:
                    interpreter.set_tensor(idx, arr.astype(input_info['dtype']))
                    used_keys.add(key)
                    matched = True
                    print(f"  {stage_name}: 按形状匹配输入 {key} -> 形状 {arr.shape} (目标形状: {input_shape})")
                    break

            if not matched:
                available_inputs = {k: v.shape for k, v in inputs.items() if k not in used_keys}
                raise ValueError(
                    f"未找到匹配形状 {input_shape} 的输入张量，stage={stage_name}。可用输入: {available_inputs}")

        # 运行推理
        interpreter.invoke()

        # 获取输出，按形状匹配语义名字
        outputs = {}
        for output_info in output_details:
            output_data = interpreter.get_tensor(output_info['index'])
            semantic_name = self.match_output_by_shape(stage_name, output_data)
            outputs[semantic_name] = output_data

            # 调试信息
            print(f"  {stage_name}: 输出 {semantic_name} -> 形状 {output_data.shape}")
        if stage_name == "stage_2":
            for key, value in outputs.items():
                print(f"output {stage_name} Key: {key}, Value: {value}")
        return outputs

    def prepare_input(self, image_array: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        准备输入，返回处理后的图片和padding信息
        """
        original_shape = image_array.shape
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        x = tf.cast(image_array, dtype=tf.float32)
        if self.pro:
            x = x / (255.0 / 0.7) + 0.15
        else:
            x = x / 255.0

        batch, height, width, channels = x.shape

        # 记录原始尺寸
        original_h, original_w = height, width

        # 计算需要padding的尺寸
        pad_height = (self.crop_size - height % self.crop_size) % self.crop_size
        pad_width = (self.crop_size - width % self.crop_size) % self.crop_size

        x = tf.pad(x, [[0, 0], [0, pad_height], [0, pad_width], [0, 0]], mode="REFLECT")

        # 返回padding信息和处理后的图片
        padding_info = {
            'original_shape': original_shape,
            'original_h': original_h,
            'original_w': original_w,
            'pad_height': pad_height,
            'pad_width': pad_width
        }
        return x.numpy(), padding_info

    def process_image(self, image: np.ndarray) -> np.ndarray:
        # 准备输入，获取padding信息
        x, padding_info = self.prepare_input(image)
        original_h, original_w = padding_info['original_h'], padding_info['original_w']

        n, h0, w0, c = x.shape

        crop_size_h, crop_size_w = self.crop_size, self.crop_size
        ph = ((h0 - 1) // crop_size_h + 1) * crop_size_h
        pw = ((w0 - 1) // crop_size_w + 1) * crop_size_w
        x_padded = np.pad(x, [[0, 0], [18, 18 + ph - h0], [18, 18 + pw - w0], [0, 0]], mode='reflect')
        n, h, w, c = x_padded.shape

        se_mean0 = np.zeros((n, 1, 1, 64), dtype=np.float32)
        n_patch = 0
        tile_dict = {}
        print(x)

        # ===== Stage 1 =====
        print("\n开始Stage 1处理...")
        for i in range(0, h - 36, crop_size_h):
            tile_dict[i] = {}
            for j in range(0, w - 36, crop_size_w):
                x_crop = x_padded[:, i:i + crop_size_h + 36, j:j + crop_size_w + 36, :]

                outputs1 = self.run_tflite_model("stage_1", {"input": x_crop})
                tmp0 = outputs1["tmp0"]
                x_crop_out = outputs1["x_crop"]
                tmp_se_mean = outputs1["tmp_se_mean"]
                se_mean0 += tmp_se_mean
                n_patch += 1
                tile_dict[i][j] = (tmp0, x_crop_out)


        se_mean0 /= n_patch
        print(f"Stage 1完成，处理了 {n_patch} 个图块")

        # ===== Stage 2 =====
        print("\n开始Stage 2处理...")
        se_mean1 = np.zeros((n, 1, 1, 128), dtype=np.float32)
        for i in range(0, h - 36, crop_size_h):
            for j in range(0, w - 36, crop_size_w):
                tmp0, x_crop = tile_dict[i][j]
                outputs2 = self.run_tflite_model("stage_2", {"tmp0": tmp0, "x_crop": x_crop, "se_mean0": se_mean0})
                opt_unet1 = outputs2["opt_unet1"]
                tmp_x1 = outputs2["tmp_x1"]
                tmp_x2 = outputs2["tmp_x2"]
                tmp_se_mean = outputs2["tmp_se_mean"]
                se_mean1 += tmp_se_mean
                tile_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2)

        se_mean1 /= n_patch
        print(f"Stage 2完成")

        # ===== Stage 3 =====
        print("\n开始Stage 3处理...")
        se_mean0_stage3 = np.zeros((n, 1, 1, 128), dtype=np.float32)
        for i in range(0, h - 36, crop_size_h):
            for j in range(0, w - 36, crop_size_w):
                opt_unet1, tmp_x1, tmp_x2 = tile_dict[i][j]
                outputs3 = self.run_tflite_model("stage_3", {"tmp_x2": tmp_x2, "se_mean1": se_mean1})
                tmp_x2_out = outputs3["tmp_x2"]
                tmp_x3 = outputs3["tmp_x3"]
                tmp_se_mean = outputs3["tmp_se_mean"]
                se_mean0_stage3 += tmp_se_mean
                tile_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2_out, tmp_x3)

        se_mean0_stage3 /= n_patch
        print(f"Stage 3完成")

        # ===== Stage 4 =====
        print("\n开始Stage 4处理...")
        se_mean1_stage4 = np.zeros((n, 1, 1, 64), dtype=np.float32)
        for i in range(0, h - 36, crop_size_h):
            for j in range(0, w - 36, crop_size_w):
                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tile_dict[i][j]
                outputs4 = self.run_tflite_model("stage_4",
                                                 {"tmp_x2": tmp_x2, "tmp_x3": tmp_x3, "se_mean0": se_mean0_stage3})
                tmp_x4 = outputs4["tmp_x4"]
                tmp_se_mean = outputs4["tmp_se_mean"]
                se_mean1_stage4 += tmp_se_mean
                tile_dict[i][j] = (opt_unet1, tmp_x1, tmp_x4)

        se_mean1_stage4 /= n_patch
        print(f"Stage 4完成")

        # ===== Stage 5 =====
        print("\n开始Stage 5处理...")
        res = []
        for i in range(0, h - 36, crop_size_h):
            temp = []
            for j in range(0, w - 36, crop_size_w):
                x_input, tmp_x1, tmp_x4 = tile_dict[i][j]
                outputs5 = self.run_tflite_model("stage_5",
                                                 {"serving_default_x:0": x_input,
                                                  "serving_default_tmp_x1:0": tmp_x1,
                                                  "serving_default_tmp_x4:0": tmp_x4,
                                                  "serving_default_se_mean1:0": se_mean1_stage4})
                x_out = outputs5["x_out"]
                if i ==0 and j ==0:
                    print(f"输出：{x_out}")


                # 后处理
                if self.pro:
                    x_processed = np.clip(np.round((x_out - 0.15) * (255 / 0.7)), 0, 255).astype(np.uint8)
                else:
                    x_processed = np.clip(np.round(x_out * 255), 0, 255).astype(np.uint8)
                temp.append(x_processed)
            temp_concat = np.concatenate(temp, axis=2)
            res.append(temp_concat)

        res_concat = np.concatenate(res, axis=1)
        print(f"Stage 5完成，总共处理了 {len(res)} 行图块")

        # 移除处理过程中添加的额外padding
        target_h = original_h * 2
        target_w = original_w * 2

        # 移除边界padding
        res_concat = res_concat[:, :h0 * 2, :w0 * 2, :]

        # 移除在prepare_input中为了对齐512倍数添加的padding
        res_concat = res_concat[:, :target_h, :target_w, :]

        print(f"\n处理完成，最终输出形状: {res_concat.shape}")
        return res_concat

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process_image(image)


# 使用示例
if __name__ == "__main__":
    import cv2
    from PIL import Image
    import time


    def load_image(image_path):
        """加载图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图片: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def save_image(image_array: np.ndarray, save_path: str):
        """保存图片"""
        if len(image_array.shape) == 4:
            image_array = image_array[0]

        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        img = Image.fromarray(image_array, 'RGB')
        img.save(save_path)
        print(f"图片已保存到: {save_path}")


    def debug_image_info(image_array: np.ndarray, name: str = "Image"):
        """调试图片信息"""
        print(f"\n{name} 信息:")
        print(f"  形状: {image_array.shape}")
        print(f"  数据类型: {image_array.dtype}")
        if image_array.dtype in [np.float32, np.float64]:
            print(f"  数值范围: [{image_array.min():.6f}, {image_array.max():.6f}]")
            print(f"  均值: {image_array.mean():.6f}")
        else:
            print(f"  数值范围: [{image_array.min()}, {image_array.max()}]")
            print(f"  均值: {image_array.mean():.2f}")


    # 初始化模型
    print("正在初始化TFLite模型...")
    start_time = time.time()
    upscaler = UpCunet2x_TFLite(model_dir=".", half=False, pro=True, alpha=0.7)
    print(f"模型加载耗时: {time.time() - start_time:.2f}秒")

    # 加载测试图片
    test_image_path = "/Volumes/Home/oysterqaq/Desktop/1.jpg"
    print(f"\n加载图片: {test_image_path}")

    if not os.path.exists(test_image_path):
        # 如果测试图片不存在，创建一个测试图片
        print("测试图片不存在，创建测试图片...")
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite("test_input.png", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    else:
        test_image = load_image(test_image_path)

    debug_image_info(test_image, "原始图片")

    # 计算期望的输出尺寸
    original_h, original_w = test_image.shape[:2]
    expected_h, expected_w = original_h * 2, original_w * 2
    print(f"期望输出尺寸: {expected_h} x {expected_w}")

    # 运行超分辨率
    print("\n开始处理图片...")
    process_start = time.time()
    result = upscaler(test_image)
    process_time = time.time() - process_start
    print(f"处理耗时: {process_time:.2f}秒")

    # 检查输出尺寸
    debug_image_info(result, "处理后图片")
    print(f"实际输出尺寸: {result.shape[1]} x {result.shape[2]}")

    if result.shape[1] == expected_h and result.shape[2] == expected_w:
        print("✓ 输出尺寸与期望一致")
    else:
        print(f"✗ 输出尺寸不匹配! 期望: {expected_h}x{expected_w}, 实际: {result.shape[1]}x{result.shape[2]}")

    # 保存结果
    save_image(result, "output_tflite_shape_matched.png")