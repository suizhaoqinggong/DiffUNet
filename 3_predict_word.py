"""
WORD数据集推理脚本
- 使用滑动窗口 + 测试时增强(mirror)
- 还原到原始图像空间 (重采样 + 去裁剪)
- 保存预测结果为nii.gz格式
"""

import numpy as np
import torch
import os
from monai.inferers import SlidingWindowInferer
from diffunet.diffunet_model import DiffUNet
from light_training.prediction import Predictor
from glob import glob
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,
                    default="/home/cjh/data/WORD-V0.1.0/fullres/val",
                    help="待预测的数据目录")
parser.add_argument("--output_dir", type=str,
                    default="./prediction_results/word",
                    help="预测结果保存目录")
parser.add_argument("--model_path", type=str,
                    default="./logs/diffunet_word/model/best_model_*.pt",
                    help="模型路径(支持通配符)")
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--mirror", action="store_true", default=False,
                    help="是否使用测试时增强(mirror)")
parser.add_argument("--use_pr25_boundary", action="store_true", default=False,
                    help="是否使用PR25边界提取模块(需与训练时保持一致)")

args = parser.parse_args()

num_classes = 17
in_channels = 1
patch_size = [128, 128, 128]


def load_model(model_path_pattern, device):
    """加载训练好的模型"""
    model_files = glob(model_path_pattern)
    if not model_files:
        raise ValueError(f"No model found at {model_path_pattern}")

    model_file = sorted(model_files)[-1]
    print(f"Loading model from: {model_file}")

    model = DiffUNet(in_channels=in_channels, out_channels=num_classes,
                      use_pr25_boundary=args.use_pr25_boundary)
    sd = torch.load(model_file, map_location="cpu")
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_k = k[7:] if k.startswith("module.") else k
        new_sd[new_k] = v
    model.load_state_dict(new_sd)
    model.to(device)
    model.eval()

    return model


def predict_dataset():
    """对整个数据集进行预测"""
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model_path, args.device)

    window_infer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=2,
        overlap=0.5
    )

    predictor = Predictor(
        window_infer=window_infer,
        mirror_axes=[0, 1, 2] if args.mirror else None
    )

    npz_files = sorted(glob(f"{args.data_dir}/*.npz"))
    print(f"Found {len(npz_files)} samples")

    for npz_path in tqdm(npz_files, total=len(npz_files)):
        case_name = os.path.basename(npz_path).replace(".npz", "")

        # 加载数据
        data = np.load(npz_path)
        image = data["data"]  # [1, H, W, D]

        # 加载properties (包含裁剪和重采样信息)
        pkl_path = npz_path.replace(".npz", ".pkl")
        with open(pkl_path, "rb") as f:
            properties = pickle.load(f)

        # 转为tensor: [C, H, W, D] -> [1, C, H, W, D]
        image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(args.device)

        # ① 滑动窗口推理 (DDIM采样)
        with torch.no_grad():
            model_output = predictor.maybe_mirror_and_predict(
                image_tensor, model, device=args.device, ddim=True
            )
        # model_output: [1, 17, H', W', D']

        # ② 重采样回裁剪前的尺寸 (还原resampling)
        model_output = predictor.predict_raw_probability(model_output, properties=properties)
        # model_output: [17, H_crop, W_crop, D_crop]

        # ③ argmax得到类别预测
        model_output = model_output.argmax(dim=0)  # [H_crop, W_crop, D_crop]

        # ④ 还原到原始尺寸 (还原cropping)
        model_output = predictor.predict_noncrop_probability(model_output, properties)
        # model_output: [H_orig, W_orig, D_orig] numpy uint8

        # ⑤ 保存到原始空间
        raw_spacing = list(properties["spacing"])
        predictor.save_to_nii(model_output,
                              raw_spacing=raw_spacing,
                              case_name=case_name,
                              save_dir=args.output_dir)


if __name__ == "__main__":
    predict_dataset()
