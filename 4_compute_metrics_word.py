"""
WORD数据集评估脚本
- 计算每个器官的Dice系数
- 计算平均Dice
- 生成详细的评估报告
"""

import numpy as np
import SimpleITK as sitk
import os
import glob
from light_training.evaluation.metric import dice

# 器官名称映射（与dataset.json一致）
ORGAN_NAMES = {
    0: "background",
    1: "liver",
    2: "spleen",
    3: "left_kidney",
    4: "right_kidney",
    5: "stomach",
    6: "gallbladder",
    7: "esophagus",
    8: "pancreas",
    9: "duodenum",
    10: "colon",
    11: "intestine",
    12: "adrenal",
    13: "rectum",
    14: "bladder",
    15: "Head_of_femur_L",
    16: "Head_of_femur_R"
}

# 路径配置
gt_dir = "/home/cjh/data/WORD-V0.1.0/labelsVal"  # 或 labelsTr
pred_dir = "./prediction_results/word"


def load_nii(path):
    """加载nii.gz文件"""
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def compute_metrics_for_case(gt_path, pred_path):
    """计算单个case的指标"""
    gt = load_nii(gt_path)
    pred = load_nii(pred_path)

    # 确保形状一致
    assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"

    results = {}
    for label_id in range(1, 17):  # 跳过背景，计算16个器官
        gt_binary = (gt == label_id).astype(np.uint8)
        pred_binary = (pred == label_id).astype(np.uint8)

        if gt_binary.sum() == 0 and pred_binary.sum() == 0:
            # 两者都为空，视为完美分割
            dice_score = 1.0
        elif gt_binary.sum() == 0 or pred_binary.sum() == 0:
            # 只有一个是空，视为完全失败
            dice_score = 0.0
        else:
            dice_score = dice(pred_binary, gt_binary)

        results[label_id] = dice_score

    return results


def evaluate_dataset():
    """评估整个数据集"""
    # 获取所有预测文件
    pred_files = glob.glob(f"{pred_dir}/*.nii.gz")

    all_results = {i: [] for i in range(1, 17)}  # 每个器官的Dice列表

    print("=" * 80)
    print("WORD Dataset Evaluation")
    print("=" * 80)

    for pred_path in pred_files:
        case_name = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, case_name)

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {case_name}")
            continue

        # 计算指标
        case_results = compute_metrics_for_case(gt_path, pred_path)

        for label_id, dice_score in case_results.items():
            all_results[label_id].append(dice_score)

    # 计算统计信息
    print("\nPer-organ Dice Scores:")
    print("-" * 80)

    mean_dices = []
    for label_id in range(1, 17):
        organ_name = ORGAN_NAMES[label_id]
        dices = all_results[label_id]

        if len(dices) > 0:
            mean_dice = np.mean(dices)
            std_dice = np.std(dices)
            mean_dices.append(mean_dice)

            print(f"{label_id:2d}. {organ_name:20s}: {mean_dice:.4f} ± {std_dice:.4f}")
        else:
            print(f"{label_id:2d}. {organ_name:20s}: No data")

    print("-" * 80)
    print(f"Mean Dice (all organs): {np.mean(mean_dices):.4f}")
    print("=" * 80)

    # 保存详细结果
    output_file = os.path.join(pred_dir, "evaluation_results.txt")
    with open(output_file, "w") as f:
        f.write("WORD Dataset Evaluation Results\n")
        f.write("=" * 80 + "\n\n")

        for label_id in range(1, 17):
            organ_name = ORGAN_NAMES[label_id]
            dices = all_results[label_id]

            if len(dices) > 0:
                mean_dice = np.mean(dices)
                std_dice = np.std(dices)
                f.write(f"{label_id:2d}. {organ_name:20s}: {mean_dice:.4f} ± {std_dice:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Mean Dice (all organs): {np.mean(mean_dices):.4f}\n")

    print(f"\nResults saved to: {output_file}")


def evaluate_single_case(case_name):
    """评估单个case"""
    pred_path = os.path.join(pred_dir, f"{case_name}.nii.gz")
    gt_path = os.path.join(gt_dir, f"{case_name}.nii.gz")

    if not os.path.exists(pred_path):
        print(f"Prediction not found: {pred_path}")
        return

    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        return

    results = compute_metrics_for_case(gt_path, pred_path)

    print(f"\nCase: {case_name}")
    print("-" * 40)

    for label_id, dice_score in results.items():
        organ_name = ORGAN_NAMES[label_id]
        print(f"{label_id:2d}. {organ_name:20s}: {dice_score:.4f}")

    mean_dice = np.mean(list(results.values()))
    print("-" * 40)
    print(f"Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    # 评估整个数据集
    evaluate_dataset()

    # 或者评估单个case
    # evaluate_single_case("word_0002")
