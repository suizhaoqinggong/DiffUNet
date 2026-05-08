"""
WORD数据集测试集指标计算脚本
- 计算16个器官的Dice和HD95指标
- GT从原始nii.gz文件中读取 (labelsVal)
- Pred从预测结果nii.gz文件中读取
- 两者均在原始图像空间
"""

from monai.utils import set_determinism
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm
from glob import glob

set_determinism(123)

parser = argparse.ArgumentParser()
parser.add_argument("--pred_name", required=True, type=str, help="预测结果目录名，如 word")
parser.add_argument("--gt_dir", type=str,
                    default="/home/cjh/data/WORD-V0.1.0/labelsVal",
                    help="GT标签目录 (原始nii.gz)")

results_root = "prediction_results"
args = parser.parse_args()

pred_name = args.pred_name
gt_dir = args.gt_dir

# WORD数据集配置
num_classes = 17  # 16个器官 + 背景
num_organs = 16

organ_names = {
    1: "liver", 2: "spleen", 3: "left_kidney", 4: "right_kidney",
    5: "stomach", 6: "gallbladder", 7: "esophagus", 8: "pancreas",
    9: "duodenum", 10: "colon", 11: "intestine", 12: "adrenal",
    13: "rectum", 14: "bladder", 15: "Head_of_femur_L", 16: "Head_of_femur_R"
}


def crop_to_bbox(mask_a, mask_b, margin=10):
    """裁剪到两个mask的联合bounding box，加速HD95计算"""
    combined = mask_a | mask_b
    if combined.sum() == 0:
        return mask_a, mask_b
    coords = np.argwhere(combined)
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, np.array(mask_a.shape))
    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    return mask_a[slices], mask_b[slices]


def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        gt_crop, pred_crop = crop_to_bbox(gt, pred)
        hd95 = metric.binary.hd95(pred_crop, gt_crop, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    elif gt.sum() == 0 and pred.sum() == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 50])


def each_cases_metric(gt, pred, voxel_spacing):
    """计算每个器官的指标 (跳过背景)"""
    class_wise_metric = np.zeros((num_organs, 2))
    for cls in range(1, num_classes):
        pred_c = (pred == cls)
        gt_c = (gt == cls)
        class_wise_metric[cls - 1, ...] = cal_metric(gt_c, pred_c, voxel_spacing)
    return class_wise_metric


if __name__ == "__main__":
    # 获取预测文件列表
    pred_dir = f"./{results_root}/{pred_name}"
    pred_files = sorted(glob(f"{pred_dir}/*.nii.gz"))
    num_cases = len(pred_files)
    print(f"Number of cases: {num_cases}")

    if num_cases == 0:
        print(f"No predictions found in {pred_dir}")
        exit(1)

    all_results = np.zeros((num_cases, num_organs, 2))  # [cases, 16 organs, (dice, hd95)]

    ind = 0
    for pred_path in tqdm(pred_files, total=num_cases):
        case_name = os.path.basename(pred_path).replace(".nii.gz", "")

        # 加载GT (原始nii.gz)
        gt_path = f"{gt_dir}/{case_name}.nii.gz"
        if not os.path.exists(gt_path):
            print(f"GT not found: {gt_path}")
            continue
        gt_itk = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)

        # 加载预测结果 (原始空间)
        pred_itk = sitk.ReadImage(pred_path)
        pred_array = sitk.GetArrayFromImage(pred_itk).astype(np.int32)

        if gt_array.shape != pred_array.shape:
            print(f"Shape mismatch: {case_name}, gt={gt_array.shape}, pred={pred_array.shape}")
            continue

        # 使用实际的voxel spacing (SimpleITK: x,y,z → numpy: z,y,x)
        voxel_spacing = list(reversed(gt_itk.GetSpacing()))

        m = each_cases_metric(gt_array, pred_array, voxel_spacing)
        all_results[ind, ...] = m
        ind += 1

    all_results = all_results[:ind]

    # 保存结果
    os.makedirs(f"./{results_root}/result_metrics/", exist_ok=True)
    save_path = f"./{results_root}/result_metrics/{pred_name}.npy"
    np.save(save_path, all_results)

    # 打印结果
    result = np.load(save_path)
    print(f"\nResult shape: {result.shape}")  # [cases, 16, 2]

    print(f"\n{'Organ':<20} {'Dice':>8} {'HD95':>8}")
    print("-" * 38)
    for i in range(num_organs):
        cls = i + 1
        dice_mean = result[:, i, 0].mean()
        hd95_mean = result[:, i, 1].mean()
        print(f"{organ_names[cls]:<20} {dice_mean:>8.4f} {hd95_mean:>8.2f}")

    print("-" * 38)
    overall_dice = result[:, :, 0].mean()
    overall_hd95 = result[:, :, 1].mean()
    print(f"{'Mean':<20} {overall_dice:>8.4f} {overall_hd95:>8.2f}")
    print(f"\nOverall Dice std: {result[:, :, 0].std():.4f}")
