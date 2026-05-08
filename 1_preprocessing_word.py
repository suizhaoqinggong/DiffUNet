"""
WORD数据集预处理脚本
- 单模态CT输入
- 17类分割 (16个器官 + 背景)
- 使用CT标准化
"""

from light_training.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
import numpy as np
import json
import os

base_dir = "/home/cjh/data/WORD-V0.1.0/"
image_dir = "imagesTr"
label_dir = "labelsTr"

def plan():
    """数据分析阶段：收集统计信息以确定最优预处理参数"""
    preprocessor = DefaultPreprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        data_type="CT"
    )
    preprocessor.run_plan()

def process_train():
    """处理训练集"""
    # 读取数据分析结果
    analysis_path = "./data_analysis_result.txt"
    if not os.path.exists(analysis_path):
        print(f"请先运行 plan() 生成数据分析文件: {analysis_path}")
        return

    with open(analysis_path, "r") as f:
        content = json.loads(f.read().strip())

    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]

    # 从数据分析结果获取推荐的目标间距
    fullres_spacing = content["fullres spacing"]
    print(f"使用目标间距: {fullres_spacing}")

    preprocessor = DefaultPreprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        data_type="CT"
    )

    output_dir = "/home/cjh/data/WORD-V0.1.0/fullres/train/"

    # WORD有16个器官+背景=17类
    all_labels = list(range(17))  # 0-16

    preprocessor.run(
        output_spacing=fullres_spacing,
        output_dir=output_dir,
        all_labels=all_labels,
        foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel,
        num_processes=8
    )

def process_val():
    """处理验证集 (imagesVal, labelsVal)"""
    analysis_path = "./data_analysis_result.txt"
    if not os.path.exists(analysis_path):
        print(f"错误：未找到数据分析文件 {analysis_path}，请先运行 plan()")
        return

    with open(analysis_path, "r") as f:
        content = json.loads(f.read().strip())

    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]
    fullres_spacing = content["fullres spacing"]

    val_image_dir = "imagesVal"
    val_label_dir = "labelsVal"

    preprocessor = DefaultPreprocessor(
        base_dir=base_dir,
        image_dir=val_image_dir,
        label_dir=val_label_dir,
        data_type="CT"
    )

    output_dir = "/home/cjh/data/WORD-V0.1.0/fullres/val/"
    all_labels = list(range(17))

    preprocessor.run(
        output_spacing=fullres_spacing,
        output_dir=output_dir,
        all_labels=all_labels,
        foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel,
        num_processes=8
    )

def process_test():
    """处理测试集"""
    analysis_path = "./data_analysis_result.txt"
    if not os.path.exists(analysis_path):
        print(f"错误：未找到数据分析文件 {analysis_path}，请先运行 plan()")
        return

    with open(analysis_path, "r") as f:
        content = json.loads(f.read().strip())

    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]
    fullres_spacing = content["fullres spacing"]

    preprocessor = DefaultPreprocessor(
        base_dir=base_dir,
        image_dir="imagesTs",
        label_dir=None,  # 测试集无标签
        data_type="CT"
    )

    output_dir = "/home/cjh/data/WORD-V0.1.0/fullres/test/"
    all_labels = list(range(17))

    preprocessor.run(
        output_spacing=fullres_spacing,
        output_dir=output_dir,
        all_labels=all_labels,
        foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel,
        num_processes=8
    )

if __name__ == "__main__":
    import sys

    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == "plan":
            plan()
        elif step == "train":
            process_train()
        elif step == "val":
            process_val()
        elif step == "test":
            process_test()
        elif step == "all":
            print("=" * 80)
            print("Step 1/3: Running data analysis...")
            print("=" * 80)
            plan()

            print("\n" + "=" * 80)
            print("Step 2/3: Processing training set...")
            print("=" * 80)
            process_train()

            print("\n" + "=" * 80)
            print("Step 3/3: Processing validation set...")
            print("=" * 80)
            process_val()

            print("\n" + "=" * 80)
            print("All preprocessing steps completed!")
            print("=" * 80)
        else:
            print(f"Unknown step: {step}")
            print("Usage: python 1_preprocessing_word.py [plan|train|val|test|all]")
    else:
        # 默认执行所有步骤
        print("=" * 80)
        print("Step 1/3: Running data analysis...")
        print("=" * 80)
        plan()

        print("\n" + "=" * 80)
        print("Step 2/3: Processing training set...")
        print("=" * 80)
        process_train()

        print("\n" + "=" * 80)
        print("Step 3/3: Processing validation set...")
        print("=" * 80)
        process_val()

        print("\n" + "=" * 80)
        print("All preprocessing steps completed!")
        print("=" * 80)
