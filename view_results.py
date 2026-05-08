import numpy as np

# 加载结果文件
result = np.load('./prediction_results/result_metrics/diffunet.npy')

print("=" * 50)
print("BraTS 2023 测试结果")
print("=" * 50)
print(f"数据形状: {result.shape}")
print(f"  - 样本数: {result.shape[0]}")
print(f"  - 类别数: {result.shape[1]} (TC, WT, ET)")
print(f"  - 指标数: {result.shape[2]} (Dice, HD95)")
print()

# 计算均值和标准差
mean = result.mean(axis=0)
std = result.std(axis=0)

print("-" * 50)
print("各类别指标均值 (Mean):")
print("-" * 50)
print(f"{'类别':<10} {'Dice':<15} {'HD95':<15}")
print(f"{'TC':<10} {mean[0,0]:<15.4f} {mean[0,1]:<15.4f}")
print(f"{'WT':<10} {mean[1,0]:<15.4f} {mean[1,1]:<15.4f}")
print(f"{'ET':<10} {mean[2,0]:<15.4f} {mean[2,1]:<15.4f}")
print()

print("-" * 50)
print("各类别指标标准差 (Std):")
print("-" * 50)
print(f"{'类别':<10} {'Dice':<15} {'HD95':<15}")
print(f"{'TC':<10} {std[0,0]:<15.4f} {std[0,1]:<15.4f}")
print(f"{'WT':<10} {std[1,0]:<15.4f} {std[1,1]:<15.4f}")
print(f"{'ET':<10} {std[2,0]:<15.4f} {std[2,1]:<15.4f}")
print()

print("-" * 50)
print("总体平均 Dice:")
print("-" * 50)
mean_dice = mean[:, 0].mean()
print(f"Mean Dice (TC+WT+ET)/3 = {mean_dice:.4f}")
print("=" * 50)
