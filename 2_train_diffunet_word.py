"""
WORD数据集DiffUNet训练脚本
- 单模态CT输入 (1通道)
- 17类分割输出 (16器官 + 背景)
"""

import numpy as np
from light_training.dataloading.dataset import get_kfold_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
import os

def func(m, epochs):
    """不确定性损失的权重函数"""
    return np.exp(-10 * (1 - m / epochs) ** 2)

# 数据路径
data_dir = "/home/cjh/data/WORD-V0.1.0/fullres/train"
val_dir = "/home/cjh/data/WORD-V0.1.0/fullres/val"

# 训练配置
logdir = "./logs/diffunet_word"
env = "DDP"  # 或 "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 10
batch_size = 2
val_every = 10
num_gpus = 2
device = "cuda:1"
patch_size = [128, 128, 128]
augmentation = True

# WORD数据集配置
num_classes = 17  # 16个器官 + 背景
in_channels = 1   # 单模态CT

class WORDTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost',
                 master_port=17750, training_script="train.py",
                 use_pr25_boundary=True):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)

        # 滑动窗口推理
        self.window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=2,
            overlap=0.5
        )
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        self.use_pr25_boundary = use_pr25_boundary

        # 加载DiffUNet模型
        from diffunet.diffunet_model import DiffUNet
        self.model = DiffUNet(in_channels=in_channels, out_channels=num_classes,
                              use_pr25_boundary=use_pr25_boundary)

        self.best_mean_dice = 0.0

        # 优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-2,
            weight_decay=3e-5,
            momentum=0.99,
            nesterov=True
        )
        self.scheduler_type = "poly"

        # 损失函数
        self.loss_func = nn.CrossEntropyLoss()
        self.lambda_pr25 = 1.0

    def training_step(self, batch):
        """训练步骤"""
        image, label = self.get_input(batch)

        # 前向传播
        pred, pred_edge_or_boundary, uncertainty = self.model(image, label)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        # 计算损失
        loss_main = self.loss_func(pred, label)
        scale = func(self.epoch, max_epoch)

        if self.use_pr25_boundary:
            from diffunet.boundary_utils import compute_boundary_gt
            boundary_gt = compute_boundary_gt(label, num_classes=num_classes)
            loss_boundary = F.mse_loss(torch.sigmoid(pred_edge_or_boundary), boundary_gt)
            loss = loss_main.mean() + self.lambda_pr25 * loss_boundary.mean() + (loss_main.detach() * uncertainty).mean() * scale
            self.log("training_loss_boundary", loss_boundary.mean(), step=self.global_step)
        else:
            loss_edge = self.loss_func(pred_edge_or_boundary, label)
            loss = loss_main.mean() + loss_edge.mean() + (loss_main.detach() * uncertainty).mean() * scale
            self.log("training_loss_edge", loss_edge.mean(), step=self.global_step)

        # 记录日志
        self.log("training_loss", loss.mean(), step=self.global_step)
        self.log("uncertainty_scale", scale, step=self.global_step)

        return loss

    def get_input(self, batch):
        """获取输入数据"""
        image = batch["data"]      # [B, 1, H, W, D] - 单通道CT
        label = batch["seg"]       # [B, 1, H, W, D]
        label = label[:, 0].long()  # [B, H, W, D]
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        """计算评估指标"""
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        """验证步骤"""
        image, label = self.get_input(batch)

        # 使用DDIM采样进行推理
        output = self.model(image, ddim=True)

        # 获取预测类别
        output = output.argmax(dim=1)  # [B, H, W, D]
        output = output.cpu().numpy()
        target = label.cpu().numpy()

        # 计算每个类别的Dice
        dices = []
        for i in range(1, num_classes):  # 跳过背景(0)
            pred_c = output == i
            target_c = target == i
            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)

        return dices

    def validation_end(self, val_outputs):
        """验证结束处理"""
        dices = val_outputs  # [num_classes-1, num_samples]

        # 计算每个类别的平均Dice
        dices_mean = []
        for i in range(num_classes - 1):
            class_dice = np.mean([d[i] for d in dices])
            dices_mean.append(class_dice)
            self.log(f"class_{i+1}_dice", class_dice, step=self.epoch)

        # 总体平均Dice
        mean_dice = np.mean(dices_mean)
        self.log("mean_dice", mean_dice, step=self.epoch)

        # 保存最佳模型
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )
            print(f"New best model saved! Mean Dice: {mean_dice:.4f}")

        # 保存最终模型
        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )

        print(f"Epoch {self.epoch}: Mean Dice = {mean_dice:.4f}")
        print(f"Per-class Dice: {dices_mean}")


def train_with_kfold(fold=0):
    """使用K折交叉验证训练"""
    trainer = WORDTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17755,
        training_script=__file__
    )

    # 使用5折交叉验证
    train_ds, val_ds, test_ds = get_kfold_loader(
        data_dir=data_dir,
        fold=fold,
        test_dir=None
    )

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


def train_with_split():
    """使用训练集+验证集分开的方式训练"""
    from light_training.dataloading.dataset import get_train_val_test_loader_seperate

    trainer = WORDTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17755,
        training_script=__file__
    )

    train_ds, val_ds, test_ds = get_train_val_test_loader_seperate(
        train_dir=data_dir,
        val_dir=val_dir,
        test_dir=None
    )

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


def train_all_data():
    """使用全部数据进行训练（80训练+20验证）"""
    from light_training.dataloading.dataset import get_all_training_loader

    trainer = WORDTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17755,
        training_script=__file__
    )

    train_ds, val_ds, test_ds = get_all_training_loader(
        data_dir=data_dir,
        fold=0,
        test_dir=None
    )

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


if __name__ == "__main__":
    # 选择训练方式：

    # 方式1: K折交叉验证 (推荐用于小规模数据集)
    # train_with_kfold(fold=0)

    # 方式2: 训练集/验证集分开 (需要预先处理验证集)
    train_with_split()

    # 方式3: 使用全部训练数据
    # train_all_data()
