import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.evaluation.metric import dice
import os

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

data_dir = "/home/cjh/data/fullres/train"

fold = 0

logdir = "./logs/diffunet"

env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:1"
patch_size = [128, 128, 128]
# patch_size = [96, 96, 96]
augmentation = True 
    
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py", use_pr25_boundary=True):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        self.use_pr25_boundary = use_pr25_boundary

        from diffunet.diffunet_model import DiffUNet
        self.model = DiffUNet(4, 4, use_pr25_boundary=use_pr25_boundary)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"

        self.loss_func = nn.CrossEntropyLoss()
        self.lambda_pr25 = 1.0

      
    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()

    
    def training_step(self, batch):
        image, label = self.get_input(batch)

        pred, pred_edge_or_boundary, uncertainty = self.model(image, label)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        loss_main = self.loss_func(pred, label)
        scale = func(self.epoch, max_epoch)

        if self.use_pr25_boundary:
            from diffunet.boundary_utils import compute_boundary_gt
            boundary_gt = compute_boundary_gt(label, num_classes=4)
            loss_boundary = F.mse_loss(torch.sigmoid(pred_edge_or_boundary), boundary_gt)
            loss = loss_main.mean() + self.lambda_pr25 * loss_boundary.mean() + (loss_main.detach() * uncertainty).mean() * scale
            self.log("training_loss_boundary", loss_boundary.mean(), step=self.global_step)
        else:
            loss_edge = self.loss_func(pred_edge_or_boundary, label)
            loss = loss_main.mean() + loss_edge.mean() + (loss_main.detach() * uncertainty).mean() * scale
            self.log("training_loss_edge", loss_edge.mean(), step=self.global_step)

        self.log("training_loss", loss.mean(), step=self.global_step)

        return loss 

    # for image, label in data_loader:
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        # label = self.convert_labels(label)

        label = label[:, 0].long()

        return image, label 

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            # hd95 = metric.binary.hd95(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image, ddim=True)
        
        output = output.argmax(dim=1)
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 4
        for i in range(1, c):
            pred_c = output == i
            target_c = target == i

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        dices_mean = []
        c = 3
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)
        
        self.log("0", dices_mean[0], step=self.epoch)
        self.log("1", dices_mean[1], step=self.epoch)
        self.log("2", dices_mean[2], step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")



        print(f"mean_dice is {mean_dice}")

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17753,
                            training_script=__file__)

    import random
    random.seed(37)
    from light_training.dataloading.dataset import get_train_val_test_loader_from_train
    train_ds, test_ds, _ = get_train_val_test_loader_from_train(data_dir=data_dir, train_rate=0.8, val_rate=0.2, test_rate=0.0)

    trainer.train(train_dataset=train_ds, val_dataset=test_ds)
