import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os

from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from PIL import ImageFilter
from PIL import Image
import timm
from vision_transformer import vit_small, DINOHead

from utils import *
from dataset import DataAugmentationDINO


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)



class TrainConfig:
    def __init__(self):
        self.output_dim = 768
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.max_epochs = 100
        self.teacher_temp = 0.04 # softmax 温度系数更小，使得更突出
        self.student_temp = 0.1  
        self.center_momentum = 0.9
        self.ema_decay = 0.9995
        self.dataset_path = "/home/lixumin/project/local_dinov2/local_match/data/"  # 替换为你自己的数据集路径
        self.num_workers = self.batch_size+1
        self.seed = 42

        self.global_crops_scale = (0.4, 1)
        self.local_crops_scale = (0.2, 0.4)
        self.local_crops_number = 8


class DINOLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(DINOLightningModule, self).__init__()
        # 使用 torchvision 提供的预训练 vit_b_16
        self.config = config

        self.student = vit_small()
        self.teacher = vit_small()

        # 初始化教师模型权重
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        embed_dim = self.student.embed_dim
        self.student = MultiCropWrapper(self.student, DINOHead(
            embed_dim,
            self.config.output_dim,
            use_bn=False,
            norm_last_layer=True,
        ))
        self.teacher = MultiCropWrapper(
            self.teacher,
            DINOHead(embed_dim, self.config.output_dim, False),
        )

        # 初始化中心向量
        self.register_buffer("center", torch.zeros(1, self.config.output_dim))


    # 前向传播
    def forward(self, x):
        return self.student(x)

    # loss
    def dino_loss(self, student_output, teacher_output):
        student_out = student_output / self.config.student_temp
        student_out = student_out - self.center.detach()
        student_out = F.log_softmax(student_out, dim=-1)

        teacher_out = F.softmax((teacher_output - self.center.detach()) / self.config.teacher_temp, dim=-1)
        loss = F.kl_div(student_out, teacher_out, reduction='batchmean', log_target=False)
        return loss
    
    def training_step(self, batch, batch_idx):
        images, _ = batch  # 假设你的数据集返回 (global_views, local_views)
        student_outputs = self.student(images)
        teacher_output = self.teacher(images[:2]) # 使用一个全局视图作为教师的输入

        loss = 0
        n_global = len(images[:2])
        n_local = len(images[2:])
        n_views = n_global + n_local
        for i, output in enumerate(student_outputs):
            for j in range(n_global):
                loss += self.dino_loss(output, teacher_output)
        loss /= (n_views * n_global)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_teacher()
        images, _ = batch
        self._update_center(self.teacher(images[:2])) # 使用训练批次的全局视图更新中心

    def _update_teacher(self):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(self.config.ema_decay).add_((1 - self.config.ema_decay) * param_s.detach().data)

    def _update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output.detach(), dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.config.center_momentum + batch_center * (1 - self.config.center_momentum)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.student.parameters(), lr=self.config.learning_rate, weight_decay=0.05)
        return optimizer

    # 验证过程
    def validation_step(self, batch, batch_idx):
        pass



class DataPLModule(pl.LightningDataModule):
    def __init__(self, dataset_name_or_path, train_batch_size, num_workers, seed, global_crops_scale, local_crops_scale, local_crops_number):
        super().__init__()
        self.dataset_name_or_path = dataset_name_or_path
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size  
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.train_dataset = None 
        self.val_dataset = None 
        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def setup(self, stage=None):
        transform = DataAugmentationDINO(
            self.global_crops_scale, self.local_crops_scale, self.local_crops_number
        )
        self.train_dataset = datasets.ImageFolder(self.dataset_name_or_path, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            # sampler=sampler,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.train_batch_size,
    #         pin_memory=True,
    #         shuffle=False,
    #     )

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def test_vit():
    from torchvision import transforms
    from PIL import Image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open('default.jpg')
    img_tensor = preprocess(img).unsqueeze(0) # 添加批次维度 (batch size = 1)
    model = DINOLightningModule(config=TrainConfig())
    with torch.no_grad(): # 在推理阶段禁用梯度计算
        output = model.model(img_tensor)
    print(output.shape) # [1, 1000]

# 设置训练过程
if __name__ == '__main__':
    # 定义数据相关的配置
    
    train_config = TrainConfig()

    # 创建数据模块
    data_module = DataPLModule(
        dataset_name_or_path=train_config.dataset_path,
        train_batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        global_crops_scale = train_config.global_crops_scale,
        local_crops_scale = train_config.local_crops_scale,
        local_crops_number = train_config.local_crops_number
    )

    # 定义 ModelCheckpoint 回调，每个 epoch 保存一次模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # 模型保存的目录
        filename="{epoch:02d}-{train_acc:.2f}",  # 保存的文件名格式
        save_top_k=-1,  # 保存所有 epoch 的模型
        verbose=True,
        mode="min",  # 监控的 metric 越低越好
        save_weights_only=False,  # 保存整个模型而不仅仅是权重
        every_n_epochs=2  # 每个 epoch 保存一次
    )

    # 定义训练器
    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        devices=[1],
        callbacks=[checkpoint_callback],  # 添加 checkpoint 回调
        
    )
    
    model = DINOLightningModule(config=train_config)
    # 开始训练
    trainer.fit(model, data_module, 
        # ckpt_path="checkpoints/epoch=199-val_loss=0.02.ckpt"
    )