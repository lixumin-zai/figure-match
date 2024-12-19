import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchvision.models import (resnet18, resnet34)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from data import MyDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random

# 定义基于 ResNet50 的 LightningModule
class ResNet18LightningModule(pl.LightningModule):
    def __init__(self, num_classes=320, learning_rate=1e-4):
        super(ResNet18LightningModule, self).__init__()
        # 使用 torchvision 提供的预训练 ResNet50
        self.model = resnet34(pretrained=True)
        # 替换最后一层全连接层以适应新的分类任务
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # 定义交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 设置学习率
        self.learning_rate = learning_rate
        # 定义一个准确率计算指标
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    # 前向传播
    def forward(self, x):
        return self.model(x)

    # 训练过程
    def training_step(self, batch, batch_idx):
        images, labels = batch
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        outputs = self(images)
        # predicted_labels = torch.argmax(outputs, dim=1)
        # print(predicted_labels, labels)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(outputs, labels)
        # 计算准确率
        acc = self.train_acc(outputs, labels)
        
        # 记录日志（实时显示）
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
    
        return loss

    
    # 验证过程
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    # 配置优化器
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DataPLModule(pl.LightningDataModule):
    def __init__(self, dataset_name_or_path, train_batch_size, val_batch_size, num_workers, seed):
        super().__init__()
        self.num_workers = num_workers
        self.dataset_name_or_path = dataset_name_or_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size      
        self.train_dataset = None 
        self.val_dataset = None    
        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def setup(self, stage=None):
        self.train_dataset = MyDataset(
                dataset_name_or_path=self.dataset_name_or_path,
            )
        self.val_dataset = MyDataset(
                dataset_name_or_path="/home/lixumin/project/local_match/project/match_model/val",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


# 设置训练过程
if __name__ == '__main__':
     # 定义数据相关的配置
    dataset_path = "/home/lixumin/project/local_match/figure-match/database"  # 修改为你的数据集路径
    batch_size = 16
    num_classes = 320  # 根据你的数据集定义
    learning_rate = 1e-4
    max_epochs = 500
    num_workers = 50
    seed = 42
    # 创建数据模块
    data_module = DataPLModule(
        dataset_name_or_path=dataset_path,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )

    # 初始化 ResNet18 模型
    model = ResNet18LightningModule(num_classes=num_classes, learning_rate=learning_rate)

    # 定义 ModelCheckpoint 回调，每个 epoch 保存一次模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # 模型保存的目录
        filename="{epoch:02d}-{train_acc:.2f}",  # 保存的文件名格式
        save_top_k=-1,  # 保存所有 epoch 的模型
        verbose=True,
        mode="min",  # 监控的 metric 越低越好
        save_weights_only=False,  # 保存整个模型而不仅仅是权重
        every_n_epochs=10  # 每个 epoch 保存一次
    )

    # 定义训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=[1],
        callbacks=[checkpoint_callback],  # 添加 checkpoint 回调
        
    )

    # 开始训练
    trainer.fit(model, data_module, 
        # ckpt_path="checkpoints/epoch=199-val_loss=0.02.ckpt"
    )