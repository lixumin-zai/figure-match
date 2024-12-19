# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:44:13
# @Author  :   lixumin1030@gmail.com
# @FileName:   data.py

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import Any, List, Optional, Union
from torchvision import transforms
from transforms import train_transforms, my_transform  # 假设你定义了训练的增强操作在transform.py中
import numpy as np

class MyDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_path: str = "",
    ):
        """
        初始化数据集类
        Args:
            dataset_name_or_path (str): 数据集路径
            transform (Optional[Any]): 数据增强变换
            label_map (Optional[dict]): 标签映射，如果需要可以提供
        """
        super().__init__()

        self.dataset_name_or_path = dataset_name_or_path
        self.transform = train_transforms
        
        # 检查路径是否存在
        if not os.path.exists(dataset_name_or_path):
            raise FileNotFoundError(f"路径 {dataset_name_or_path} 不存在！")
        
        # 获取所有图像文件和对应的标签
        self.image, self.labels = self._load_images_and_labels()
        self.dataset_length = len(self.image)
        print(f"数据集长度: {self.dataset_length}\n数据路径: {dataset_name_or_path}")

    def _load_images_and_labels(self):
        """
        从文件夹中加载图像路径和标签
        Returns:
            image_paths (List[str]): 图像文件路径列表
            labels (List[int]): 对应的标签列表
        """
        images = []
        labels = []

        for label_name in os.listdir(self.dataset_name_or_path):
            label_dir = os.path.join(self.dataset_name_or_path, label_name)
            if label_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
                images.append(Image.open(label_dir).convert("RGB"))
                labels.append(int(label_name.split(".")[0])-1)

        return images, labels

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx):
        """
        获取数据
        Args:
            idx: 索引
        Returns:
            image: 增强后的图像
            label: 标签
        """
        image = self.image[idx]
        label = self.labels[idx]
        # 应用数据增强变换
        if self.transform:
            image = my_transform(image)
            
            image = self.transform(Image.fromarray(image))
        
        return image, label


# 测试代码
if __name__ == "__main__":
    # 定义图像增强变换
    from torchvision.models import resnet18
    # 加载数据集
    dataset_path = "/home/lixumin/project/local_match/figure-match/database"  # 替换为你自己的数据集路径
    dataset = MyDataset(dataset_name_or_path=dataset_path)

        # 加载模型
    model = resnet18(pretrained=True)
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    from torchvision.utils import save_image
    # # 遍历数据加载器
    for images, labels in data_loader:
        # image = images.squeeze(0)
        # print(image.shape)
        # save_image(image, "show.png")
        # input()
        print(images.shape)
        output = model(images)
        print(output.shape)
