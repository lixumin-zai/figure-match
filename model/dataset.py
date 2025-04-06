import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms
import random
from PIL import ImageFilter
from transforms import image_transforms


class MyDataset(Dataset):
    def __init__(self, dataset_name_or_path, global_crops_scale, local_crops_scale, local_crops_number):
        self.dataset = datasets.ImageFolder(dataset_name_or_path)
        self.dataset_length = len(self.dataset)
        print(f"{self.dataset_length}\n{dataset_name_or_path}")
        
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

        self.data_process = DataAugmentationDINO(self.global_crops_scale, self.local_crops_scale, self.local_crops_number)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx):
        image = self.dataset[idx]
        return self.data_process(image[0])


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            normalize,
        ])

    def __call__(self, image):
        image = Image.fromarray(image_transforms(image))
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

if __name__ == "__main__":


    images_path = "/home/lixumin/project/local_dinov2/local_match/data"
    dataset = MyDataset(images_path, (0.4, 1), (0.05, 0.4), 8)

    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    for i in data_loader:
        print("***", len(i[0]))
        input()
        # break
    # print(image[2].shape)