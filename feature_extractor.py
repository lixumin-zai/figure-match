from torchvision.models import (resnet18, resnet34, resnet50)
import torch
from PIL import Image
from transforms import test_transforms, my_test_transform
import torch.nn as nn
import io
import pickle
import os
import tqdm
import cv2

class FeatureExtractor:
    def __init__(self, 
        checkpoint_path="epoch=199-train_acc=1.00.ckpt", 
        num_classes=320
    ):
        # 初始化 ResNet18 模型
        self.model = resnet34()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'),weights_only=True)
        state_dict = checkpoint['state_dict']
        # 去掉 'model.' 前缀
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # 如果你不需要再进行训练，切换模型为评估模式
        self.device = torch.device("cuda")
        self.model.load_state_dict(new_state_dict)
        self.model.eval().to(self.device)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval().to(self.device)

    def preprocess_transform(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = my_test_transform(image)
        cv2.imwrite("show.png", image)
        input()
        image = test_transforms(Image.fromarray(image)).to(self.device)
        image = image.unsqueeze(0)
        return image

    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = test_transforms(image).to(self.device)
        image = image.unsqueeze(0)
        return image

    def extract_features(self, image):
        with torch.no_grad():
            features = self.feature_extractor(image).squeeze().cpu().detach().numpy()
        return features

    def __call__(self, image_bytes):
        # self.model.eval().cuda()
        image = self.preprocess(image_bytes)
        output = self.model(image)
        index = torch.argmax(output, dim=1) # 类别
        return index

def save_features_from_folder(folder_path, feature_extractor, output_file):
    feature_list = []  # 用于存储每个图像的文件名和特征

    # 遍历文件夹中的所有图像文件
    for filename in tqdm.tqdm(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取图像并提取特征
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            # if int(filename.split(".")[0]) < 276:
            #     image = feature_extractor.preprocess_transform(image_bytes)
            # else:
            #     continue
            image = feature_extractor.preprocess(image_bytes)

            features = feature_extractor.extract_features(image)
            
            # 存储文件名和对应的特征
            feature_data = {
                'filename': filename,
                'features': features
            }
            feature_list.append(feature_data)
    
    # 保存特征列表到输出文件 (使用 pickle)
    with open(output_file, 'wb') as f:
        pickle.dump(feature_list, f)

    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    # 实例化推理类，传入检查点路径
    feature_extractor = FeatureExtractor()

    #with open("/home/lixumin/project/local_match/project/match_model/5.png", "rb") as f:
    #    image_bytes = f.read()

    # predicted_labels = feature_extractor(image_bytes)
    # print(predicted_labels)

    # 特征提取
    # 保存图像特征到文件中
    folder_path = "./database/"
    output_file = "image_features.pkl"
    
    save_features_from_folder(folder_path, feature_extractor, output_file)
