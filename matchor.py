from feature_extractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from numpy.linalg import norm

class Matchor:
    def __init__(self, ):
        self.feature_extractor = FeatureExtractor()

        self.feature_data = self.get_feature_data("./image_features.pkl")
        with open(f"./default.jpg", "rb") as f:
            self.default_image_bytes = f.read()

    def get_feature_data(self, file_path):
        with open(file_path, 'rb') as f:
            feature_list = pickle.load(f)
        return feature_list

    def cosine_similarity_metric(self, feature1, feature2):
        cosine_similarity = np.dot(feature1, feature2) / (norm(feature1) * norm(feature2))
        # distance = 1 - cosine_similarity
        return cosine_similarity

    def match_top5(self, image_bytes):
        # 提取新图像的特征
        image = self.feature_extractor.preprocess(image_bytes)
        feature = self.feature_extractor.extract_features(image)

        # 用于保存所有图像的相似度和文件名
        similarity_list = []

        # 遍历所有保存的特征
        for item in self.feature_data:
            filename = item['filename']
            saved_features = item['features']

            # 计算与新图像特征的余弦相似度
            similarity = self.cosine_similarity_metric(feature, saved_features)

            # 将相似度和文件名保存到列表中
            similarity_list.append((similarity, filename))

        # 对相似度列表按相似度从高到低排序
        similarity_list.sort(key=lambda x: x[0], reverse=True)

        # 选择相似度最高的前5个结果
        top_5_similar = similarity_list[:5]
        return top_5_similar

    def __call__(self, image_bytes):
        top_5_similar = self.match_top5(image_bytes)

        if top_5_similar[0][0]< 0.8:
            return [self.default_image_bytes]
        
        images_bytes = []
        for similar_info in top_5_similar[:3]:
            print(similar_info)
            with open(f"database/{similar_info[1]}", "rb") as f:
                image_bytes = f.read()
                images_bytes.append(image_bytes)
        return images_bytes
        
        #with open(f"database/{top_5_similar[0][1]}", "rb") as f:
        #    image_bytes = f.read()
        #return image_bytes


if __name__ == "__main__":
    match = Matchor()
    with open("1.png", "rb") as f:
        image_bytes = f.read()
    print(match.match_top5(image_bytes))
    # ./database/74.jpg
