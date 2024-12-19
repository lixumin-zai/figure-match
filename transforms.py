
from torchvision import transforms


def add_padding(image):
    """对图像添加填充，使其变为正方形。"""
    width, height = image.size
    # 计算左右、上下的填充尺寸
    if width > height:
        padding = (0, (width - height) // 2, 0, (width - height + 1) // 2)  # 左、上、右、下
    else:
        padding = ((height - width) // 2, 0, (height - width + 1) // 2, 0)  # 左、上、右、下
    return transforms.functional.pad(image, padding, fill=0)  # 使用0进行填充


train_transforms = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),

    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义转换
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),            # 1. 转为灰度图（单通道）
    transforms.Lambda(add_padding),                         # 2. 添加 padding 使其成为正方形
    transforms.Resize((768, 768)),                          # 3. 缩放到 768x768
    transforms.Grayscale(num_output_channels=3),            # 4. 转回三通道
    transforms.ToTensor(),                                  # 5. 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # 6. 归一化
                         std=[0.229, 0.224, 0.225])
])


# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:43:39
# @Author  :   lixumin1030@gmail.com
# @FileName:   transform.py


import albumentations as alb
# from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageFilter
import random
import string
import os
import io
        # font_name = os.listdir("/home/lixumin/project/recognition_diagram/test_model/font")
        # font = ImageFont.truetype(f"/home/lixumin/project/recognition_diagram/test_model/font/{random.choice(font_name)}", font_size)
FONT_PATH = "/home/lixumin/project/recognition_diagram/test_model/font"

def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]
    return f

def Image2cv2(pil_image):
    """ 将 PIL 图像转换为 OpenCV 图像 """
    # 将 PIL 图像转换为 numpy 数组
    cv2_image = np.array(pil_image)
    # 转换颜色通道从 RGB 到 BGR
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def cv22Image(cv2_image):
    """ 将 OpenCV 图像转换为 PIL 图像 """
    # 转换颜色通道从 BGR 到 RGB
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # 将 numpy 数组转换为 PIL 图像
    pil_image = Image.fromarray(cv2_image)
    return pil_image

def bytes2Image(image_bytes):
    # 使用 io.BytesIO 将字节数据转为一个字节流对象
    image_stream = io.BytesIO(image_bytes)
    
    # 使用 PIL.Image.open 从字节流对象中打开图像
    image = Image.open(image_stream)
    
    return image

def cv2ImageToBytes(image, format='.jpg'):
    """
    将 OpenCV 图像转换为字节数据
    :param image: OpenCV 图像
    :param format: 图像格式（默认为 .jpg）
    :return: 图像的字节数据
    """
    # 使用 imencode 将图像编码为指定格式
    success, encoded_image = cv2.imencode(format, image)
    if not success:
        raise ValueError("Could not encode image")
    
    # 将编码后的图像转换为字节数据
    return encoded_image.tobytes()
################################################################################################

class ResizeIfNeeded(alb.ImageOnlyTransform):
    def __init__(self, max_size, min_size, always_apply=False, p=1.0):
        super(ResizeIfNeeded, self).__init__(always_apply, p)
        self.max_size = max_size
        self.min_size = min_size

    def apply(self, img, **params):
        # 获取图片的高度和宽度
        # img = simulate_color_jitter(img)
        height, width = img.shape[:2]
        # 获取最长边和最短边
        max_side = max(height, width)
        min_side = min(height, width)
        
        # 如果最长边超过 max_size，则等比例缩放
        if max_side > self.max_size:
            scale = self.max_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = new_height, new_width  # 更新高度和宽度

        # 如果最短边小于 min_size，则等比例缩放
        min_side = min(height, width)  # 更新后的最短边
        max_side = max(height, width)
        if max_side < self.min_size:
            scale = self.min_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img

################################################################################################

class ResizeIfMaxSideExceeds(alb.ImageOnlyTransform):
    def __init__(self, max_size, always_apply=False, p=1.0):
        super(ResizeIfMaxSideExceeds, self).__init__(always_apply, p)
        self.max_size = max_size

    def apply(self, img, **params):
        # 获取图片的高度和宽度
        height, width = img.shape[:2]
        # 获取最长边
        max_side = max(height, width)
        
        if max_side > self.max_size:
            # 计算缩放比例
            scale = self.max_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            # 等比例缩放图片
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    
################################################################################################

class ConditionalRandomScale(alb.ImageOnlyTransform):
    def __init__(self, scale_limit=(-0.5, 1), enlarge=False, always_apply=False, p=1.0):
        super(ConditionalRandomScale, self).__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.enlarge = enlarge

    def apply(self, img, **params):
        temperator = random.choice([cv2.INTER_LINEAR, cv2.INTER_LINEAR, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA])
        if self.enlarge:
            return alb.RandomScale(scale_limit=self.scale_limit, interpolation=temperator, p=1)(image=img)['image']
        else:
            pass

        if img.shape[0] >= 320 and img.shape[1] >= 320:
            return alb.RandomScale(scale_limit=self.scale_limit, interpolation=temperator, p=1)(image=img)['image']
        else:
            return img
        
################################################################################################

class Line_blur(alb.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Line_blur, self).__init__(always_apply, p)
        

    def apply(self, img, **params):
        img = cv22Image(img)
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 20))
        offset = (0, 0)
        img.paste(shadow, offset, shadow)
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        # 将字节流转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        # 使用cv2.imdecode将numpy数组转换为cv2图像
        cv2_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        return cv2_img

################################################################################################


def apply_watermark(src_image, text, text_size, rotation_angle):
    # Load the original image
    # Create a single watermark
    watermark = np.zeros((random.randint(60, 200), random.randint(100, 400), 3), dtype=np.uint8)*255
    r, g, b = random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)
    watermark = put_text_husky(watermark, text, (r, g, b), text_size, "Times New Roman")

    # Define horizontal and vertical repeat counts based on the size of the source image
    h_repeat = src_image.shape[1] // watermark.shape[1] + 1
    v_repeat = src_image.shape[0] // watermark.shape[0] + 1

    # Create tiled watermark
    tiled_watermark = np.tile(watermark, (v_repeat, h_repeat, 1))

    # Crop the tiled watermark to the size of the original image
    tiled_watermark = tiled_watermark[:src_image.shape[0], :src_image.shape[1]]
    # Rotate the watermark
    center = (tiled_watermark.shape[1] // 2, tiled_watermark.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_watermark = cv2.warpAffine(tiled_watermark, rotation_matrix, (tiled_watermark.shape[1], tiled_watermark.shape[0]))
    # src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2BGRA)
    
    # src_image = cv22Image(src_image)
    # rotated_watermark = cv22Image(rotated_watermark)

    image = cv2.addWeighted(src_image, 0.8, rotated_watermark, random.uniform(0.2, 0.5), 1)
    # image = np.where(image == (242, 242, 242), 255, image)
    return image

def put_text_husky(img, text, color, font_size, font_name, italic=False, underline=False):
    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Set font style
    font_style = ''
    if italic:
        font_style += 'I'
    if underline:
        font_style += 'U'
    
    # Load font or default
    try:
        # font = ImageFont.truetype(f'{font_name}{font_style}.ttf', font_size)
        font_name = os.listdir(FONT_PATH)
        font = ImageFont.truetype(f"{FONT_PATH}/{random.choice(font_name)}", font_size)

    except IOError:
        print(f"Font {font_name} with style {font_style} not found. Using default font.")
        font = ImageFont.load_default()

    # Calculate text bounding box for center alignment
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    orgX = (img.shape[1] - text_width) // 2
    orgY = (img.shape[0] - text_height) // 2

    # Draw text
    
    draw.text((orgX, orgY), text, font=font, fill=(int(color[0]), int(color[1]), int(color[2])))
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

import random

def generate_random_chinese_char():
    # 生成一个随机的汉字Unicode编码值
    unicode_val = random.randint(0x4E00, 0x9FFF)
    # 将Unicode编码值转换为对应的字符
    return chr(unicode_val)

def generate_balanced_watermark_text(length=15):
    # 定义可能的字符池
    chinese_chars = "".join([generate_random_chinese_char() for i in range(20)])
    english_chars = string.ascii_uppercase  # A-Z
    digits = string.digits  # 0-9
    
    # 确保每种类型的字符至少出现一次
    if length < 3:
        raise ValueError("Length must be at least 3 to include at least one of each character type.")
    
    # 生成包含至少一个中文、一个英文字母和一个数字的基础水印文本
    watermark_text = [
        random.choice(chinese_chars),
        random.choice(english_chars),
        random.choice(digits)
    ]
    
    # 填充剩余的字符
    all_chars = chinese_chars + english_chars + digits
    watermark_text += [random.choice(all_chars) for _ in range(length - 3)]
    
    # 混洗以增加随机性
    random.shuffle(watermark_text)
    
    # 将列表转换为字符串
    return ''.join(watermark_text)

class watermark(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        random_watermark_text = generate_balanced_watermark_text(random.randint(3, 7))  # 生成15个字符的水印文本
        # print(random_watermark)
        font_size = random.randint(25, 100)
        rotation_angle = random.randint(-50, 50)
        # Example usage:
        result_img = apply_watermark(img, random_watermark_text, font_size, rotation_angle)
        return result_img

################################################################################################
def generate_random_rgb():
    r = random.randint(0, 20)
    g = random.randint(0, 20)
    b = random.randint(0, 20)
    return (r, g, b)

def add_random_shadows(input_image):
    # 将OpenCV图像转换为PIL图像
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(input_image)

    width, height = pil_image.size

    # 创建阴影图层
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)

    # 随机生成阴影形状
    opacity = random.randint(50, 170) # 不透明度
    for _ in range(random.randint(5, 10)):  # 生成5-10个随机形状
        # 随机选择形状类型：圆形或椭圆形
        shape_type = random.choice(['ellipse', 'ellipse', 'ellipse', 'ellipse', 'rectangle'])
        # 随机生成位置和大小
        x0 = random.randint(0, int(width / 1.3))
        y0 = random.randint(0, int(height / 1.3))
        x1 = random.randint(int(1.2*x0), width)
        y1 = random.randint(int(1.2*y0), height)
        # 随机生成透明度
        # 绘制形状
        fill_data = generate_random_rgb() + tuple([opacity])
        if shape_type == 'ellipse':
            shadow_draw.ellipse([x0, y0, x1, y1], fill=fill_data)
        else:
            shadow_draw.rectangle([x0, y0, x1, y1], fill=fill_data)

    # 使用高斯模糊使阴影更自然
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))

    # 将阴影合并到原图上
    pil_image.paste(shadow, mask=shadow)

    # 将PIL图像转换回OpenCV图像
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result_image

class add_shadown(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        new_img = add_random_shadows(img)  # 生成15个字符的水印文本
        return new_img
    ################################################################################################

class Crop_image(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        x_min, y_min, img = crop_image(img)  # 生成15个字符的水印文本
        return img

def crop_image(img):
    img_data = img.copy()
    img_data = img_data[20:img_data.shape[1]-20, 20:img_data.shape[0]-20]
    nnz_inds = np.where(img_data <= (210, 210, 210))
    y_min = max(np.min(nnz_inds[0]) - np.random.randint(0, 50), 0)
    y_max = min(np.max(nnz_inds[0]) + np.random.randint(0, 50), img.shape[0])
    x_min = max(np.min(nnz_inds[1]) - np.random.randint(0, 50), 0)
    x_max = min(np.max(nnz_inds[1]) + np.random.randint(0, 50), img.shape[1])

    img = img_data[y_min:y_max, x_min:x_max]
    return  x_min, y_min, img
################################################################################################

class Random_Crop_image(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        img = random_crop_image(img)  # 生成15个字符的水印文本
        return img

def random_crop_image(image):
    img_data = image.copy()
    h, w = img_data.shape[:2]
    y_min = 0
    y_max = np.random.randint(int(h*0.5), int(h*0.8))
    x_min = 0
    x_max = np.random.randint(int(w*0.5), int(w*0.8))
    cropped_img = img_data[y_min:y_max, x_min:x_max]  # 裁剪后的前景图像
    ch, cw = cropped_img.shape[:2]  # 裁剪后图像的宽高

    bg_image = get_random_bg()
    bg_image = cv2.resize(bg_image, (w, h))

    # 定义原始图像的四个点（通常为矩形的四个角）
    pts1 = np.float32([[0, 0], [cw - 1, 0], [0, ch - 1], [cw - 1, ch - 1]])

    # 生成目标图像的四个随机点，保证点在图像内部，形成随机的透视效果
    margin = int(min([h, w])*0.2)  # 透视变换的边距
    pts2 = np.float32([[random.randint(0, margin), random.randint(0, margin)],
                       [random.randint(w - margin - 1, w - 1), random.randint(0, margin)],
                       [random.randint(0, margin), random.randint(h - margin - 1, h - 1)],
                       [random.randint(w - margin - 1, w - 1), random.randint(h - margin - 1, h - 1)]])

    # 生成透视变换矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)


    # 对前景图像进行透视变换
    warped_image = cv2.warpPerspective(cropped_img, matrix, (w, h))

    # 创建蒙版，检测透视变换后的图像中的非黑色区域
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

    # 反转蒙版，生成背景的蒙版区域
    mask_inv = cv2.bitwise_not(mask)

    # 提取背景图像中的对应区域
    background_part = cv2.bitwise_and(bg_image, bg_image, mask=mask_inv)

    # 提取透视变换后的前景图像中的非黑色部分
    foreground_part = cv2.bitwise_and(warped_image, warped_image, mask=mask)

    # 合成前景和背景
    result = cv2.add(background_part, foreground_part)

    return result

def get_random_bg():
    bg_image_name = os.listdir("/home/lixumin/project/local_match/coco_20k/")
    image = cv2.imread("/home/lixumin/project/local_match/coco_20k/"+random.choice(bg_image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

################################################################################################

class AddMoire(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        img = add_moire_pattern(img, frequency=random.randint(3,10), angle=random.randint(20,180), intensity=random.uniform(0.1, 0.2))
        return img

def add_moire_pattern(image, frequency=30, angle=0, intensity=0.5):
    """
    给图像添加彩色摩尔纹效果
    :param image: 输入图像 (H, W, 3)
    :param frequency: 正弦波的频率，越大图案越密集
    :param angle: 摩尔纹的旋转角度
    :param intensity: 摩尔纹与原图的混合比例，0-1 之间
    :return: 带有摩尔纹效果的图像
    """
    h, w, _ = image.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # 计算旋转的正弦波纹方向
    angle = np.deg2rad(angle)
    x_new = xx * np.cos(angle) - yy * np.sin(angle)
    y_new = xx * np.sin(angle) + yy * np.cos(angle)

    # 创建正弦波摩尔纹图案
    sine_wave_r = np.sin(2 * np.pi * (x_new / frequency)) * 127 + 128  # 红色通道
    sine_wave_g = np.sin(2 * np.pi * (y_new / frequency)) * 127 + 128  # 绿色通道
    sine_wave_b = np.sin(2 * np.pi * ((x_new + y_new) / frequency)) * 127 + 128  # 蓝色通道

    # 将正弦波摩尔纹组合成三通道图像
    random_center = (random.randint(0, w - 1), random.randint(0, h - 1))

    moire_pattern = np.stack([sine_wave_r, sine_wave_g, sine_wave_b], axis=-1).astype(np.uint8)
    
    if random.randint(0, 1):
        moire_pattern = apply_bulge_effect(moire_pattern, center=random_center, strength=random.randint(2, 10))

    # 将摩尔纹叠加到原图像
    moire_image = cv2.addWeighted(image, 1 - intensity, moire_pattern, intensity, 0)

    return moire_image


def apply_bulge_effect(image, center=None, strength=1.5):
    """
    给图像添加全局隆起（鼓包）效果，覆盖整个图像区域
    :param image: 输入图像
    :param center: 鼓包效果的中心位置 (x, y)，如果为 None 则默认为图像中心
    :param strength: 鼓包效果的强度，值越大隆起越明显
    :return: 添加隆起效果的图像
    """
    h, w = image.shape[:2]

    # 鼓包的中心，默认为图像中心
    if center is None:
        center = (w // 2, h // 2)

    # 生成原图像的坐标网格
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # 计算到鼓包中心的距离
    dx = x - center[0]
    dy = y - center[1]
    distance = np.sqrt(dx**2 + dy**2)

    # 计算最大距离，即从中心到图像的四个角的最大距离
    max_distance = np.sqrt(max(center[0], w - center[0])**2 + max(center[1], h - center[1])**2)

    # 计算变形比例，确保整个图像范围内都有鼓包效果
    factor = 1 - (distance / max_distance)**strength

    # 防止factor过小或为负值，保持合理的范围
    factor = np.clip(factor, 0, 1)

    # 计算新的坐标
    new_x = (dx * factor + center[0]).astype(np.float32)
    new_y = (dy * factor + center[1]).astype(np.float32)

    # 应用remap函数进行几何变换
    distorted_image = cv2.remap(image, new_x, new_y, interpolation=cv2.INTER_LINEAR)

    return distorted_image
################################################################################################

class AddSimulate(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        img = simulate_realistic_glare(img, num_shapes=random.randint(1,5), max_size_ratio=random.uniform(0.1,0.3))
        return img

def simulate_realistic_glare(image, num_shapes=8, max_alpha=0.1, max_size_ratio=0.3):
    """
    在图像上模拟更接近真实效果的屏幕反光。

    :param image: 输入图像.
    :param num_shapes: 用于模拟反光的不规则形状数.
    :param max_alpha: 最大透明度.
    :param max_size_ratio: 形状的最大尺寸与图像尺寸的比例.
    :return: 带有真实反光效果的图像.
    """
    output = image.copy()
    height, width = image.shape[:2]

    for _ in range(num_shapes):
        # 随机生成形状的位置、大小和角度
        center = (np.random.randint(0, width), np.random.randint(0, height))
        axes = (np.random.randint(10, int(width * max_size_ratio)), 
                np.random.randint(10, int(height * max_size_ratio)))
        angle = np.random.uniform(0, 360)

        # 生成渐变的透明度和颜色
        alpha = np.random.uniform(0, max_alpha)
        color = (np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255))

        # 创建一个透明图层
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
        
        # 添加模糊效果
        overlay = cv2.GaussianBlur(overlay, (21, 21), sigmaX=10, sigmaY=10)

        # 将叠加层与输出图像组合
        output = cv2.addWeighted(output, 1, overlay, 0.1, 0)

    return output

################################################################################################
my_transform =  alb_wrapper(
    alb.Compose(
        [
            Crop_image(p=0.9),
            ConditionalRandomScale(scale_limit=(-0.1, -0.0), p=1),
            alb.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.95),
            alb.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.8),
            Random_Crop_image(p=0.7),
            AddMoire(p=0.8),
            AddSimulate(p=0.8),
            add_shadown(p=0.8),
            alb.GaussNoise(10, p=0.8),
            alb.GaussianBlur((3, 3), p=0.8),
            Line_blur(p=0.95),
            watermark(p=0.5),
            alb.Blur(blur_limit=5, p=1),
            alb.MotionBlur(blur_limit=(3, 5), p=0.3),  # 动态模糊
            alb.RandomBrightnessContrast(p=0.8),
        ],
    )
)
my_test_transform = alb_wrapper(
    alb.Compose(
        [
            Crop_image(p=1),
            alb.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.95),
            alb.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.8),
            add_shadown(p=1),
            AddMoire(p=1),
        ]
        )
    )
my_test_transform_v2 = alb_wrapper(
    alb.Compose(
        [
            AddMoire(p=1),
        ]
        )
    )
if __name__ == '__main__':
    root_path = "/home/lixumin/project/local_match/figure-match/database/"
    image_name = sorted(os.listdir(root_path), key=lambda x:int(x.split(".")[0]))
    for i in image_name[1:333]:
        print(i)
        image_path = root_path + i
        image = Image.open(image_path).convert('RGB')
        image = Image.fromarray(my_transform(image))
        image.save(f"./show.jpg")
        input()
