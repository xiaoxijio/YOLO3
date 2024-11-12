import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    """将一个矩形图像 img 填充成正方形图像, 填充区域使用指定的 pad_value 值"""

    _, h, w = img.shape
    dim_diff = np.abs(h - w)  # 计算高度和宽度的差值, 以确定需要填充的像素数
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2  # 将 dim_diff 均匀分成两部分（左/右或上/下）
    # pad = (0, 0, pad1, pad2)，表示在高度方向（上、下）分别填充 pad1 和 pad2 个像素
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)  # constant常数填充

    return img, pad


def resize(image, size):
    """对 image 进行插值调整（缩放），将其尺寸调整为指定的 size"""
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)  # nearest最近邻插值法
    return image


def random_resize(images, min_size=288, max_size=448):
    """随机选择一个新的尺寸 new_size，然后对 images 进行插值调整，将其尺寸缩放为 new_size"""
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path))
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img_path = 'D:\study\pycharm\人工智能\YOLO\data\coco' + img_path
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))  # 转为 Tensor 和 RGB

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape  # _是通道数, 不考虑
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0)  # 将 img填充成正方形图像
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        label_path = 'D:\study\pycharm\人工智能\YOLO\data\coco' + label_path

        targets = None
        # 将边界框的坐标转换为适应填充后的图像尺寸
        if os.path.exists(label_path):
            # 边界框数据(class, x_center, y_center, width, height)
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))  # 加载边界框数据

            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)  # 计算未填充和未缩放图像的边界框坐标
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)  # 左上角 (x1, y1) 和右下角 (x2, y2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            x1 += pad[0]  # 调整填充
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            boxes[:, 1] = ((x1 + x2) / 2) / padded_w  # 计算新的中心坐标和宽高，并归一化
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))  # len(boxes) 表示标签文件中包含的边界框的数量
            targets[:, 1:] = boxes  # 第一列存放类别, 后面存 box数据

        # if self.augment:  # 图像增强
        #     if np.random.random() < 0.5:
        #         img, targets = horisontal_flip(img, targets)  # 水平翻转

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        targets = [boxes for boxes in targets if boxes is not None]  # 过滤掉没有标注框的样本（None），避免不必要的计算
        for i, boxes in enumerate(targets):  # 向目标添加样例索引
            boxes[:, 0] = i  # 这种索引便于后续在模型中计算每个框对应的样本
        targets = torch.cat(targets, 0)  # 将所有目标框沿第 0维（样本维）进行拼接，以便处理成单一张量格式

        if self.multiscale and self.batch_count % 10 == 0:  # 如果启用了多尺度训练， 每10个批次就随机选择一个新尺寸
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])  # 调整图像大小堆叠到一个批次张量中
        self.batch_count += 1  # 用于触发多尺度训练
        return paths, imgs, targets  # 每个样本的路径、调整后的图像张量和目标框张量

    def __len__(self):
        return len(self.img_files)
