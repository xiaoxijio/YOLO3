from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="数据路径")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="模型定义文件的路径")
    parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='权重文件路径')
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="类标签文件的路径")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="目标置信阈值")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="非极大值抑制阈值")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--n_cpu", type=int, default=0, help="批处理生成过程中使用的CPU线程数")
    parser.add_argument("--img_size", type=int, default=416, help='每个图像维度的大小')
    parser.add_argument("--checkpoint_model", type=str, help="检查点模型路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    model = Darknet(args.model_def, img_size=args.img_size).to(device)

    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    model.eval()

    dataloader = DataLoader(
        ImageFolder(args.image_folder, img_size=args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu
    )

    classes = load_classes(args.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs = []  # 存储图像路径
    img_detections = []  # 存储每个图像索引的检测结果

    print("执行目标检测")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))

        with torch.no_grad():
            detections = model(input_imgs)  # 放到模型里预测
            detections = non_max_suppression(detections, args.conf_thres, args.nms_thres)  # 对预测出的结果非极大值抑制

        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, 推理时间: %s" % (batch_i, inference_time))

        imgs.extend(img_paths)
        img_detections.extend(detections)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("保存图片")
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) 图片: '%s'" % (img_i, path))

        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # 绘制检测的边界框和标签
        if detections is not None:
            # 重新缩放框到原始图像
            detections = rescale_boxes(detections, args.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ 类别: %s, 置信度: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # 创建一个矩形补丁
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # 将 bbox 添加到 plot
                ax.add_patch(bbox)
                # 添加标签
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # 保存生成的图像与检测
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
