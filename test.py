import argparse

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


@torch.no_grad()
def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels = []
    sample_metrics = []  # 元组列表 (TP, conf, pred)

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        labels += targets[:, 1].tolist()  # 提取标签
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])  # 重新调节目标
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        outputs = model(imgs)
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="模型定义文件的路径")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="数据配置文件路径")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="权重文件路径")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="类标签文件的路径")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="检测到所需的阈值")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="目标置信阈值")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="非极大值抑制阈值")
    parser.add_argument("--n_cpu", type=int, default=8, help="批处理生成过程中使用的CPU线程数")
    parser.add_argument("--img_size", type=int, default=416, help="每个图像维度的大小")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(args.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    model = Darknet(args.model_def).to(device)
    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    print("计算 mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        img_size=args.img_size,
        batch_size=8,
    )

    print("平均精度:")
    for i, c in enumerate(ap_class):
        print(f"+ 类别 '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
