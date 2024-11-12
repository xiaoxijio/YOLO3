from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from terminaltables import AsciiTable

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="梯度累加数 before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="模型定义文件的路径")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="数据配置文件路径")
    parser.add_argument("--pretrained_weights", type=str, help="如果指定，则从检查点模型开始")
    parser.add_argument("--n_cpu", type=int, default=0, help="批处理生成过程中使用的CPU线程数")
    parser.add_argument("--img_size", type=int, default=416, help="每个图像维度的大小")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="保存模型权重的间隔时间")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="验证集评估的间隔时间")
    parser.add_argument("--compute_map", default=False, help="每10批计算一次map")
    parser.add_argument("--multiscale_training", default=True, help="允许多规模的培训")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    model = Darknet(args.model_def).to(device)
    model.apply(weights_init_normal)

    # 如果指定，我们从检查点开始
    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            model.load_darknet_weights(args.pretrained_weights)

    # 获取数据
    dataset = ListDataset(train_path, augment=True, multiscale=args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # print('imgs', imgs.shape)
            # print('targets', targets.shape)
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % args.gradient_accumulations:  # 累积几个 batch 的梯度再进行一次更新
                optimizer.step()
                optimizer.zero_grad()

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # epoch所剩下的近似时间
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)

            model.seen += imgs.size(0)

        if epoch % args.evaluation_interval == 0:
            print("---- 评估模型 ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=args.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
