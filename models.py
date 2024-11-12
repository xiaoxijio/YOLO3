import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.parse_config import *
from utils.utils import build_targets, to_cpu


def create_modules(module_defs):
    hyperparams = module_defs.pop(0)  # module_def第一个元素是超参数定义 不属于网络层数据 获取并去掉
    output_filters = [int(hyperparams["channels"])]  # 通道数
    module_list = nn.ModuleList()  # 按顺序存储各层模
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()  # 遍历 module_defs，为每个模块定义构建一个 nn.Sequential 实例，并添加到 modules 中

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])  # 归一化
            filters = int(module_def["filters"])  # 卷积核数量
            kernel_size = int(module_def["size"])  # 卷积核尺寸
            pad = (kernel_size - 1) // 2  # 填充大小
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:  # 若启用了批归一化，则添加 nn.BatchNorm2d
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":  # 若激活函数为 leaky， 则添加 nn.LeakyReLU 激活层
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":  # yolo3里面好像没有池化层吧 不看
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":  # 上采样层
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")  # 上采样层使用最近邻插值法
            modules.add_module(f"upsample_{module_i}", upsample)  # 以 stride 作为缩放因子

        elif module_def["type"] == "route":  # 路由层 输入1：26*26*256 输入2：26*26*128  输出：26*26*（256+128）
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":  # 捷径层  残差连接
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":  # YOLO 检测层 一共有三层 检测大中小三个图像
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]

            anchors = [int(x) for x in module_def["anchors"].split(",")]  # 加载锚框
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])  # 类别数量
            img_size = int(hyperparams["height"])  # 输入图像大小

            yolo_layer = YOLOLayer(anchors, num_classes, img_size)  # 检测层
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules)  # 将构建好的 modules 添加到 module_list
        output_filters.append(filters)  # 更新 output_filters，为下一层提供当前层的输出通道数

    return hyperparams, module_list  # 返回超参数和模块列表


class EmptyLayer(nn.Module):
    """“route”和“shortcut”图层的占位符"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    """ nn.Upsample已弃用 """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    """检测层"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors  # 锚框的值是在训练前通过聚类方法从数据集中预选出的
        self.num_anchors = len(anchors)  # 锚框数量
        self.num_classes = num_classes  # 目标类别数
        self.ignore_thres = 0.5  # 阈值
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1  # 用于调整检测中物体和无物体的损失权重
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim  # 输入图片的尺寸
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        """根据特征图的网格大小（grid_size）来计算网格的偏移量，以便将预测的相对坐标转换为绝对坐标"""
        self.grid_size = grid_size  # eg: 当前格大小 13*13
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size  # 原来图片大小 / 特征图大小   定义了每个网格的像素距离

        # 计算每个网格的偏移量
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  # eg: (1,1,13,13)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)  # eg: (1,1,13,13)
        # 据特征图与原图的比值，将锚框按这个比值缩放，以适配当前特征图的尺寸
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        """将输入的特征图转换成检测结果，包括目标框的中心坐标、宽高、置信度和类别预测等"""
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        # print(x.shape)  # eg: (64, 255, 13, 13)
        prediction = (  # (batch_size, num_anchors, grid_size, grid_size, num_classes + 5) 其中 5 表示 x、y、w、h 和置信度
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # print(prediction.shape)  # eg: (64, 3, 13, 13, 85) 255 = 3 * 85

        # 提取输出结果
        x = torch.sigmoid(prediction[..., 0])  # Center x 话说这个x跟输入x对标了, 不过问题不大 后面也没用到输入x了
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # 目标置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # 类别预测

        if grid_size != self.grid_size:  # 如果网格大小不匹配当前我们计算新的偏移量
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)  # 相对位置转绝对位置

        # 用锚添加偏移和缩放  特征图中的实际位置
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x  # 获得相对网格的绝对位置
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # 计算出目标框的实际宽高
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # 还原到原始图中
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0  # output 包含了每个检测框的绝对坐标、置信度和类别预测
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,  # 预测的 x y w h
                pred_cls=pred_cls,  # 预测的类别
                target=targets,  # 真实值
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # iou_scores: 真实值与最匹配的 anchor的IOU得分值
            # class_mask: 分类正确的索引
            # obj_mask: 目标框所在位置的最好 anchor置为 1
            # noobj_mask obj_mask那里置 0，还有计算的 iou大于阈值的也置 0，其他都为 1 tx, ty, tw, th
            # 对应的对于该大小的特征图的 xywh目标值也就是我们需要拟合的值 tconf 目标置信度

            # Loss: 屏蔽没目标的损失计算（除了置信度的, 因为有物体的和没物体的都要算置信度）
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # 只计算有目标的
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # 置信度损失
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])  # 分别针对包含和不包含目标的网格
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj  # 有物体越接近 1越好 没物体的越接近 0越好
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])  # 分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls  # 总损失

            # 指标
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3对象检测模型"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)  # 读取 yolo3网络结构数据
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":  # 当残差连接的输入和输出形状（包括通道数）一致时, 通常可以直接相加
                layer_i = int(module_def["from"])  # 不一致时, 通常使用 1x1 卷积来调整输入的通道数, 使其与输出匹配
                x = layer_outputs[-1] + layer_outputs[layer_i]  # 将当前层的输出与前面某一层的输出相加
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """ 解析并加载存储在 weights_path 中的权重"""
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # 训练期间看到的图像数量
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # 设置加载权重的层数上限
        cutoff = None
        if "darknet53.conv.74" in weights_path:  # 如果文件名包含 darknet53.conv.74
            cutoff = 75  # 仅加载 Darknet 网络的前 75 层的权重

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:  # 如果达到 cutoff，则停止加载
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:  # 如果有 加载批归一化层的参数
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # 权重数量
                    # 偏置
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # 权重
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # 均值
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # 方差
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # 加载卷积层的偏置
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # 加载卷积层的权重
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
