import numpy as np
import torch
from tqdm import tqdm


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """ 在 path 加载类标签 """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    计算平均精度，给定召回率和精度曲线。
    来源:https://github.com/rafaelpadilla/Object-Detection-Metrics

    :param tp: 真实阳性数
    :param conf: 0-1的对象值
    :param pred_cls: 预测的对象类
    :param target_cls: 真正的对象类
    :return: 平均精度
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """
    计算平均精度，给定召回率和精度曲线
    代码来自: https://github.com/rbgirshick/py-faster-rcnn
    :param recall: 召回曲线
    :param precision: 精度曲线
    :return: 平均精度
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ 计算每个样本的真实阳性数TP、预测分数和预测标签。 """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]  # 预测框坐标
        pred_scores = output[:, 4]  # 预测分数
        pred_labels = output[:, -1]  # 预测标签

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]  # 真实目标框
        target_labels = annotations[:, 0] if len(annotations) else []  # 真实标签
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                if len(detected_boxes) == len(annotations):
                    break  # 如果找到目标就中断

                if pred_label not in target_labels:
                    continue  # 如果label不是目标标签之一，则忽略

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    删除对象置信度得分低于 conf_thres 的检测并执行非极大值抑制，以进一步过滤检测。
    返回形状为：(x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # (center x, center y, width, height) --> (x1, y1, x2, y2)
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]  # 过滤出低于阈值的置信度分数
        if not image_pred.size(0):  # 如果过滤后没有框保留下来，则跳过该图片
            continue
        # eg: 对象置信度: 检测框包含物体的可能性为 0.8  类置信度: 这个物体是 人 的可能性为 0.9
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]  # 对象置信度 * 类置信度
        image_pred = image_pred[(-score).argsort()]  # 排序分数
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # 获取类别置信度和类别预测结果
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # 非极大值抑制
        keep_boxes = []  # 初始化为空列表，用于保存最终的非极大值抑制结果
        while detections.size(0):  # 对于每一个剩余的 detections
            # 置信度最高的检测框 detections[0]与剩余的框比较
            # 筛选出哪些框与当前框 detections[0] 的重叠程度较高
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]  # 检查 detections[0] 与其他框的类别标签是否相同
            invalid = large_overlap & label_match  # 与 detections[0] 类别相同且重叠率超过阈值的框(重复框)
            weights = detections[invalid, 4:5]  # 重叠框的置信度权重
            # 通过加权方式合并重叠框(保留了重叠框的信息，同时计算出一个更精确的框)
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]  # 保存合并后的检测框
            detections = detections[~invalid]
        if keep_boxes:  # 如果 keep_boxes 不为空，最终将结果存入 output 的对应位置
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # batch_size
    nA = pred_boxes.size(1)  # 3个锚框
    nC = pred_cls.size(-1)  # 类别的数量
    nG = pred_boxes.size(2)  # 特征图网格大小

    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)  # 初始化全0, anchor包含物体, 即为1，默认为0 考虑前景
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)  # 初始化全1, anchor不包含物体, 则为1，默认为1 考虑背景
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)  # 类别掩膜，类别预测正确即为1，默认全为0
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)  # 预测框与真实框的iou得分
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)  # 真实框相对于网格的位置
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    target_boxes = target[:, 2:6] * nG  # target中的xywh都是0-1的，可以得到其在当前gridsize上的xywh
    gxy = target_boxes[:, :2]  # 就是得到了在特征图上的实际位置, 不是在真实图上的实际位置哦
    gwh = target_boxes[:, 2:]

    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # 每一种规格的anchor跟每个标签上的框的IOU得分
    # print(ious.shape)
    best_ious, best_n = ious.max(0)  # 得到其最高分以及哪种规格框和当前目标最相似

    b, target_labels = target[:, :2].long().t()  # 真实框所对应的 batch，以及每个框所代表的实际类别
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # 位置信息，向下取整了

    obj_mask[b, best_n, gj, gi] = 1  # 实际包含物体的设置成1
    noobj_mask[b, best_n, gj, gi] = 0  # 相反

    # 当超过忽略阈值时，将noobj掩码设置为零
    for i, anchor_ious in enumerate(ious.t()):  # IOU超过了指定的阈值就相当于有物体了
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    tx[b, best_n, gj, gi] = gx - gx.floor()  # 根据真实框所在位置，得到其相当于网络的位置
    ty[b, best_n, gj, gi] = gy - gy.floor()

    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    tcls[b, best_n, gj, gi, target_labels] = 1  # 将真实框的标签转换为one-hot编码形式
    # 计算标签正确性和 iou在最好的锚
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes,
                                             x1y1x2y2=False)  # 与真实框相匹配的预测框之间的iou值

    tconf = obj_mask.float()  # 真实框的置信度，也就是1
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
