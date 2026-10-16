
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯手写 YOLOv8 ONNX 目标检测测评程序

依赖：
    pip install onnxruntime opencv-python numpy

数据集目录：
    dataset/
      images/val/*.jpg
      labels/val/*.txt

TXT 每行格式：
    class_id x_center y_center width height

坐标为相对于原图宽高的归一化值。

说明：
1. 针对标准 YOLOv8 Detect ONNX 输出：4 个框参数 + 各类别分数。
2. 默认模型输出未内置 NMS。
3. 不依赖 Ultralytics。
4. 支持模型加载、推理、精度评测、性能测试及结果导出。
"""

import csv
import json
import time
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ============================================================
# 1. 图像预处理
# ============================================================

def letterbox(image, new_shape=(640, 640),
              color=(114, 114, 114)):
    """保持宽高比缩放并填充。"""
    h, w = image.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    target_h, target_w = new_shape

    scale = min(target_h / h, target_w / w)

    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h

    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, (left, top)


def preprocess(image, input_hw):
    """将 BGR 图片转换为模型输入张量。"""
    start = time.perf_counter()

    padded, scale, pad = letterbox(image, input_hw)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    tensor = tensor / 255.0
    tensor = np.ascontiguousarray(tensor)

    elapsed_ms = (time.perf_counter() - start) * 1000

    meta = {
        "scale": scale,
        "pad": pad,
        "orig_hw": image.shape[:2]
    }

    return tensor, meta, elapsed_ms


# ============================================================
# 2. 边界框与 NMS
# ============================================================

def xywh_to_xyxy(boxes):
    """中心点宽高格式转换为左上右下坐标格式。"""
    out = np.empty_like(boxes, dtype=np.float32)

    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return out


def box_iou_one_to_many(box, boxes):
    """计算一个 xyxy 框与多个 xyxy 框的 IoU。"""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter = (
        np.maximum(0, xx2 - xx1)
        * np.maximum(0, yy2 - yy1)
    )

    area1 = (
        max(0.0, float(box[2] - box[0]))
        * max(0.0, float(box[3] - box[1]))
    )

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def nms_xyxy(boxes, scores, iou_threshold):
    """使用 NumPy 实现贪心非极大值抑制。"""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T

    areas = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = (
            np.maximum(0, xx2 - xx1)
            * np.maximum(0, yy2 - yy1)
        )

        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)

        order = rest[iou <= iou_threshold]

    return keep


# ============================================================
# 3. ONNX 输出解析与后处理
# ============================================================

def decode_output(
    outputs,
    meta,
    conf_threshold=0.001,
    iou_threshold=0.45,
    max_det=300
):
    """
    解析标准 YOLOv8 Detect 输出。

    支持：
        [1, C, N]
        [1, N, C]

    其中 C = 4 + 类别数。
    返回原图坐标下的检测结果。
    """
    start = time.perf_counter()

    pred = np.asarray(outputs[0])

    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(
                f"当前脚本只处理 batch=1，实际输出：{pred.shape}"
            )
        pred = pred[0]

    if pred.ndim != 2:
        raise ValueError(
            f"不支持的 ONNX 输出形状：{np.asarray(outputs[0]).shape}"
        )

    # 常见输出为 [84, 8400]，需要转置为 [8400, 84]
    if pred.shape[0] <= 512 and pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] < 5:
        raise ValueError(
            f"输出形状 {pred.shape} 不符合标准 YOLOv8 Detect 格式"
        )

    pred = pred.astype(np.float32, copy=False)

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    valid = np.isfinite(scores) & (scores >= conf_threshold)
    valid &= np.isfinite(boxes_xywh).all(axis=1)

    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(boxes_xywh) == 0:
        return [], (time.perf_counter() - start) * 1000

    boxes = xywh_to_xyxy(boxes_xywh)

    scale = meta["scale"]
    pad_x, pad_y = meta["pad"]
    orig_h, orig_w = meta["orig_hw"]

    # 将模型输入坐标映射回原图坐标
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

    # 按类别分别进行 NMS
    keep_all = []

    for cls_id in np.unique(class_ids):
        inds = np.flatnonzero(class_ids == cls_id)

        local_keep = nms_xyxy(
            boxes[inds],
            scores[inds],
            iou_threshold
        )

        keep_all.extend(inds[local_keep].tolist())

    keep_all = sorted(
        keep_all,
        key=lambda i: float(scores[i]),
        reverse=True
    )[:max_det]

    detections = [
        {
            "xyxy": boxes[i].astype(float).tolist(),
            "score": float(scores[i]),
            "class_id": int(class_ids[i])
        }
        for i in keep_all
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return detections, elapsed_ms


def get_input_hw(session, default=640):
    """读取 ONNX 模型输入尺寸。"""
    shape = session.get_inputs()[0].shape

    h = (
        shape[2]
        if len(shape) >= 4 and isinstance(shape[2], int)
        else default
    )

    w = (
        shape[3]
        if len(shape) >= 4 and isinstance(shape[3], int)
        else default
    )

    return int(h), int(w)


# ============================================================
# 4. 标签读取与指标计算
# ============================================================

def load_labels(label_path, image_hw, num_classes=None):
    """读取 YOLO TXT 标签并转换为原图像素 xyxy 坐标。"""
    h, w = image_hw

    if not label_path.exists():
        raise FileNotFoundError(f"缺少标签文件：{label_path}")

    text = label_path.read_text(encoding="utf-8-sig").strip()

    if not text:
        return []

    gt = []

    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_no} 格式错误，应为 5 列"
            )

        cls_f, xc, yc, bw, bh = map(float, parts)
        cls_id = int(cls_f)

        if cls_f != cls_id or cls_id < 0:
            raise ValueError(
                f"{label_path}:{line_no} 类别 ID 必须为非负整数"
            )

        if num_classes is not None and cls_id >= num_classes:
            raise ValueError(
                f"{label_path}:{line_no} 类别 ID {cls_id} "
                f"超出模型类别数 {num_classes}"
            )

        if (
            not np.isfinite([xc, yc, bw, bh]).all()
            or bw < 0
            or bh < 0
        ):
            raise ValueError(
                f"{label_path}:{line_no} 坐标非法"
            )

        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        gt.append({
            "xyxy": [x1, y1, x2, y2],
            "class_id": cls_id
        })

    return gt


def match_image(preds, gts, iou_threshold):
    """
    对单张图片进行类别一致的贪心匹配。

    预测按置信度降序匹配，每个 GT 最多匹配一次。
    返回 TP 标记数组、TP 数量、FN 数量。
    """
    order = sorted(
        range(len(preds)),
        key=lambda i: preds[i]["score"],
        reverse=True
    )

    matched = set()
    tp = np.zeros(len(preds), dtype=np.float32)

    for pi in order:
        pred = preds[pi]

        candidates = [
            (gi, gt)
            for gi, gt in enumerate(gts)
            if gi not in matched
            and gt["class_id"] == pred["class_id"]
        ]

        if not candidates:
            continue

        gt_boxes = np.asarray(
            [gt["xyxy"] for _, gt in candidates],
            dtype=np.float32
        )

        ious = box_iou_one_to_many(
            np.asarray(pred["xyxy"], dtype=np.float32),
            gt_boxes
        )

        best_local = int(np.argmax(ious))

        if ious[best_local] >= iou_threshold:
            matched.add(candidates[best_local][0])
            tp[pi] = 1.0

    return tp, len(matched), len(gts) - len(matched)


def compute_ap(recall, precision):
    """使用单调包络和 101 点插值计算 AP。"""
    if len(recall) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    samples = np.linspace(0, 1, 101)
    vals = []

    for r in samples:
        valid = mrec >= r
        vals.append(
            float(np.max(mpre[valid]))
            if np.any(valid)
            else 0.0
        )

    return float(np.mean(vals))


def evaluate_class_ap(
    all_preds,
    gt_by_image,
    class_id,
    iou_threshold
):
    """在指定类别和 IoU 阈值下计算 AP。"""
    total_gt = sum(
        sum(1 for gt in gts if gt["class_id"] == class_id)
        for gts in gt_by_image.values()
    )

    if total_gt == 0:
        return None, 0, 0

    preds = []

    for image_key, detections in all_preds.items():
        for det in detections:
            if det["class_id"] == class_id:
                preds.append((det["score"], image_key, det))

    preds.sort(key=lambda x: x[0], reverse=True)

    matched_by_image = {
        key: set()
        for key in gt_by_image
    }

    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)

    for i, (_, image_key, pred) in enumerate(preds):
        gts = gt_by_image[image_key]

        candidates = [
            (gi, gt)
            for gi, gt in enumerate(gts)
            if gt["class_id"] == class_id
            and gi not in matched_by_image[image_key]
        ]

        if not candidates:
            fp[i] = 1.0
            continue

        gt_boxes = np.asarray(
            [gt["xyxy"] for _, gt in candidates],
            dtype=np.float32
        )

        ious = box_iou_one_to_many(
            np.asarray(pred["xyxy"], dtype=np.float32),
            gt_boxes
        )

        best = int(np.argmax(ious))

        if ious[best] >= iou_threshold:
            matched_by_image[image_key].add(
                candidates[best][0]
            )
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(
        tp_cum + fp_cum,
        1e-12
    )

    ap = compute_ap(recall, precision)

    return ap, int(tp.sum()), int(fp.sum())


# ============================================================
# 5. YOLOv8 ONNX 测评器
# ============================================================

class YOLOv8ONNXEvaluator:
    """
    YOLOv8 ONNX 目标检测测评器。

    主要接口：
        load_model()      加载模型
        predict_image()   单张图片推理
        evaluate()        执行完整测评
        get_metrics()     获取测评指标
        export_results()  导出测评结果
        print_report()    打印测评摘要
        run()             一键执行完整流程
    """

    def __init__(
        self,
        model_path,
        images_dir,
        labels_dir,
        output_dir="evaluation_results",
        image_size=640,
        conf_threshold=0.25,
        nms_iou_threshold=0.45,
        max_detections=300,
        providers=None,
        class_names_file="",
        warmup_rounds=3
    ):
        self.model_path = Path(model_path)
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)

        self.image_size = int(image_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_detections = int(max_detections)

        self.requested_providers = (
            providers or ["CPUExecutionProvider"]
        )

        if isinstance(self.requested_providers, str):
            self.requested_providers = [
                x.strip()
                for x in self.requested_providers.split(",")
                if x.strip()
            ]

        self.class_names_file = str(class_names_file or "")
        self.warmup_rounds = int(warmup_rounds)

        if not 0 <= self.conf_threshold <= 1:
            raise ValueError(
                "conf_threshold 必须在 0 到 1 之间"
            )

        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError(
                "nms_iou_threshold 必须在 0 到 1 之间"
            )

        if self.max_detections < 1:
            raise ValueError(
                "max_detections 必须大于 0"
            )

        if self.image_size < 1:
            raise ValueError(
                "image_size 必须大于 0"
            )

        self.session = None
        self.input_name = None
        self.input_hw = None
        self.num_classes = None
        self.class_names = []
        self.providers = []

        self.metrics = None
        self.per_class_metrics = []
        self.timing_rows = []
        self.all_predictions = {}
        self.ground_truth_by_image = {}
        self._has_evaluated = False

    # --------------------------------------------------------
    # 模型加载
    # --------------------------------------------------------

    def load_model(self):
        """加载 ONNX 模型并推断模型类别数。"""
        if self.session is not None:
            return self

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"模型不存在：{self.model_path}"
            )

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"图片目录不存在：{self.images_dir}"
            )

        if not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"标签目录不存在：{self.labels_dir}"
            )

        available = ort.get_available_providers()

        self.providers = [
            p for p in self.requested_providers
            if p in available
        ]

        if not self.providers:
            raise RuntimeError(
                f"请求的 ONNX Runtime providers 不可用。"
                f"请求：{self.requested_providers}；"
                f"可用：{available}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        self.input_hw = get_input_hw(
            self.session,
            self.image_size
        )

        # 使用虚拟输入检查模型输出
        dummy = np.zeros(
            (1, 3, self.input_hw[0], self.input_hw[1]),
            dtype=np.float32
        )

        sample_outputs = self.session.run(
            None,
            {self.input_name: dummy}
        )

        raw = np.asarray(sample_outputs[0])

        if raw.ndim == 3:
            dims = raw.shape[1:]

            channel_candidates = [
                d for d in dims
                if isinstance(d, (int, np.integer))
                and d <= 512
            ]

            if not channel_candidates:
                raise ValueError(
                    f"无法从模型输出推断类别数，"
                    f"输出形状：{raw.shape}"
                )

            channels = min(channel_candidates)

        elif raw.ndim == 2:
            channels = min(raw.shape)

        else:
            raise ValueError(
                f"不支持的模型输出形状：{raw.shape}"
            )

        self.num_classes = int(channels - 4)

        if self.num_classes < 1:
            raise ValueError(
                f"输出形状 {raw.shape} 不符合标准 "
                "YOLOv8 Detect 输出格式"
            )

        # 读取类别名称
        if self.class_names_file:
            names_path = Path(self.class_names_file)

            if not names_path.is_file():
                raise FileNotFoundError(
                    f"类别名称文件不存在：{names_path}"
                )

            self.class_names = [
                line.strip()
                for line in names_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]

            if len(self.class_names) != self.num_classes:
                raise ValueError(
                    f"类别名称数 {len(self.class_names)} "
                    f"与模型类别数 {self.num_classes} 不一致"
                )

        else:
            self.class_names = [
                str(i) for i in range(self.num_classes)
            ]

        print(f"模型加载成功：{self.model_path}")
        print(f"模型输入尺寸：{self.input_hw}")
        print(f"模型类别数量：{self.num_classes}")
        print(f"执行设备：{self.providers}")

        return self

    # --------------------------------------------------------
    # 单张图片推理
    # --------------------------------------------------------

    def predict_image(self, image):
        """
        对单张 BGR 图片执行推理。

        返回：
            detections：检测框列表
            timing：各阶段耗时
        """
        self.load_model()

        tensor, meta, preprocess_ms = preprocess(
            image,
            self.input_hw
        )

        t0 = time.perf_counter()

        outputs = self.session.run(
            None,
            {self.input_name: tensor}
        )

        inference_ms = (
            time.perf_counter() - t0
        ) * 1000

        detections, postprocess_ms = decode_output(
            outputs,
            meta,
            conf_threshold=0.001,
            iou_threshold=self.nms_iou_threshold,
            max_det=self.max_detections
        )

        timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": (
                preprocess_ms
                + inference_ms
                + postprocess_ms
            )
        }

        return detections, timing

    # --------------------------------------------------------
    # 完整数据集测评
    # --------------------------------------------------------

    def evaluate(self):
        """遍历测试集，计算精度指标与性能指标。"""
        self.load_model()
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        images = sorted(
            p for p in self.images_dir.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )

        if not images:
            raise RuntimeError(
                f"图片目录中没有找到图片：{self.images_dir}"
            )

        print(f"测试图片数量：{len(images)}")

        # 模型预热，不计入正式测评耗时
        dummy_image = np.zeros(
            (self.input_hw[0], self.input_hw[1], 3),
            dtype=np.uint8
        )

        for _ in range(max(0, self.warmup_rounds)):
            tensor, _, _ = preprocess(
                dummy_image,
                self.input_hw
            )

            self.session.run(
                None,
                {self.input_name: tensor}
            )

        all_preds = {}
        gt_by_image = {}
        timing_rows = []
        total_gt = 0
        total_predictions_at_conf = 0

        for idx, image_path in enumerate(images, 1):
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[跳过] 无法读取图片：{image_path}")
                continue

            rel = image_path.relative_to(self.images_dir)

            label_path = (
                self.labels_dir / rel
            ).with_suffix(".txt")

            gts = load_labels(
                label_path,
                image.shape[:2],
                self.num_classes
            )

            detections, timing = self.predict_image(image)

            key = str(rel).replace("\\", "/")

            all_preds[key] = detections
            gt_by_image[key] = gts

            total_gt += len(gts)

            total_predictions_at_conf += sum(
                d["score"] >= self.conf_threshold
                for d in detections
            )

            timing_rows.append({
                "image": key,
                **timing,
                "num_gt": len(gts),
                "num_predictions": len(detections)
            })

            if idx % 50 == 0 or idx == len(images):
                print(
                    f"已处理 {idx}/{len(images)} 张图片"
                )

        if not gt_by_image:
            raise RuntimeError(
                "没有成功处理任何图片"
            )

        # 计算指定置信度阈值下的 P/R/F1
        tp_total = 0
        fp_total = 0
        fn_total = 0

        for key, preds in all_preds.items():
            filtered = [
                d for d in preds
                if d["score"] >= self.conf_threshold
            ]

            _, tp, fn = match_image(
                filtered,
                gt_by_image[key],
                0.5
            )

            tp_total += tp
            fn_total += fn
            fp_total += len(filtered) - tp

        precision = (
            tp_total / (tp_total + fp_total)
            if tp_total + fp_total
            else 0.0
        )

        recall = (
            tp_total / (tp_total + fn_total)
            if tp_total + fn_total
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        # 计算各类别 AP50、AP50:95
        iou_thresholds = np.arange(
            0.50,
            0.951,
            0.05
        )

        per_class = []

        for cls_id, cls_name in enumerate(self.class_names):
            ap_values = []
            ap50 = None

            for threshold in iou_thresholds:
                ap, _, _ = evaluate_class_ap(
                    all_preds,
                    gt_by_image,
                    cls_id,
                    float(threshold)
                )

                if ap is not None:
                    ap_values.append(ap)

                    if abs(threshold - 0.5) < 1e-6:
                        ap50 = ap

            gt_count = sum(
                sum(
                    g["class_id"] == cls_id
                    for g in gts
                )
                for gts in gt_by_image.values()
            )

            if ap50 is not None:
                cls_tp = 0
                cls_fp = 0
                cls_fn = 0

                for key, preds in all_preds.items():
                    class_preds = [
                        d for d in preds
                        if d["class_id"] == cls_id
                        and d["score"] >= self.conf_threshold
                    ]

                    class_gts = [
                        g for g in gt_by_image[key]
                        if g["class_id"] == cls_id
                    ]

                    _, ctp, cfn = match_image(
                        class_preds,
                        class_gts,
                        0.5
                    )

                    cls_tp += ctp
                    cls_fn += cfn
                    cls_fp += len(class_preds) - ctp

                cp = (
                    cls_tp / (cls_tp + cls_fp)
                    if cls_tp + cls_fp
                    else 0.0
                )

                cr = (
                    cls_tp / (cls_tp + cls_fn)
                    if cls_tp + cls_fn
                    else 0.0
                )

                cf1 = (
                    2 * cp * cr / (cp + cr)
                    if cp + cr
                    else 0.0
                )

                per_class.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "gt_count": gt_count,
                    "precision": cp,
                    "recall": cr,
                    "f1": cf1,
                    "ap50": ap50,
                    "ap50_95": float(np.mean(ap_values))
                })

            else:
                per_class.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "gt_count": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ap50": None,
                    "ap50_95": None
                })

        valid_classes = [
            x for x in per_class
            if x["ap50"] is not None
        ]

        map50 = (
            float(np.mean([
                x["ap50"] for x in valid_classes
            ]))
            if valid_classes
            else 0.0
        )

        map5095 = (
            float(np.mean([
                x["ap50_95"] for x in valid_classes
            ]))
            if valid_classes
            else 0.0
        )

        # 计算各阶段平均耗时
        pre_mean = mean(
            x["preprocess_ms"] for x in timing_rows
        )

        infer_mean = mean(
            x["inference_ms"] for x in timing_rows
        )

        post_mean = mean(
            x["postprocess_ms"] for x in timing_rows
        )

        total_values = [
            x["total_ms"] for x in timing_rows
        ]

        total_mean = mean(total_values)

        total_std = (
            stdev(total_values)
            if len(total_values) > 1
            else 0.0
        )

        def ratio(ms: float) -> str:
            """单个阶段耗时占总耗时的百分比。"""
            if total_mean > 0:
                return f"{ms / total_mean * 100:.1f}%"
            return "0.0%"

        # 报告字段：字段名与 report_generator/sections 中的
        # s04_accuracy.py / s05_speed.py 字段一一对应（边缘端量化模型），
        # 取值为报告就绪的格式化字符串，可经 run.py 直接填充渲染。
        summary = {
            "model": str(self.model_path.resolve()),
            "images_dir": str(self.images_dir.resolve()),
            "labels_dir": str(self.labels_dir.resolve()),
            "providers": self.providers,
            "input_size": list(self.input_hw),
            "num_classes": self.num_classes,
            "num_images": len(gt_by_image),
            "num_ground_truth_objects": total_gt,
            "num_predictions_at_conf_threshold": (
                total_predictions_at_conf
            ),
            "confidence_threshold_for_precision_recall_f1": (
                self.conf_threshold
            ),
            "nms_iou_threshold": self.nms_iou_threshold,

            # 第 4 章 检测精度（s04_accuracy.py）
            "det_edge_map05": f"{map50:.4f}",
            "det_edge_map95": f"{map5095:.4f}",
            "det_edge_precision": f"{precision:.4f}",
            "det_edge_recall": f"{recall:.4f}",
            "det_edge_samples": str(len(gt_by_image)),

            # 第 5 章 推理速度（s05_speed.py）
            "speed_edge_latency": f"{total_mean:.3f}",
            "speed_edge_fps": (
                f"{1000.0 / total_mean:.2f}"
                if total_mean > 0
                else "0.00"
            ),
            "stage_edge_pre": f"{pre_mean:.3f}",
            "stage_edge_infer": f"{infer_mean:.3f}",
            "stage_edge_post": f"{post_mean:.3f}",
            "stage_edge_total": f"{total_mean:.3f}",
            "ratio_edge_pre": ratio(pre_mean),
            "ratio_edge_infer": ratio(infer_mean),
            "ratio_edge_post": ratio(post_mean),

            # 补充诊断指标（不用于渲染报告，供查阅）
            "f1": f1,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "inference_fps": (
                1000.0 / infer_mean
                if infer_mean > 0
                else 0.0
            ),
            "end_to_end_std_ms": total_std,
            "end_to_end_min_ms": min(total_values),
            "end_to_end_max_ms": max(total_values),

            "per_class": per_class,

            "notes": [
                "Precision/Recall/F1 使用指定 conf 阈值及 IoU=0.5。",
                "mAP 使用低置信度候选，按 IoU=0.50:0.05:0.95 计算。",
                "FPS 基于逐张图片单次计时的均值。",
                "本程序的 NMS 先于 AP 计算，不是完整 COCO 官方评测器。"
            ]
        }

        self.metrics = summary
        self.per_class_metrics = per_class
        self.timing_rows = timing_rows
        self.all_predictions = all_preds
        self.ground_truth_by_image = gt_by_image
        self._has_evaluated = True

        return summary

    # --------------------------------------------------------
    # 获取测评指标
    # --------------------------------------------------------

    def get_metrics(self):
        """获取测评结果。"""
        if not self._has_evaluated:
            raise RuntimeError(
                "尚未执行测评，请先调用 evaluate() 或 run()"
            )

        return self.metrics

    # --------------------------------------------------------
    # 导出结果
    # --------------------------------------------------------

    def export_results(self):
        """将测评结果导出为 JSON 和 CSV 文件。"""
        if not self._has_evaluated:
            raise RuntimeError(
                "尚未执行测评，请先调用 evaluate() 或 run()"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # 整体测评汇总
        json_path = (
            self.output_dir / "metrics_summary.json"
        )

        json_path.write_text(
            json.dumps(
                self.metrics,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        # 各类别指标
        class_csv = (
            self.output_dir / "per_class_metrics.csv"
        )

        fields = [
            "class_id",
            "class_name",
            "gt_count",
            "precision",
            "recall",
            "f1",
            "ap50",
            "ap50_95"
        ]

        with class_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )
            writer.writeheader()
            writer.writerows(self.per_class_metrics)

        # 每张图片的性能数据
        timing_csv = (
            self.output_dir / "per_image_timings.csv"
        )

        timing_fields = [
            "image",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "total_ms",
            "num_gt",
            "num_predictions"
        ]

        with timing_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=timing_fields
            )
            writer.writeheader()
            writer.writerows(self.timing_rows)

        return {
            "summary_json": str(json_path),
            "per_class_csv": str(class_csv),
            "per_image_timings_csv": str(timing_csv)
        }

    # --------------------------------------------------------
    # 打印测评报告
    # --------------------------------------------------------

    def print_report(self):
        """在终端打印测评摘要。"""
        summary = self.get_metrics()

        print("\n" + "=" * 60)
        print("ONNX 目标检测模型测评结果")
        print("=" * 60)

        print(f"测试图片数       : {summary['num_images']}")
        print(
            f"真实目标数       : "
            f"{summary['num_ground_truth_objects']}"
        )

        print(
            f"TP / FP / FN     : "
            f"{summary['tp']} / {summary['fp']} / {summary['fn']}"
        )

        print(f"Precision        : {summary['det_edge_precision']}")
        print(f"Recall           : {summary['det_edge_recall']}")
        print(f"F1-score         : {summary['f1']:.6f}")
        print(f"mAP@0.5          : {summary['det_edge_map05']}")
        print(f"mAP@0.5:0.95     : {summary['det_edge_map95']}")

        print(
            f"预处理均值       : "
            f"{summary['stage_edge_pre']} ms/image"
        )
        print(
            f"推理均值         : "
            f"{summary['stage_edge_infer']} ms/image"
        )
        print(
            f"后处理均值       : "
            f"{summary['stage_edge_post']} ms/image"
        )
        print(
            f"端到端均值       : "
            f"{summary['stage_edge_total']} ms/image"
        )
        print(f"端到端 FPS       : {summary['speed_edge_fps']}")
        print(f"推理 FPS         : {summary['inference_fps']:.2f}")

        print("\n类别指标：")

        for item in self.per_class_metrics:
            if item["ap50"] is None:
                print(
                    f"  {item['class_name']}: "
                    "无 GT，AP 不参与 mAP"
                )
            else:
                print(
                    f"  {item['class_name']}: "
                    f"P={item['precision']:.4f}, "
                    f"R={item['recall']:.4f}, "
                    f"F1={item['f1']:.4f}, "
                    f"AP50={item['ap50']:.4f}, "
                    f"AP50:95={item['ap50_95']:.4f}"
                )

    # --------------------------------------------------------
    # 一键执行
    # --------------------------------------------------------

    def run(self):
        """执行完整测评、导出结果并打印报告。"""
        self.evaluate()

        output_files = self.export_results()

        self.print_report()

        print("\n结果文件：")

        for label, path in output_files.items():
            print(f"  {label}: {path}")

        return self.metrics


# ============================================================
# 6. 程序入口
# ============================================================

if __name__ == "__main__":

    # 测评参数集中配置
    evaluator = YOLOv8ONNXEvaluator(
        model_path="./yolov8n.onnx",
        images_dir="../dataset_info/data/coco128/images/train2017",
        labels_dir="../dataset_info/data/coco128/labels/train2017",
        output_dir="../output",

        image_size=640,
        conf_threshold=0.25,
        nms_iou_threshold=0.45,
        max_detections=300,

        providers=["CPUExecutionProvider"],

        # 可选：每行一个类别名称
        class_names_file="",

        warmup_rounds=3
    )

    # 执行完整测评
    evaluator.run()
