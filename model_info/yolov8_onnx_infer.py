#!/usr/bin/env python3


import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with aspect ratio preserved; return image, scale, (pad_left, pad_top)."""
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / h, new_shape[1] / w)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    pad_w = new_shape[1] - resized_w
    pad_h = new_shape[0] - resized_h

    # Match the common Ultralytics centered letterbox behavior.
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (left, top)


def preprocess(image, input_hw):
    """BGR HWC uint8 -> RGB NCHW float32 tensor, plus geometry for box restoration."""
    start = time.perf_counter()

    padded, scale, pad = letterbox(image, input_hw)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)

    elapsed_ms = (time.perf_counter() - start) * 1000
    meta = {"scale": scale, "pad": pad, "orig_hw": image.shape[:2]}
    return tensor, meta, elapsed_ms


def xywh_to_xyxy(boxes):
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms_xyxy(boxes, scores, iou_threshold):
    """Greedy NMS implemented in NumPy; returns indices into boxes."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
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

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-7)
        order = rest[iou <= iou_threshold]

    return keep


def postprocess(outputs, meta, conf_threshold=0.25, iou_threshold=0.45, max_det=300):
    """
    Decode standard YOLOv8 detection output.
    Returns detections as list of dicts: xyxy in original-image pixels, score, class_id.
    """
    start = time.perf_counter()

    pred = np.asarray(outputs[0])
    # Remove batch dimension.
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape: {np.asarray(outputs[0]).shape}")

    # YOLOv8 commonly emits [channels, candidates], e.g. [84, 8400].
    # Convert to [candidates, channels] when the first dimension looks like channels.
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 512:
        pred = pred.T

    if pred.shape[1] < 5:
        raise ValueError(
            f"Unexpected detection output shape {pred.shape}; expected boxes plus class scores."
        )

    # Standard YOLOv8 detection output: cx, cy, w, h, class scores (no objectness).
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return [], (time.perf_counter() - start) * 1000

    boxes = xywh_to_xyxy(boxes_xywh)

    # Undo letterbox padding and scale back to original image dimensions.
    scale = meta["scale"]
    pad_x, pad_y = meta["pad"]
    orig_h, orig_w = meta["orig_hw"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

    # Class-aware NMS: detections from different classes do not suppress each other.
    keep_all = []
    for cls in np.unique(class_ids):
        cls_indices = np.flatnonzero(class_ids == cls)
        cls_keep_local = nms_xyxy(boxes[cls_indices], scores[cls_indices], iou_threshold)
        keep_all.extend(cls_indices[cls_keep_local].tolist())

    keep_all = sorted(keep_all, key=lambda i: float(scores[i]), reverse=True)[:max_det]
    detections = [
        {
            "xyxy": boxes[i].round(2).tolist(),
            "score": float(scores[i]),
            "class_id": int(class_ids[i]),
        }
        for i in keep_all
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return detections, elapsed_ms


def draw_detections(image, detections, class_names=None):
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        cls_id = det["class_id"]
        label_name = (
            class_names[cls_id]
            if class_names is not None and cls_id < len(class_names)
            else str(cls_id)
        )
        label = f"{label_name} {det['score']:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(output, (x1, y_text), (x1 + tw + 4, y_text + th + 6), (0, 255, 0), -1)
        cv2.putText(
            output, label, (x1 + 2, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )
    return output


def _session_input_hw(session, default: int = 640) -> tuple[int, int]:
    """从会话输入形状解析静态 (H, W)，动态维度回退到 default。"""
    shape = session.get_inputs()[0].shape  # Usually [N, C, H, W], possibly dynamic.
    h = shape[2] if isinstance(shape[2], int) else default
    w = shape[3] if isinstance(shape[3], int) else default
    return int(h), int(w)


@dataclass
class StageTimings:
    """单次推理各阶段耗时（毫秒）。"""

    preprocess_ms: float  # 预处理
    inference_ms: float  # 模型推理
    postprocess_ms: float  # 后处理

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class BenchmarkResult:
    """多次重复推理的统计结果，字段对应报告第 5 章「推理速度评测」。"""

    warmup: int  # 预热次数
    rounds: int  # 正式测试次数
    preprocess_ms: float  # 预处理耗时均值(ms)
    inference_ms: float  # 推理耗时均值(ms)
    postprocess_ms: float  # 后处理耗时均值(ms)
    total_ms: float  # 端到端总耗时均值(ms)
    total_std_ms: float  # 端到端总耗时标准差(ms)
    total_min_ms: float  # 端到端总耗时最小值(ms)
    total_max_ms: float  # 端到端总耗时最大值(ms)
    fps: float  # 端到端 FPS（含预处理/后处理）
    infer_fps: float  # 仅模型推理 FPS
    num_detections: int  # 末次推理检测数量
    detections: list  # 末次推理检测详情


def load_session(model_path, providers=("CPUExecutionProvider",)):
    """创建推理会话，返回 (session, 输入尺寸 (H, W))。"""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    available = ort.get_available_providers()
    chosen = [p for p in providers if p in available]
    if not chosen:
        raise RuntimeError(
            f"Requested providers {list(providers)} unavailable. Available: {available}"
        )

    session = ort.InferenceSession(str(model_path), providers=chosen)
    return session, _session_input_hw(session)


def run_inference(session, image, conf=0.25, iou=0.45, max_det=300, input_hw=None):
    """执行一次完整推理（预处理 → session.run → 后处理）。

    返回 (detections, StageTimings)。
    """
    if input_hw is None:
        input_hw = _session_input_hw(session)

    # 1) Preprocessing
    tensor, meta, preprocess_ms = preprocess(image, input_hw)

    # 2) Model inference
    infer_start = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    inference_ms = (time.perf_counter() - infer_start) * 1000

    # 3) Postprocessing
    detections, postprocess_ms = postprocess(outputs, meta, conf, iou, max_det)
    return detections, StageTimings(preprocess_ms, inference_ms, postprocess_ms)


def benchmark(session, image, warmup=3, rounds=10, conf=0.25, iou=0.45, max_det=300,
              input_hw=None) -> BenchmarkResult:
    """重复基准测试：warmup 次预热不计入统计，取 rounds 次的均值/标准差/极值。"""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    for _ in range(warmup):
        run_inference(session, image, conf, iou, max_det, input_hw)

    timings = []
    detections: list = []
    for _ in range(rounds):
        detections, timing = run_inference(session, image, conf, iou, max_det, input_hw)
        timings.append(timing)

    totals = [t.total_ms for t in timings]
    return BenchmarkResult(
        warmup=warmup,
        rounds=rounds,
        preprocess_ms=mean(t.preprocess_ms for t in timings),
        inference_ms=mean(t.inference_ms for t in timings),
        postprocess_ms=mean(t.postprocess_ms for t in timings),
        total_ms=mean(totals),
        total_std_ms=stdev(totals) if rounds > 1 else 0.0,
        total_min_ms=min(totals),
        total_max_ms=max(totals),
        fps=1000.0 / mean(totals),
        infer_fps=1000.0 / mean(t.inference_ms for t in timings),
        num_detections=len(detections),
        detections=detections,
    )
