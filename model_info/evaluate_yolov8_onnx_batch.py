"""ONNX 模型 Batch Size 推理性能测试（报告 5.4）。

复用 evaluate_yolov8_onnx_manual.py 的模型加载与前/后处理，
对 Batch Size = 1 / 2 / 4 测量：
- 平均耗时(ms)：单张图片平均端到端耗时（单批总耗时 ÷ 批内图片数）
- FPS：吞吐 = 1000 / 单张平均耗时
- CPU 利用率 / GPU(NPU) 利用率 / 内存占用：测试期间后台采样取均值

模型输入 batch 维固定为 1 时，"单批"按连续单张推理计时；
支持动态 batch 时整批一次推理。

运行方式（需在 model_info 目录下执行）：
    python3 evaluate_yolov8_onnx_batch.py
"""

import json
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import psutil

from evaluate_yolov8_onnx_manual import (
    IMAGE_SUFFIXES,
    YOLOv8ONNXEvaluator,
    decode_output,
    preprocess,
)

BATCH_SIZES = (1, 2, 4)


# ============================================================
# 1. 资源占用监控
# ============================================================

class ResourceMonitor:
    """测试期间后台采样 CPU / 内存 / GPU 利用率，stop 时返回均值。

    GPU 采样按平台选择：
    - NVIDIA：pynvml
    - macOS：ioreg 读取 Apple GPU 的 Device Utilization %
    - 其他平台 / 采集失败：返回 None，对应字段留空，不填虚假值
    """

    SAMPLE_INTERVAL_S = 0.2

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._cpu: list[float] = []
        self._mem: list[float] = []
        self._gpu: list[float] = []
        self._gpu_sampler: Callable[[], float | None] | None = (
            self._make_gpu_sampler()
        )
        # psutil 首次调用仅建立基线，返回值忽略
        psutil.cpu_percent(interval=None)

    @staticmethod
    def _make_gpu_sampler() -> Callable[[], float | None] | None:
        """返回无参 GPU 利用率采样函数（0-100），无法采集时返回 None。"""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            def sample_nvidia() -> float:
                return float(
                    pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                )

            return sample_nvidia
        except Exception:
            # 非 NVIDIA 平台 pynvml 初始化失败属预期，转 macOS 采样
            pass

        if sys.platform != "darwin":
            return None

        def sample_apple() -> float | None:
            try:
                out = subprocess.run(
                    ["ioreg", "-r", "-d", "1", "-w", "0",
                     "-c", "IOAccelerator"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                return None
            m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
            return float(m.group(1)) if m else None

        return sample_apple

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._cpu.append(psutil.cpu_percent(interval=None))
            self._mem.append(psutil.virtual_memory().percent)
            if self._gpu_sampler is not None:
                value = self._gpu_sampler()
                if value is not None:
                    self._gpu.append(value)
            self._stop_event.wait(self.SAMPLE_INTERVAL_S)

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 1) if values else None

    def stop(self) -> dict:
        """停止采样并返回平均值，无法采集的项为 None。"""
        self._stop_event.set()
        self._thread.join()
        return {
            "cpu": self._mean(self._cpu),
            "gpu": self._mean(self._gpu),
            "mem": self._mean(self._mem),
        }


# ============================================================
# 2. 批量推理计时
# ============================================================

def is_dynamic_batch(evaluator: YOLOv8ONNXEvaluator) -> bool:
    """判断模型输入 batch 维是否为动态（非固定整数）。"""
    batch_dim = evaluator.session.get_inputs()[0].shape[0]
    return not isinstance(batch_dim, int)


def list_images(images_dir: Path) -> list[Path]:
    """按文件名排序返回目录下全部可用图片路径。"""
    images = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"图片目录中没有可用图片：{images_dir}")
    return images


def run_one_batch(
    evaluator: YOLOv8ONNXEvaluator,
    images: list[np.ndarray],
    dynamic_batch: bool,
) -> float:
    """对一批图片执行端到端推理，返回该批耗时（ms）。

    动态 batch 时整批一次推理；否则批内连续单张推理。
    与 evaluate_yolov8_onnx_manual 一致，后处理使用低置信度候选。
    """
    tensors = []
    metas = []

    start = time.perf_counter()

    for image in images:
        tensor, meta, _ = preprocess(image, evaluator.input_hw)
        tensors.append(tensor)
        metas.append(meta)

    if dynamic_batch:
        outputs = evaluator.session.run(
            None,
            {evaluator.input_name: np.concatenate(tensors, axis=0)},
        )
        for i, meta in enumerate(metas):
            decode_output(
                [out[i:i + 1] for out in outputs],
                meta,
                conf_threshold=0.001,
                iou_threshold=evaluator.nms_iou_threshold,
                max_det=evaluator.max_detections,
            )
    else:
        for tensor, meta in zip(tensors, metas):
            outputs = evaluator.session.run(
                None, {evaluator.input_name: tensor}
            )
            decode_output(
                outputs,
                meta,
                conf_threshold=0.001,
                iou_threshold=evaluator.nms_iou_threshold,
                max_det=evaluator.max_detections,
            )

    return (time.perf_counter() - start) * 1000.0


def benchmark_batch_size(
    evaluator: YOLOv8ONNXEvaluator,
    images: list[np.ndarray],
    batch_size: int,
    dynamic_batch: bool,
    warmup_batches: int = 3,
) -> dict:
    """测量指定 batch size 的耗时 / 吞吐 / 资源占用。

    仅使用完整批次（不足一批的图片跳过）。
    """
    usable = len(images) - len(images) % batch_size
    if usable < batch_size:
        raise RuntimeError(
            f"图片数量不足以组成一个 batch_size={batch_size} 的批次"
        )

    groups = [
        images[i:i + batch_size]
        for i in range(0, usable, batch_size)
    ]

    for group in groups[:warmup_batches]:
        run_one_batch(evaluator, group, dynamic_batch)

    monitor = ResourceMonitor()
    monitor.start()
    batch_times_ms = []
    try:
        for group in groups:
            batch_times_ms.append(
                run_one_batch(evaluator, group, dynamic_batch)
            )
    finally:
        usage = monitor.stop()

    per_batch_ms = statistics.fmean(batch_times_ms)
    per_image_ms = per_batch_ms

    return {
        "latency": f"{per_image_ms:.3f}",
        "fps": f"{1000.0 / per_image_ms:.2f}",
        "cpu": f"{usage['cpu']}%" if usage["cpu"] is not None else "",
        "npu": f"{usage['gpu']}%" if usage["gpu"] is not None else "",
        "mem": f"{usage['mem']}%" if usage["mem"] is not None else "",
        "batch_count": len(groups),
    }


# ============================================================
# 3. 结果导出与程序入口
# ============================================================

def export_results(summary: dict) -> Path:
    """导出 batch 性能指标，字段名与 s05_speed.py 的 5.4 对齐。"""
    out_path = Path("../output/batch_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    # 测评参数与 evaluate_yolov8_onnx_manual.py 保持一致
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

        warmup_rounds=3
    )
    evaluator.load_model()

    dynamic_batch = is_dynamic_batch(evaluator)
    print(f"模型是否支持动态 batch：{dynamic_batch}")

    images = [
        cv2.imread(str(p))
        for p in list_images(evaluator.images_dir)
    ]
    if any(img is None for img in images):
        raise RuntimeError("存在读取失败的图片，请检查数据集")

    summary = {
        "model": str(evaluator.model_path.resolve()),
        "providers": evaluator.providers,
        "input_size": list(evaluator.input_hw),
        "num_images": len(images),
        "batch_execution": (
            "整批一次推理"
            if dynamic_batch
            else "模型 batch 维固定为 1，单批按连续单张推理计时"
        ),
    }

    for batch_size in BATCH_SIZES:
        print(f"\n===== Batch Size = {batch_size} =====")
        result = benchmark_batch_size(
            evaluator,
            images,
            batch_size,
            dynamic_batch,
            warmup_batches=evaluator.warmup_rounds,
        )
        prefix = f"bs{batch_size}"
        summary[f"{prefix}_latency"] = result["latency"]
        summary[f"{prefix}_fps"] = result["fps"]
        summary[f"{prefix}_cpu"] = result["cpu"]
        summary[f"{prefix}_npu"] = result["npu"]
        summary[f"{prefix}_mem"] = result["mem"]

        print(f"单批平均耗时  : {result['latency']} ms")
        print(f"吞吐 FPS      : {result['fps']}")
        print(f"CPU 利用率    : {result['cpu'] or '（未采集到）'}")
        print(f"GPU/NPU 利用率: {result['npu'] or '（未采集到）'}")
        print(f"内存占用      : {result['mem'] or '（未采集到）'}")

    summary["notes"] = [
        "平均耗时为单张图片平均端到端耗时（单批总耗时 ÷ 批内图片数，"
        "含预处理+推理+后处理）。",
        "FPS = 1000 / 平均耗时。",
        "CPU/内存占用为整机利用率；GPU 利用率取测试期间采样均值；"
        "Apple ANE 利用率系统未提供采集接口。",
    ]

    out_path = export_results(summary)

    print(f"\n结果文件：{out_path}")


if __name__ == "__main__":
    main()
