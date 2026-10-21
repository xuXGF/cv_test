"""ONNX 模型 Batch Size 推理性能测试（报告 5.4）。

复用 evaluate_yolov8_onnx_manual.py 的模型加载与前/后处理，
对 Batch Size = 1 / 2 / 4 测量：
- 平均耗时(ms)：单批端到端总耗时（预处理+推理+后处理）
- 分段耗时(ms)：单批内预处理 / 推理 / 后处理各自的平均耗时
- FPS：吞吐 = 批内图片数 / 单批耗时 × 1000
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

# 使脚本在 model_info 目录下直接运行时也能导入项目根目录的 tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import format_memory

from evaluate_yolov8_onnx_manual import (
    IMAGE_SUFFIXES,
    YOLOv8ONNXEvaluator,
    decode_output,
    preprocess,
)

BATCH_SIZES = (1, 4, 8)


# ============================================================
# 1. 资源占用监控
# ============================================================

class ResourceMonitor:
    """测试期间后台采样本进程 CPU / 内存占用与 GPU 利用率，stop 时返回均值。

    - CPU / 内存：仅统计本测试进程（CPU 按核数归一化到 0-100%；
      内存记录本进程占用的物理内存 RSS，单位 MB）
    - GPU 采样按平台选择（只能取设备级利用率，无法按进程拆分）：
      NVIDIA：pynvml；macOS：ioreg 读取 Apple GPU 的 Device Utilization %
      其他平台 / 采集失败：返回 None，对应字段留空，不填虚假值
    """

    SAMPLE_INTERVAL_S = 0.2
    # GPU 采样需 spawn ioreg 子进程，开销大；降频避免与推理争抢 CPU 干扰计时
    GPU_SAMPLE_EVERY_N = 10

    def __init__(self, sample_gpu: bool = True):
        self._stop_event = threading.Event()
        self._thread = None
        self._process = psutil.Process()
        self._cpu: list[float] = []
        self._mem: list[float] = []
        self._gpu: list[float] = []
        self._gpu_sampler: Callable[[], float | None] | None = (
            self._make_gpu_sampler() if sample_gpu else None
        )
        # psutil 首次调用仅建立基线，返回值忽略
        self._process.cpu_percent(interval=None)

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
        round_idx = 0
        while not self._stop_event.is_set():
            # 进程 CPU 占用：cpu_percent 为单核口径（多线程可超 100%），
            # 除以核数归一化为整机 0-100%
            proc_cpu = self._process.cpu_percent(interval=None)
            self._cpu.append(proc_cpu / psutil.cpu_count())
            self._mem.append(
                self._process.memory_info().rss / (1024 * 1024)  # MB
            )
            if (
                self._gpu_sampler is not None
                and round_idx % self.GPU_SAMPLE_EVERY_N == 0
            ):
                value = self._gpu_sampler()
                if value is not None:
                    self._gpu.append(value)
            round_idx += 1
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
) -> dict[str, float]:
    """对一批图片执行端到端推理，返回各阶段耗时（ms）。

    动态 batch 时整批一次推理；否则批内连续单张推理。
    与 evaluate_yolov8_onnx_manual 一致，后处理使用低置信度候选。
    返回键：preprocess / inference / postprocess / total。
    """
    tensors = []
    metas = []

    batch_start = time.perf_counter()

    start = time.perf_counter()
    for image in images:
        tensor, meta, _ = preprocess(image, evaluator.input_hw)
        tensors.append(tensor)
        metas.append(meta)
    preprocess_ms = (time.perf_counter() - start) * 1000.0

    if dynamic_batch:
        start = time.perf_counter()
        outputs = evaluator.session.run(
            None,
            {evaluator.input_name: np.concatenate(tensors, axis=0)},
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        for i, meta in enumerate(metas):
            decode_output(
                [out[i:i + 1] for out in outputs],
                meta,
                conf_threshold=0.001,
                iou_threshold=evaluator.nms_iou_threshold,
                max_det=evaluator.max_detections,
            )
        postprocess_ms = (time.perf_counter() - start) * 1000.0
    else:
        # 固定 batch=1 时推理与后处理逐张交替执行，逐段累加计时
        inference_ms = 0.0
        postprocess_ms = 0.0
        for tensor, meta in zip(tensors, metas):
            start = time.perf_counter()
            outputs = evaluator.session.run(
                None, {evaluator.input_name: tensor}
            )
            inference_ms += (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            decode_output(
                outputs,
                meta,
                conf_threshold=0.001,
                iou_threshold=evaluator.nms_iou_threshold,
                max_det=evaluator.max_detections,
            )
            postprocess_ms += (time.perf_counter() - start) * 1000.0

    return {
        "preprocess": preprocess_ms,
        "inference": inference_ms,
        "postprocess": postprocess_ms,
        "total": (time.perf_counter() - batch_start) * 1000.0,
    }


def benchmark_batch_size(
    evaluator: YOLOv8ONNXEvaluator,
    images: list[np.ndarray],
    batch_size: int,
    dynamic_batch: bool,
    warmup_batches: int = 3,
    sample_gpu: bool = True,
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

    monitor = ResourceMonitor(sample_gpu=sample_gpu)
    monitor.start()
    batch_times_ms = []
    stage_times: dict[str, list[float]] = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
    }
    try:
        for group in groups:
            timings = run_one_batch(evaluator, group, dynamic_batch)
            batch_times_ms.append(timings["total"])
            for stage, values in stage_times.items():
                values.append(timings[stage])
    finally:
        usage = monitor.stop()

    per_batch_ms = statistics.fmean(batch_times_ms)

    return {
        "latency": f"{per_batch_ms:.3f}",
        "pre_ms": f"{statistics.fmean(stage_times['preprocess']):.3f}",
        "infer_ms": f"{statistics.fmean(stage_times['inference']):.3f}",
        "post_ms": f"{statistics.fmean(stage_times['postprocess']):.3f}",
        "fps": f"{batch_size / (per_batch_ms / 1000.0):.2f}",
        "cpu": f"{usage['cpu']}%" if usage["cpu"] is not None else "",
        "npu": f"{usage['gpu']}%" if usage["gpu"] is not None else "",
        "mem": format_memory(usage["mem"]) if usage["mem"] is not None else "",
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

    # 模型跑在 CPU 上时 GPU 利用率无意义（只会采到系统其他进程的占用），不采样
    sample_gpu = any(
        p != "CPUExecutionProvider" for p in evaluator.providers
    )
    print(f"是否采样 GPU 利用率：{sample_gpu}")

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
            sample_gpu=sample_gpu,
        )
        prefix = f"bs{batch_size}"
        summary[f"{prefix}_latency"] = result["latency"]
        summary[f"{prefix}_pre"] = result["pre_ms"]
        summary[f"{prefix}_infer"] = result["infer_ms"]
        summary[f"{prefix}_post"] = result["post_ms"]
        summary[f"{prefix}_fps"] = result["fps"]
        summary[f"{prefix}_cpu"] = result["cpu"]
        summary[f"{prefix}_npu"] = result["npu"]
        summary[f"{prefix}_mem"] = result["mem"]

        print(f"单批平均耗时  : {result['latency']} ms")
        print(f"  预处理耗时  : {result['pre_ms']} ms")
        print(f"  推理耗时    : {result['infer_ms']} ms")
        print(f"  后处理耗时  : {result['post_ms']} ms")
        print(f"吞吐 FPS      : {result['fps']}")
        print(f"CPU 利用率    : {result['cpu'] or '（未采集到）'}")
        print(f"GPU/NPU 利用率: {result['npu'] or '（未采集到）'}")
        print(f"内存占用      : {result['mem'] or '（未采集到）'}")

    summary["notes"] = [
        "平均耗时为单批端到端总耗时（批内全部图片的预处理+推理+后处理）。",
        "分段耗时为单批内预处理/推理/后处理各自的平均耗时，"
        "三者之和约等于平均耗时（差值为计时打点开销）。",
        "FPS = 批内图片数 / 单批耗时 × 1000，即每秒处理图片数。",
        "CPU/内存占用为本测试进程占用（CPU 按核数归一化到 0-100%，"
        "内存为占用的物理内存 RSS，单位 MB/GB）。",
        "GPU 利用率为设备级采样均值，无法按进程拆分；"
        "模型仅使用 CPU 执行提供者时不采样、留空；"
        "Apple ANE 利用率系统未提供采集接口。",
    ]

    out_path = export_results(summary)

    print(f"\n结果文件：{out_path}")


if __name__ == "__main__":
    main()
