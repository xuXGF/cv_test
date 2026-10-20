"""ONNX 模型信息采集与推理基准测试模块。

输出的信息用于填写测评报告：

* ``collect_model_info()``：模型元数据，对应报告 1.2 节「算法信息」、
  1.4 节「测试环境信息」与 3.1 节「模型配置」中的模型字段。
* ``benchmark()``：预处理/推理/后处理耗时与 FPS 统计，
  对应报告第 5 章「推理速度评测」。

用法::

    from model import collect_model_info, load_session, benchmark

    info = collect_model_info()  # 缺省使用包内自带 yolov8n.onnx
    print(info.model_name, info.param_count, info.input_size)

    session, input_hw = load_session("model/yolov8n.onnx")
    result = benchmark(session, image_bgr, warmup=3, rounds=10)
    print(result.total_ms, result.fps)

说明：

* 采集不到的字段保持空字符串（例如 FLOPs），不自动填入虚假值。
* 字段名与 ``report_generator.sections`` 中对应章节字段保持一致。
"""

from __future__ import annotations

from .info import (
    ModelInfo,
    collect_model_info,
    default_model_path,
    default_test_image_path,
)
from .yolov8_onnx_infer import (
    BenchmarkResult,
    StageTimings,
    benchmark,
    draw_detections,
    load_session,
    run_inference,
)

__all__ = [
    "ModelInfo",
    "collect_model_info",
    "default_model_path",
    "default_test_image_path",
    "BenchmarkResult",
    "StageTimings",
    "benchmark",
    "draw_detections",
    "load_session",
    "run_inference",
]
