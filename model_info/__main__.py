"""命令行入口：``python -m model`` 打印模型元数据与推理基准测试结果。

用法::

    python -m model                      # 使用包内自带 yolov8n.onnx 与 test.
    python -m model <模型路径> [图片路径]   # 指定模型与测试图片
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import cv2

from . import benchmark, collect_model_info, draw_detections, load_session
from .info import default_model_path, default_test_image_path

_SAVE_NAME = "result.jpg"  # 可视化结果保存文件名（保存到当前目录）


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model_path()
    image_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_test_image_path()

    # 1) 模型元数据
    info = collect_model_info(model_path)
    for key, value in asdict(info).items():
        print(f"{key}: {value}")

    # 2) 推理基准测试
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    session, input_hw = load_session(info.model_path)
    result = benchmark(session, image)

    def ratio(ms: float) -> str:
        return f"{ms / result.total_ms * 100:.1f}%"

    print("=" * 52)
    print(
        f"Benchmark:      warmup={result.warmup} rounds={result.rounds} "
        f"image={image_path.name} input={input_hw}"
    )
    print(f"Preprocess:     {result.preprocess_ms:.3f} ms  ({ratio(result.preprocess_ms)})")
    print(
        f"Inference:      {result.inference_ms:.3f} ms  ({ratio(result.inference_ms)})"
        f"  fps={result.infer_fps:.2f}"
    )
    print(f"Postprocess:    {result.postprocess_ms:.3f} ms  ({ratio(result.postprocess_ms)})")
    print(
        f"Total:          {result.total_ms:.3f} ms  "
        f"std={result.total_std_ms:.3f}  min={result.total_min_ms:.3f}  "
        f"max={result.total_max_ms:.3f}"
    )
    print(f"FPS (端到端):   {result.fps:.2f}")
    print(f"Detections:     {result.num_detections}")
    print("=" * 52)

    # 3) 检测明细
    for det in result.detections:
        print(
            f"class={det['class_id']:>3} score={det['score']:.4f} "
            f"xyxy={det['xyxy']}"
        )

    # 4) 可视化结果
    save_path = Path(_SAVE_NAME).resolve()
    rendered = draw_detections(image, result.detections)
    if not cv2.imwrite(str(save_path), rendered):
        raise IOError(f"无法保存结果图: {save_path}")
    print(f"Saved visualization: {save_path}")


if __name__ == "__main__":
    main()
