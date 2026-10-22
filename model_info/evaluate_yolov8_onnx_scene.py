
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX 边缘场景精度测评程序（独立脚本）

参考 evaluate_yolov8_onnx_manual.py 实现，专门用于报告 4.5 节
「边缘场景精度」：对边缘场景数据子集逐场景实测精度指标。

与原脚本的功能区分：

1. 原脚本（evaluate_yolov8_onnx_manual.py）：评测主数据集，
   导出 output/metrics_summary.json（报告 4.3 / 5 章数据源），
   本脚本不修改其任何代码与导出结果。
2. 本脚本：直接导入原脚本的 YOLOv8ONNXEvaluator 复用同一套
   模型加载、预处理、后处理与指标计算逻辑（与 4.3 节边缘端
   量化模型口径一致），仅切换数据目录逐场景评测，结果单独
   导出到 output/scene_metrics.json。

场景清单驱动：

* 场景列表、场景描述（desc）与云端模型逐场景指标（cloud_*）
  均来自 ``local_info/local_info.json`` 的 ``scene`` 字段，
  ``path`` / ``desc`` / 各 ``cloud_*`` 列表按下标一一对应。
* 导出的每个场景条目包含：场景目录名（scene）、目录绝对路径
  （path）、场景描述（desc）、边缘端实测指标（det_edge_*），
  以及原样同步的云端指标（cloud_*，仅存档供查阅，不渲染进
  报告 Markdown）。

场景数据目录布局（与 dataset_info 中 2.2 节场景一致）：
    dataset_info/data/scene/
      <场景目录>/images/*.jpg
      <场景目录>/labels/*.txt

依赖：
    pip install onnxruntime opencv-python numpy
"""

import json
from pathlib import Path

from evaluate_yolov8_onnx_manual import YOLOv8ONNXEvaluator

# 项目根目录（model_info 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 测评参数与 evaluate_yolov8_onnx_manual.py 的 __main__ 配置保持一致
# （路径统一基于本脚本位置解析，不依赖运行时工作目录）
_MODEL_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _MODEL_DIR / "yolov8n.onnx"
LOCAL_INFO_JSON = _PROJECT_ROOT / "local_info" / "local_info.json"
OUTPUT_DIR = _PROJECT_ROOT / "output"
OUTPUT_JSON = OUTPUT_DIR / "scene_metrics.json"

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300
PROVIDERS = ["CPUExecutionProvider"]
WARMUP_ROUNDS = 3

#: 每个场景导出的边缘端实测字段（与 report_generator/sections/s04_accuracy.py 4.5 节对应）
_SCENE_FIELDS = (
    "det_edge_map05",
    "det_edge_map95",
    "det_edge_precision",
    "det_edge_recall",
    "det_edge_samples",
)

#: 云端模型逐场景指标字段（local_info 手填，原样同步到导出，不渲染进报告）
_CLOUD_FIELDS = (
    "cloud_map05",
    "cloud_map95",
    "cloud_precision",
    "cloud_recall",
)


def load_scene_config() -> dict:
    """读取 ``local_info.json`` 的 ``scene`` 配置。

    文件不存在、缺少 ``scene.path`` 场景清单时抛出带路径的异常，
    不做静默兜底。

    :return: ``scene`` 字段（含 path/desc 及各 cloud_* 指标列表）。
    """
    if not LOCAL_INFO_JSON.is_file():
        raise FileNotFoundError(f"本地信息文件不存在：{LOCAL_INFO_JSON}")

    config = json.loads(
        LOCAL_INFO_JSON.read_text(encoding="utf-8")
    )

    scene = config.get("scene")

    if not isinstance(scene, dict) or not scene.get("path"):
        raise ValueError(
            f"local_info.json 缺少 scene.path 场景清单：{LOCAL_INFO_JSON}"
        )

    return scene


def resolve_scene_items(scene: dict) -> list[dict]:
    """按 ``local_info`` 场景清单解析待评测场景。

    ``path`` / ``desc`` / 各 ``cloud_*`` 列表按下标一一对应，
    数量不一致时抛出异常；重复场景地址只评测一次（保留首个下标
    的描述与云端指标）。

    :return: 场景条目列表，每项含 scene_dir（Path）、desc 与 cloud 字典。
    """
    paths = scene["path"]
    if not isinstance(paths, list):
        raise TypeError(f"scene.path 应为列表，实际为: {type(paths).__name__}")

    descs = scene.get("desc", [])
    if descs and len(descs) != len(paths):
        raise ValueError(
            f"scene.path 与 scene.desc 数量不一致: {len(paths)} != {len(descs)}"
        )

    cloud_values: dict[str, list] = {}
    for field in _CLOUD_FIELDS:
        values = scene.get(field, [])
        if values and len(values) != len(paths):
            raise ValueError(
                f"scene.path 与 {field} 数量不一致: {len(paths)} != {len(values)}"
            )
        cloud_values[field] = values

    items: list[dict] = []
    seen: set[Path] = set()

    for i, raw in enumerate(paths):
        scene_dir = Path(raw)

        if not scene_dir.is_absolute():
            scene_dir = _PROJECT_ROOT / scene_dir
        scene_dir = scene_dir.resolve()

        if scene_dir in seen:
            continue
        seen.add(scene_dir)

        if not scene_dir.is_dir():
            raise FileNotFoundError(f"边缘场景目录不存在：{scene_dir}")
        if not (scene_dir / "images").is_dir() or not (scene_dir / "labels").is_dir():
            raise FileNotFoundError(
                f"边缘场景目录缺少 images/ 或 labels/ 子目录：{scene_dir}"
            )

        items.append({
            "scene_dir": scene_dir,
            "desc": descs[i] if i < len(descs) else "",
            "cloud": {
                field: (
                    cloud_values[field][i]
                    if i < len(cloud_values[field])
                    else ""
                )
                for field in _CLOUD_FIELDS
            },
        })

    if not items:
        raise RuntimeError("local_info scene.path 中没有可用场景")

    return items


def evaluate_scenes(scene_items: list[dict]) -> list[dict]:
    """逐场景评测并汇总精度指标。

    复用同一个评测器实例（模型会话只加载一次），按场景切换
    images_dir / labels_dir；各场景与原脚本主数据集共用同一套
    预处理、后处理与指标计算流程。

    :param scene_items: resolve_scene_items() 返回的场景条目列表。
    :return: 场景指标列表，每项含 scene/path/desc、det_edge_* 与 cloud_* 字段。
    """
    first = scene_items[0]["scene_dir"]

    evaluator = YOLOv8ONNXEvaluator(
        model_path=str(_MODEL_PATH),
        images_dir=first / "images",
        labels_dir=first / "labels",
        output_dir=str(OUTPUT_DIR),
        image_size=IMAGE_SIZE,
        conf_threshold=CONF_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        max_detections=MAX_DETECTIONS,
        providers=PROVIDERS,
        warmup_rounds=WARMUP_ROUNDS,
    )

    results: list[dict] = []

    for index, item in enumerate(scene_items):
        scene_dir = item["scene_dir"]

        if index > 0:
            # 复用已加载的模型会话，仅切换评测数据目录
            evaluator.images_dir = scene_dir / "images"
            evaluator.labels_dir = scene_dir / "labels"

        print(f"\n===== 边缘场景 {scene_dir.name} =====")
        summary = evaluator.evaluate()

        entry: dict = {
            "scene": scene_dir.name,
            "path": str(scene_dir),
            "desc": item["desc"],
        }
        entry.update({field: summary[field] for field in _SCENE_FIELDS})
        entry.update(item["cloud"])
        results.append(entry)

    return results


def export_results(results: list[dict]) -> Path:
    """将逐场景指标导出为 JSON 文件，返回文件路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps({"scenes": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return OUTPUT_JSON


def print_summary(results: list[dict]) -> None:
    """在终端打印逐场景精度摘要。"""
    print("\n" + "=" * 60)
    print("边缘场景精度测评结果")
    print("=" * 60)

    for item in results:
        print(
            f"{item['scene']}: "
            f"mAP@0.5={item['det_edge_map05']}, "
            f"mAP@0.5:0.95={item['det_edge_map95']}, "
            f"P={item['det_edge_precision']}, "
            f"R={item['det_edge_recall']}, "
            f"样本数={item['det_edge_samples']}"
        )


def main() -> None:
    scene = load_scene_config()
    scene_items = resolve_scene_items(scene)

    print(f"场景清单来源：{LOCAL_INFO_JSON}")
    print(f"场景数量：{len(scene_items)}")

    results = evaluate_scenes(scene_items)
    json_path = export_results(results)

    print_summary(results)

    print(f"\n结果文件：{json_path}")


if __name__ == "__main__":
    main()
