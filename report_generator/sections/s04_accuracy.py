"""第 4 章：算法精度评测。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from tools import index_scene_metrics

from ..render import render

#: 项目根目录（sections -> report_generator -> 项目根），用于解析场景相对路径
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TEMPLATE = r"""# 4. 算法精度评测

## 4.1 精度评测指标

根据模型任务类型选择对应的精度评价指标。

| 任务类型 | 测评指标           | 指标说明   | 目标要求   |
| ---- | -------------- | ------ | ------ |
| 分类任务 | Top-1 Accuracy | {{metric_top1_desc}} | {{metric_top1_target}} |
| 分类任务 | Top-5 Accuracy | {{metric_top5_desc}} | {{metric_top5_target}} |
| 检测任务 | mAP\@0.5       | {{metric_map05_desc}} | {{metric_map05_target}} |
| 检测任务 | mAP\@0.5:0.95  | {{metric_map95_desc}} | {{metric_map95_target}} |
| 分割任务 | IoU            | {{metric_iou_desc}} | {{metric_iou_target}} |
| 分割任务 | mIoU           | {{metric_miou_desc}} | {{metric_miou_target}} |

## 4.2 分类任务精度

> 仅分类模型填写。

| 模型      | Top-1准确率 | Top-5准确率 |  测试样本数 |
| ------- | -------: | -------: | -----: |
| 云端原始模型  |   {{cls_cloud_top1}} |   {{cls_cloud_top5}} | {{cls_cloud_samples}} |
| 边缘端量化模型 |   {{cls_edge_top1}} |   {{cls_edge_top5}} | {{cls_edge_samples}} |

## 4.3 检测任务精度

> 仅检测模型填写。

| 模型      | mAP\@0.5 | mAP\@0.5:0.95 | Precision | Recall |  测试样本数 |
| ------- | -------: | ------------: | --------: | -----: | -----: |
| 云端原始模型  |   {{det_cloud_map05}} |        {{det_cloud_map95}} |    {{det_cloud_precision}} | {{det_cloud_recall}} | {{det_cloud_samples}} |
| 边缘端量化模型 |   {{det_edge_map05}} |        {{det_edge_map95}} |    {{det_edge_precision}} | {{det_edge_recall}} | {{det_edge_samples}} |

## 4.4 分割任务精度

> 仅分割模型填写。

| 模型      |    IoU |   mIoU | Pixel Accuracy |  测试样本数 |
| ------- | -----: | -----: | -------------: | -----: |
| 云端原始模型  | {{seg_cloud_iou}} | {{seg_cloud_miou}} |         {{seg_cloud_pixel_acc}} | {{seg_cloud_samples}} |
| 边缘端量化模型 | {{seg_edge_iou}} | {{seg_edge_miou}} |         {{seg_edge_pixel_acc}} | {{seg_edge_samples}} |

## 4.5 边缘场景精度

> 使用 2.2 节边缘场景数据，在边缘端量化模型上按 4.3 节相同口径实测。

| 场景编号 | 测试场景    | mAP\@0.5 | mAP\@0.5:0.95 | Precision | Recall |  测试样本数 |
| ---- | ------- | -------: | ------------: | --------: | -----: | -----: |
{{scene_acc_rows}}"""


@dataclass
class AccuracySection:
    """第 4 章：算法精度评测。"""

    # 4.1 精度评测指标
    metric_top1_desc: str = ""  # Top-1 Accuracy-指标说明
    metric_top1_target: str = ""  # Top-1 Accuracy-目标要求
    metric_top5_desc: str = ""  # Top-5 Accuracy-指标说明
    metric_top5_target: str = ""  # Top-5 Accuracy-目标要求
    metric_map05_desc: str = ""  # mAP@0.5-指标说明
    metric_map05_target: str = ""  # mAP@0.5-目标要求
    metric_map95_desc: str = ""  # mAP@0.5:0.95-指标说明
    metric_map95_target: str = ""  # mAP@0.5:0.95-目标要求
    metric_iou_desc: str = ""  # IoU-指标说明
    metric_iou_target: str = ""  # IoU-目标要求
    metric_miou_desc: str = ""  # mIoU-指标说明
    metric_miou_target: str = ""  # mIoU-目标要求

    # 4.2 分类任务精度
    cls_cloud_top1: str = ""  # 云端原始模型-Top-1准确率
    cls_cloud_top5: str = ""  # 云端原始模型-Top-5准确率
    cls_cloud_samples: str = ""  # 云端原始模型-测试样本数
    cls_edge_top1: str = ""  # 边缘端量化模型-Top-1准确率
    cls_edge_top5: str = ""  # 边缘端量化模型-Top-5准确率
    cls_edge_samples: str = ""  # 边缘端量化模型-测试样本数

    # 4.3 检测任务精度
    det_cloud_map05: str = ""  # 云端原始模型-mAP@0.5
    det_cloud_map95: str = ""  # 云端原始模型-mAP@0.5:0.95
    det_cloud_precision: str = ""  # 云端原始模型-Precision
    det_cloud_recall: str = ""  # 云端原始模型-Recall
    det_cloud_samples: str = ""  # 云端原始模型-测试样本数
    det_edge_map05: str = ""  # 边缘端量化模型-mAP@0.5
    det_edge_map95: str = ""  # 边缘端量化模型-mAP@0.5:0.95
    det_edge_precision: str = ""  # 边缘端量化模型-Precision
    det_edge_recall: str = ""  # 边缘端量化模型-Recall
    det_edge_samples: str = ""  # 边缘端量化模型-测试样本数

    # 4.4 分割任务精度
    seg_cloud_iou: str = ""  # 云端原始模型-IoU
    seg_cloud_miou: str = ""  # 云端原始模型-mIoU
    seg_cloud_pixel_acc: str = ""  # 云端原始模型-Pixel Accuracy
    seg_cloud_samples: str = ""  # 云端原始模型-测试样本数
    seg_edge_iou: str = ""  # 边缘端量化模型-IoU
    seg_edge_miou: str = ""  # 边缘端量化模型-mIoU
    seg_edge_pixel_acc: str = ""  # 边缘端量化模型-Pixel Accuracy
    seg_edge_samples: str = ""  # 边缘端量化模型-测试样本数

    # 4.5 边缘场景精度（由 fill_scene_accuracy 按 2.2 节场景清单生成，不手填）
    scene_acc_rows: str = ""  # 场景精度表格行（每行：编号/场景/mAP/P/R/样本数）

    def fill_scene_accuracy(self) -> None:
        """4.5 边缘场景精度：按 2.2 节场景顺序填充边缘端量化模型实测指标。

        场景清单沿用 ``local_info.json`` 的 ``scene`` 字段的 ``path`` 列表
        （与 2.2 节同源）；各场景指标来自独立脚本
        evaluate_yolov8_onnx_scene.py 的导出（output/scene_metrics.json 的
        ``scenes`` 列表，复用 evaluate_yolov8_onnx_manual.py 的评测器按
        4.3 节相同口径实测），按场景目录绝对路径与场景地址匹配。
        场景未配置时表格为空；已配置的场景缺少实测指标时抛出异常，
        不做静默兜底。
        """
        scene = getattr(self, "scene", None)
        if not scene or not scene.get("path"):
            return
        if not isinstance(scene, dict):
            raise TypeError(
                f"local_info 的 scene 应为对象（含 path/desc 列表），实际为: {type(scene).__name__}"
            )
        paths = scene.get("path", [])
        if not isinstance(paths, list):
            raise TypeError("scene.path 应为列表")
        if not paths:
            return

        scenes = getattr(self, "scenes", None)
        if scenes is None:
            raise ValueError(
                "4.5 边缘场景精度缺少实测数据（output/scene_metrics.json 中无 scenes 字段），"
                "请先运行 evaluate_yolov8_onnx_scene.py 生成场景测评结果"
            )
        if not isinstance(scenes, list):
            raise TypeError(f"测评导出的 scenes 应为列表，实际为: {type(scenes).__name__}")

        metrics_by_path = index_scene_metrics(scenes)

        rows: list[str] = []
        for i, path in enumerate(paths, start=1):
            scene_dir = Path(path)
            if not scene_dir.is_absolute():
                scene_dir = _PROJECT_ROOT / scene_dir
            metrics = metrics_by_path.get(str(scene_dir.resolve()))
            if metrics is None:
                raise ValueError(
                    f"4.5 边缘场景精度缺少场景 {i}（{path}）的实测指标，"
                    "请检查 local_info.json 的 scene.path 与测评导出的 scenes 是否对应"
                )
            rows.append(
                f"| {i:02d} | **场景{i}** "
                f"| {metrics['det_edge_map05']} | {metrics['det_edge_map95']} "
                f"| {metrics['det_edge_precision']} | {metrics['det_edge_recall']} "
                f"| {metrics['det_edge_samples']} |"
            )
        self.scene_acc_rows = "\n".join(rows)

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))