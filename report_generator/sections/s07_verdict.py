"""第 7 章：评测结果判定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from tools import index_scene_metrics, pct, to_float

from ..render import render

#: 项目根目录（sections -> report_generator -> 项目根），用于解析场景相对路径
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TEMPLATE = r"""# 7. 评测结果判定

## 7.1 精度指标达标情况

| 指标       |   云端原始值 |    目标值 |    实测值 |   精度损失 | 是否达标   |
| -------- | -----: | -----: | -----: | -----: | ------ |
| mAP\@0.5  | {{v_map05_cloud}} | {{v_map05_target}} | {{v_map05_actual}} | {{v_map05_loss}} | {{v_map05_ok}} |
| Precision | {{v_precision_cloud}} | {{v_precision_target}} | {{v_precision_actual}} | {{v_precision_loss}} | {{v_precision_ok}} |
| Recall    | {{v_recall_cloud}} | {{v_recall_target}} | {{v_recall_actual}} | {{v_recall_loss}} | {{v_recall_ok}} |

## 7.2 推理性能指标统计

| 指标           |   云端原始值 |    目标值 |    实测值 | 是否达标   |
| ------------ | -----: | -----: | -----: | ------ |
| 端到端耗时(ms) | {{v_latency_cloud}} | {{v_latency_target}} | {{v_latency_actual}} | {{v_latency_ok}} |
| FPS          | {{v_fps_cloud}} | {{v_fps_target}} | {{v_fps_actual}} | {{v_fps_ok}} |

## 7.3 场景精度损失

| 场景 | 原始模型-正常数据集 | 原始模型-场景 | 原始模型差距 | 边端模型-正常数据集 | 边端模型-场景 | 边端模型差距 | 是否达标 |
| ---- | ----: | ----: | ----: | ----: | ----: | ----: | ---- |
{{scene_loss_rows}}

## 7.4 评测结果综合判定

**精度指标：** {{verdict_accuracy=\[达标/不达标]}}

**推理性能：** {{verdict_perf=\[达标/不达标]}}

**精度损失：** {{verdict_loss=\[满足/不满足要求]}}

**边缘场景泛化：** {{verdict_generalization=\[满足/不满足要求]}}

**综合判定：** {{verdict_overall=\[通过/需优化]}}

**判定说明：**

{{ana_74=\[填写综合评测结果及判定依据。]}}"""


@dataclass
class VerdictSection:
    """第 7 章：评测结果判定。"""

    # 7.1 精度指标达标情况（检测任务：mAP@0.5 / Precision / Recall）
    # 云端原始值来自 local_info.json（det_cloud_*），目标值手填（v_*_target），
    # 全部展示列由 fill_accuracy_rows 计算/回填
    v_map05_cloud: str = ""  # mAP@0.5-云端原始值
    v_map05_target: str = ""  # mAP@0.5-目标值（手填）
    v_map05_actual: str = ""  # mAP@0.5-实测值
    v_map05_loss: str = ""  # mAP@0.5-精度损失
    v_map05_ok: str = ""  # mAP@0.5-是否达标
    v_precision_cloud: str = ""  # Precision-云端原始值
    v_precision_target: str = ""  # Precision-目标值
    v_precision_actual: str = ""  # Precision-实测值
    v_precision_loss: str = ""  # Precision-精度损失
    v_precision_ok: str = ""  # Precision-是否达标
    v_recall_cloud: str = ""  # Recall-云端原始值
    v_recall_target: str = ""  # Recall-目标值
    v_recall_actual: str = ""  # Recall-实测值
    v_recall_loss: str = ""  # Recall-精度损失
    v_recall_ok: str = ""  # Recall-是否达标

    # 7.2 推理性能指标统计（端到端耗时 / FPS）
    # 云端原始值来自 local_info.json（speed_cloud_latency），目标值手填，
    # FPS 与判定由 fill_perf_rows 计算（FPS = 1000 / 端到端耗时）
    v_latency_cloud: str = ""  # 端到端耗时-云端原始值
    v_latency_target: str = ""  # 端到端耗时-目标值（手填，上限）
    v_latency_actual: str = ""  # 端到端耗时-实测值
    v_latency_ok: str = ""  # 端到端耗时-是否达标
    v_fps_cloud: str = ""  # FPS-云端原始值
    v_fps_target: str = ""  # FPS-目标值（手填，下限）
    v_fps_actual: str = ""  # FPS-实测值
    v_fps_ok: str = ""  # FPS-是否达标

    # 7.3 场景精度损失（由 fill_scene_loss 按场景生成，不手填）
    scene_loss_rows: str = ""  # 场景精度损失表格行（每行：场景/指标/正常与场景精度/差距/判定）

    # 7.4 评测结果综合判定
    verdict_accuracy: str = ""  # 精度指标（未填时默认「\[达标/不达标]」）
    verdict_perf: str = ""  # 推理性能（未填时默认「\[达标/不达标]」）
    verdict_loss: str = ""  # 精度损失（未填时默认「\[满足/不满足要求]」）
    verdict_generalization: str = ""  # 边缘场景泛化（未填时默认「\[满足/不满足要求]」）
    verdict_overall: str = ""  # 综合判定（未填时默认「\[通过/需优化]」）
    ana_74: str = ""  # 判定说明正文

    def fill_accuracy_rows(self) -> None:
        """7.1 检测精度达标情况：云端原始值/实测值/损失/判定由公式计算，不手填。

        目标值（v_*_target）为手填字段，取值口径见 local_info.json。
        输入字段由 run.py 注入：云端列（det_cloud_*）来自 local_info.json，
        边缘端列（det_edge_*）来自 metrics_summary.json。

        云端原始值 = local_info.json 中云端原始模型指标
        实测值 = 边缘端量化模型指标
        精度损失 = 云端原始模型指标 - 边缘端量化模型指标，
                   括号内为相对损失 = 精度损失 / 云端原始模型指标 × 100%
        是否达标 = 实测值 ≥ 目标值
        """
        rows = (
            ("map05", "det_cloud_map05", "det_edge_map05"),
            ("precision", "det_cloud_precision", "det_edge_precision"),
            ("recall", "det_cloud_recall", "det_edge_recall"),
        )

        for name, cloud_field, edge_field in rows:
            target = to_float(
                getattr(self, f"v_{name}_target", ""), f"v_{name}_target"
            )
            cloud = to_float(getattr(self, cloud_field, ""), cloud_field)
            edge = to_float(getattr(self, edge_field, ""), edge_field)

            loss = cloud - edge
            setattr(self, f"v_{name}_cloud", f"{cloud:.4f}")
            setattr(self, f"v_{name}_actual", f"{edge:.4f}")
            setattr(
                self,
                f"v_{name}_loss",
                f"{loss:.4f}（{pct(loss, cloud, cloud_field)}）",
            )
            setattr(self, f"v_{name}_ok", "是" if edge >= target else "否")

    def fill_perf_rows(self) -> None:
        """7.2 推理性能指标统计：FPS 与判定由公式计算，不手填。

        目标值（v_latency_target / v_fps_target）为手填字段，
        取值口径见 local_info.json。输入字段由 run.py 注入：
        云端端到端耗时（speed_cloud_latency）来自 local_info.json，
        边缘端端到端耗时（speed_edge_latency）来自 metrics_summary.json。

        FPS = 1000 / 端到端耗时(ms)
        端到端耗时达标 = 实测值 ≤ 目标值；FPS 达标 = 实测值 ≥ 目标值
        """
        cloud_latency = to_float(
            getattr(self, "speed_cloud_latency", ""), "speed_cloud_latency"
        )
        edge_latency = to_float(
            getattr(self, "speed_edge_latency", ""), "speed_edge_latency"
        )

        latency_target = to_float(
            getattr(self, "v_latency_target", ""), "v_latency_target"
        )
        fps_target = to_float(
            getattr(self, "v_fps_target", ""), "v_fps_target"
        )

        self.v_latency_cloud = f"{cloud_latency:.3f}"
        self.v_latency_actual = f"{edge_latency:.3f}"
        self.v_latency_ok = "是" if edge_latency <= latency_target else "否"

        self.v_fps_cloud = f"{1000.0 / cloud_latency:.2f}"
        self.v_fps_actual = f"{1000.0 / edge_latency:.2f}"
        self.v_fps_ok = "是" if 1000.0 / edge_latency >= fps_target else "否"

    def fill_scene_loss(self) -> None:
        """7.3 场景精度损失：逐场景对比正常数据集精度与场景精度的差距。

        正常数据集精度取 4.3 节 mAP@0.5（原始模型 det_cloud_map05、
        边端模型 det_edge_map05）；场景精度取 4.5 节场景实测
        （原始模型 cloud_map05、边端模型 det_edge_map05，来自
        output/scene_metrics.json）。场景顺序沿用 local_info.json 的
        scene.path，按场景目录绝对路径匹配。

        差距 =（场景精度 − 正常数据集精度）/ 正常数据集精度 × 100%，
        正值表示场景精度高于正常数据集（涨），负值表示低于正常数据集（降）；
        边端模型差距 ≥ 原始模型差距即判定达标，不设绝对阈值。
        场景未配置时表格为空；已配置场景缺少实测指标时抛出异常，
        不做静默兜底。
        """
        scene = getattr(self, "scene", None)
        if not scene:
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
                "7.3 场景精度损失缺少实测数据（output/scene_metrics.json 中无 scenes 字段），"
                "请先运行 evaluate_yolov8_onnx_scene.py 生成场景测评结果"
            )
        metrics_by_path = index_scene_metrics(scenes)

        rows: list[str] = []
        for i, path in enumerate(paths, start=1):
            scene_dir = Path(path)
            if not scene_dir.is_absolute():
                scene_dir = _PROJECT_ROOT / scene_dir
            metrics = metrics_by_path.get(str(scene_dir.resolve()))
            if metrics is None:
                raise ValueError(
                    f"7.3 场景精度损失缺少场景 {i}（{path}）的实测指标，"
                    "请检查 local_info.json 的 scene.path 与测评导出的 scenes 是否对应"
                )

            cloud_normal = to_float(
                getattr(self, "det_cloud_map05", ""), "det_cloud_map05"
            )
            edge_normal = to_float(
                getattr(self, "det_edge_map05", ""), "det_edge_map05"
            )
            cloud_scene = to_float(
                metrics.get("cloud_map05"), f"cloud_map05（场景{i}）"
            )
            edge_scene = to_float(
                metrics.get("det_edge_map05"), f"det_edge_map05（场景{i}）"
            )

            if cloud_normal == 0:
                raise ValueError(f"det_cloud_map05 为 0，无法计算场景 {i} 的精度差距")
            if edge_normal == 0:
                raise ValueError(f"det_edge_map05 为 0，无法计算场景 {i} 的精度差距")

            cloud_gap = (cloud_scene - cloud_normal) / cloud_normal
            edge_gap = (edge_scene - edge_normal) / edge_normal

            rows.append(
                f"| 场景{i} "
                f"| {cloud_normal:.4f} | {cloud_scene:.4f} "
                f"| {pct(cloud_scene - cloud_normal, cloud_normal, 'det_cloud_map05')} "
                f"| {edge_normal:.4f} | {edge_scene:.4f} "
                f"| {pct(edge_scene - edge_normal, edge_normal, 'det_edge_map05')} "
                f"| {'是' if edge_gap >= cloud_gap else '否'} |"
            )

        self.scene_loss_rows = "\n".join(rows)

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))