"""第 7 章：评测结果判定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tools import pct, to_float

from ..render import render

_TEMPLATE = r"""# 7. 评测结果判定

## 7.1 精度指标达标情况

| 指标       |   云端原始值 |    目标值 |    实测值 |   精度损失 | 是否达标   |
| -------- | -----: | -----: | -----: | -----: | ------ |
| mAP\@0.5  | {{v_map05_cloud}} | {{v_map05_target}} | {{v_map05_actual}} | {{v_map05_loss}} | {{v_map05_ok}} |
| Precision | {{v_precision_cloud}} | {{v_precision_target}} | {{v_precision_actual}} | {{v_precision_loss}} | {{v_precision_ok}} |
| Recall    | {{v_recall_cloud}} | {{v_recall_target}} | {{v_recall_actual}} | {{v_recall_loss}} | {{v_recall_ok}} |

## 7.2 推理性能指标统计

| 指标     |    目标值 |    实测值 | 是否达标   |
| ------ | -----: | -----: | ------ |
| 平均推理耗时 | {{v_perf_latency_target}} | {{v_perf_latency_actual}} | {{v_perf_latency_ok}} |
| FPS    | {{v_perf_fps_target}} | {{v_perf_fps_actual}} | {{v_perf_fps_ok}} |
| 预处理耗时  | {{v_perf_pre_target}} | {{v_perf_pre_actual}} | {{v_perf_pre_ok}} |
| 推理耗时   | {{v_perf_infer_target}} | {{v_perf_infer_actual}} | {{v_perf_infer_ok}} |
| 后处理耗时  | {{v_perf_post_target}} | {{v_perf_post_actual}} | {{v_perf_post_ok}} |

## 7.3 精度损失是否满足要求

| 项目       | 要求     | 实际结果   | 判定       |
| -------- | ------ | ------ | -------- |
| 量化精度损失   | {{req_quant_loss=例如 <1%}} | {{res_quant_loss}} | {{jud_quant_loss=满足 / 不满足}} |
| 边缘场景精度损失 | {{req_edge_loss}} | {{res_edge_loss}} | {{jud_edge_loss=满足 / 不满足}} |
| 关键指标变化   | {{req_key_change}} | {{res_key_change}} | {{jud_key_change=满足 / 不满足}} |

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

    # 7.2 推理性能指标统计
    v_perf_latency_target: str = ""  # 平均推理耗时-目标值
    v_perf_latency_actual: str = ""  # 平均推理耗时-实测值
    v_perf_latency_ok: str = ""  # 平均推理耗时-是否达标
    v_perf_fps_target: str = ""  # FPS-目标值
    v_perf_fps_actual: str = ""  # FPS-实测值
    v_perf_fps_ok: str = ""  # FPS-是否达标
    v_perf_pre_target: str = ""  # 预处理耗时-目标值
    v_perf_pre_actual: str = ""  # 预处理耗时-实测值
    v_perf_pre_ok: str = ""  # 预处理耗时-是否达标
    v_perf_infer_target: str = ""  # 推理耗时-目标值
    v_perf_infer_actual: str = ""  # 推理耗时-实测值
    v_perf_infer_ok: str = ""  # 推理耗时-是否达标
    v_perf_post_target: str = ""  # 后处理耗时-目标值
    v_perf_post_actual: str = ""  # 后处理耗时-实测值
    v_perf_post_ok: str = ""  # 后处理耗时-是否达标

    # 7.3 精度损失是否满足要求
    req_quant_loss: str = ""  # 量化精度损失-要求（未填时默认「例如 <1%」）
    res_quant_loss: str = ""  # 量化精度损失-实际结果
    jud_quant_loss: str = ""  # 量化精度损失-判定（未填时默认「满足 / 不满足」）
    req_edge_loss: str = ""  # 边缘场景精度损失-要求
    res_edge_loss: str = ""  # 边缘场景精度损失-实际结果
    jud_edge_loss: str = ""  # 边缘场景精度损失-判定（未填时默认「满足 / 不满足」）
    req_key_change: str = ""  # 关键指标变化-要求
    res_key_change: str = ""  # 关键指标变化-实际结果
    jud_key_change: str = ""  # 关键指标变化-判定（未填时默认「满足 / 不满足」）

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

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))