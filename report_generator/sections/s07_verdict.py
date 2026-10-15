"""第 7 章：评测结果判定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 7. 评测结果判定

## 7.1 精度指标达标情况

| 指标          |    目标值 |    实测值 |   精度损失 | 是否达标   |
| ----------- | -----: | -----: | -----: | ------ |
| 主要精度指标      | {{v_acc_main_target}} | {{v_acc_main_actual}} | {{v_acc_main_loss}} | {{v_acc_main_ok}} |
| Top-1/Top-5 | {{v_top_target}} | {{v_top_actual}} | {{v_top_loss}} | {{v_top_ok}} |
| mAP         | {{v_map_target}} | {{v_map_actual}} | {{v_map_loss}} | {{v_map_ok}} |
| IoU/mIoU    | {{v_iou_target}} | {{v_iou_actual}} | {{v_iou_loss}} | {{v_iou_ok}} |

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

    # 7.1 精度指标达标情况
    v_acc_main_target: str = ""  # 主要精度指标-目标值
    v_acc_main_actual: str = ""  # 主要精度指标-实测值
    v_acc_main_loss: str = ""  # 主要精度指标-精度损失
    v_acc_main_ok: str = ""  # 主要精度指标-是否达标
    v_top_target: str = ""  # Top-1/Top-5-目标值
    v_top_actual: str = ""  # Top-1/Top-5-实测值
    v_top_loss: str = ""  # Top-1/Top-5-精度损失
    v_top_ok: str = ""  # Top-1/Top-5-是否达标
    v_map_target: str = ""  # mAP-目标值
    v_map_actual: str = ""  # mAP-实测值
    v_map_loss: str = ""  # mAP-精度损失
    v_map_ok: str = ""  # mAP-是否达标
    v_iou_target: str = ""  # IoU/mIoU-目标值
    v_iou_actual: str = ""  # IoU/mIoU-实测值
    v_iou_loss: str = ""  # IoU/mIoU-精度损失
    v_iou_ok: str = ""  # IoU/mIoU-是否达标

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

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))