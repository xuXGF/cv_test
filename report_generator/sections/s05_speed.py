"""第 5 章：推理速度评测。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tools import pct, to_float

from ..render import render

_TEMPLATE = r"""# 5. 推理速度评测

## 5.1 平均推理耗时及FPS

| 测试对象    | 平均推理耗时(ms) |    FPS |
| ------- | ---------: | -----: |
| 云端原始模型  |     {{speed_cloud_latency}} | {{speed_cloud_fps}} |
| 边缘端量化模型 |     {{speed_edge_latency}} | {{speed_edge_fps}} |

## 5.2 预处理、推理、后处理耗时

| 测试对象    | 预处理耗时(ms) | 推理耗时(ms) | 后处理耗时(ms) | 总耗时(ms) |
| ------- | --------: | -------: | --------: | ------: |
| 云端原始模型  |    {{stage_cloud_pre}} |   {{stage_cloud_infer}} |    {{stage_cloud_post}} |  {{stage_cloud_total}} |
| 边缘端量化模型 |    {{stage_edge_pre}} |   {{stage_edge_infer}} |    {{stage_edge_post}} |  {{stage_edge_total}} |

## 5.3 各阶段耗时占比

| 测试对象    |  预处理占比 |   推理占比 |  后处理占比 |
| ------- | -----: | -----: | -----: |
| 云端原始模型  | {{ratio_cloud_pre}} | {{ratio_cloud_infer}} | {{ratio_cloud_post}} |
| 边缘端量化模型 | {{ratio_edge_pre}} | {{ratio_edge_infer}} | {{ratio_edge_post}} |

> 各阶段耗时占比 = 对应阶段耗时 / 总耗时 × 100%

## 5.4 推理性能测试结果

| 测试条件           | 耗时(ms) |    FPS | CPU利用率 | GPU/NPU利用率 |   内存占用 |
| -------------- | -------: | -----: | -----: | ---------: | -----: |
| Batch Size = 1 |   {{bs1_latency}} | {{bs1_fps}} | {{bs1_cpu}} |     {{bs1_npu}} | {{bs1_mem}} |
| Batch Size = 4 |   {{bs4_latency}} | {{bs4_fps}} | {{bs4_cpu}} |     {{bs4_npu}} | {{bs4_mem}} |
| Batch Size = 8 |   {{bs8_latency}} | {{bs8_fps}} | {{bs8_cpu}} |     {{bs8_npu}} | {{bs8_mem}} |"""


@dataclass
class SpeedSection:
    """第 5 章：推理速度评测。"""

    # 5.1 平均推理耗时及FPS
    speed_cloud_latency: str = ""  # 云端原始模型-平均推理耗时(ms)
    speed_cloud_fps: str = ""  # 云端原始模型-FPS
    speed_edge_latency: str = ""  # 边缘端量化模型-平均推理耗时(ms)
    speed_edge_fps: str = ""  # 边缘端量化模型-FPS

    # 5.2 预处理、推理、后处理耗时
    stage_cloud_pre: str = ""  # 云端原始模型-预处理耗时(ms)
    stage_cloud_infer: str = ""  # 云端原始模型-推理耗时(ms)
    stage_cloud_post: str = ""  # 云端原始模型-后处理耗时(ms)
    stage_cloud_total: str = ""  # 云端原始模型-总耗时(ms)
    stage_edge_pre: str = ""  # 边缘端量化模型-预处理耗时(ms)
    stage_edge_infer: str = ""  # 边缘端量化模型-推理耗时(ms)
    stage_edge_post: str = ""  # 边缘端量化模型-后处理耗时(ms)
    stage_edge_total: str = ""  # 边缘端量化模型-总耗时(ms)

    # 5.3 各阶段耗时占比
    ratio_cloud_pre: str = ""  # 云端原始模型-预处理占比
    ratio_cloud_infer: str = ""  # 云端原始模型-推理占比
    ratio_cloud_post: str = ""  # 云端原始模型-后处理占比
    ratio_edge_pre: str = ""  # 边缘端量化模型-预处理占比
    ratio_edge_infer: str = ""  # 边缘端量化模型-推理占比
    ratio_edge_post: str = ""  # 边缘端量化模型-后处理占比

    # 5.4 推理性能测试结果
    bs1_latency: str = ""  # Batch Size=1-平均耗时(ms)
    bs1_fps: str = ""  # Batch Size=1-FPS
    bs1_cpu: str = ""  # Batch Size=1-CPU利用率
    bs1_npu: str = ""  # Batch Size=1-GPU/NPU利用率
    bs1_mem: str = ""  # Batch Size=1-内存占用
    bs4_latency: str = ""  # Batch Size=4-平均耗时(ms)
    bs4_fps: str = ""  # Batch Size=4-FPS
    bs4_cpu: str = ""  # Batch Size=4-CPU利用率
    bs4_npu: str = ""  # Batch Size=4-GPU/NPU利用率
    bs4_mem: str = ""  # Batch Size=4-内存占用
    bs8_latency: str = ""  # Batch Size=8-平均耗时(ms)
    bs8_fps: str = ""  # Batch Size=8-FPS
    bs8_cpu: str = ""  # Batch Size=8-CPU利用率
    bs8_npu: str = ""  # Batch Size=8-GPU/NPU利用率
    bs8_mem: str = ""  # Batch Size=8-内存占用

    def fill_cloud_ratios(self) -> None:
        """5.3 云端各阶段耗时占比：由 5.2 手填耗时计算，不手填。

        占比 = 对应阶段耗时 / 总耗时 × 100%
        """
        total = to_float(self.stage_cloud_total, "stage_cloud_total")
        self.ratio_cloud_pre = pct(
            to_float(self.stage_cloud_pre, "stage_cloud_pre"),
            total,
            "stage_cloud_total",
        )
        self.ratio_cloud_infer = pct(
            to_float(self.stage_cloud_infer, "stage_cloud_infer"),
            total,
            "stage_cloud_total",
        )
        self.ratio_cloud_post = pct(
            to_float(self.stage_cloud_post, "stage_cloud_post"),
            total,
            "stage_cloud_total",
        )

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))