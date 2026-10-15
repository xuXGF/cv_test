"""第 8 章：量化调优建议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 8. 量化调优建议

## 8.1 精度优化建议

| 当前问题      | 优化建议      | 预期效果   | 优先级    |
| --------- | --------- | ------ | ------ |
| 量化导致精度下降  | 优化量化校准数据集 | {{sug_acc_1_effect}} | {{sug_acc_1_priority=高}}      |
| 边缘场景精度下降  | 增加边缘场景数据  | {{sug_acc_2_effect}} | {{sug_acc_2_priority=高}}      |
| 小目标识别能力下降 | {{sug_acc_3_advice}}    | {{sug_acc_3_effect}} | {{sug_acc_3_priority=中}}      |
| 低照度场景性能下降 | {{sug_acc_4_advice}}    | {{sug_acc_4_effect}} | {{sug_acc_4_priority=中}}      |
| 其他        | {{sug_acc_5_advice}}    | {{sug_acc_5_effect}} | {{sug_acc_5_priority}} |

可根据实际情况考虑：

{{ana_81=\[由大模型进行罗列一下建议]}}

## 8.2 推理性能优化建议

| 当前问题     | 优化建议               | 预期效果   | 优先级    |
| -------- | ------------------ | ------ | ------ |
| 预处理耗时较高  | 优化图像预处理流程          | {{sug_perf_1_effect}} | {{sug_perf_1_priority}} |
| 模型推理耗时较高 | 算子融合/模型结构优化        | {{sug_perf_2_effect}} | {{sug_perf_2_priority}} |
| 后处理耗时较高  | 优化后处理算法            | {{sug_perf_3_effect}} | {{sug_perf_3_priority}} |
| 数据搬运耗时较高 | 减少CPU/GPU/NPU间数据拷贝 | {{sug_perf_4_effect}} | {{sug_perf_4_priority}} |
| 硬件利用率不足  | 优化线程及推理调度          | {{sug_perf_5_effect}} | {{sug_perf_5_priority}} |

## 8.3 性能瓶颈优化建议

| 性能瓶颈   | 当前表现   | 优化方案   | 预期收益   | 优先级    |
| ------ | ------ | ------ | ------ | ------ |
| {{sug_bn_1_bottleneck}} | {{sug_bn_1_current}} | {{sug_bn_1_plan}} | {{sug_bn_1_gain}} | {{sug_bn_1_priority}} |
| {{sug_bn_2_bottleneck}} | {{sug_bn_2_current}} | {{sug_bn_2_plan}} | {{sug_bn_2_gain}} | {{sug_bn_2_priority}} |
| {{sug_bn_3_bottleneck}} | {{sug_bn_3_current}} | {{sug_bn_3_plan}} | {{sug_bn_3_gain}} | {{sug_bn_3_priority}} |"""


@dataclass
class SuggestionSection:
    """第 8 章：量化调优建议。"""

    # 8.1 精度优化建议
    sug_acc_1_effect: str = ""  # 量化导致精度下降-预期效果
    sug_acc_1_priority: str = ""  # 量化导致精度下降-优先级（未填时默认「高」）
    sug_acc_2_effect: str = ""  # 边缘场景精度下降-预期效果
    sug_acc_2_priority: str = ""  # 边缘场景精度下降-优先级（未填时默认「高」）
    sug_acc_3_advice: str = ""  # 小目标识别能力下降-优化建议
    sug_acc_3_effect: str = ""  # 小目标识别能力下降-预期效果
    sug_acc_3_priority: str = ""  # 小目标识别能力下降-优先级（未填时默认「中」）
    sug_acc_4_advice: str = ""  # 低照度场景性能下降-优化建议
    sug_acc_4_effect: str = ""  # 低照度场景性能下降-预期效果
    sug_acc_4_priority: str = ""  # 低照度场景性能下降-优先级（未填时默认「中」）
    sug_acc_5_advice: str = ""  # 其他-优化建议
    sug_acc_5_effect: str = ""  # 其他-预期效果
    sug_acc_5_priority: str = ""  # 其他-优先级
    ana_81: str = ""  # 8.1 建议正文

    # 8.2 推理性能优化建议
    sug_perf_1_effect: str = ""  # 预处理耗时较高-预期效果
    sug_perf_1_priority: str = ""  # 预处理耗时较高-优先级
    sug_perf_2_effect: str = ""  # 模型推理耗时较高-预期效果
    sug_perf_2_priority: str = ""  # 模型推理耗时较高-优先级
    sug_perf_3_effect: str = ""  # 后处理耗时较高-预期效果
    sug_perf_3_priority: str = ""  # 后处理耗时较高-优先级
    sug_perf_4_effect: str = ""  # 数据搬运耗时较高-预期效果
    sug_perf_4_priority: str = ""  # 数据搬运耗时较高-优先级
    sug_perf_5_effect: str = ""  # 硬件利用率不足-预期效果
    sug_perf_5_priority: str = ""  # 硬件利用率不足-优先级

    # 8.3 性能瓶颈优化建议
    sug_bn_1_bottleneck: str = ""  # 第1行-性能瓶颈
    sug_bn_1_current: str = ""  # 第1行-当前表现
    sug_bn_1_plan: str = ""  # 第1行-优化方案
    sug_bn_1_gain: str = ""  # 第1行-预期收益
    sug_bn_1_priority: str = ""  # 第1行-优先级
    sug_bn_2_bottleneck: str = ""  # 第2行-性能瓶颈
    sug_bn_2_current: str = ""  # 第2行-当前表现
    sug_bn_2_plan: str = ""  # 第2行-优化方案
    sug_bn_2_gain: str = ""  # 第2行-预期收益
    sug_bn_2_priority: str = ""  # 第2行-优先级
    sug_bn_3_bottleneck: str = ""  # 第3行-性能瓶颈
    sug_bn_3_current: str = ""  # 第3行-当前表现
    sug_bn_3_plan: str = ""  # 第3行-优化方案
    sug_bn_3_gain: str = ""  # 第3行-预期收益
    sug_bn_3_priority: str = ""  # 第3行-优先级

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))