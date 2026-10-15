"""第 1 章：报告基本信息。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 1. 报告基本信息

## 1.1 报告基本信息

| 项目   | 内容     |
| ---- | ------ |
| 报告名称 | {{report_name}} |
| 报告编号 | {{report_no}} |
| 模型名称 | {{model_name}} |
| 算法类型 | {{algo_type}} |
| 测评日期 | {{test_date}} |
| 测评结论 | {{conclusion}} |

## 1.2 算法信息

| 项目     | 内容     |
| ------ | ------ |
| 算法名称   | {{algo_name}} |
| 算法类型   | {{algo_category}} |
| 模型架构   | {{model_arch}} |
| 模型版本   | {{model_version}} |
| 参数量    | {{param_count}} |
| FLOPs  | {{flops}} |
| 模型大小   | {{model_size}} |
| 输入尺寸   | {{input_size}} |
| 输入格式   | {{input_format}} |
| 输出格式   | {{output_format}} |
| 类别数量   | {{num_classes}} |
| 训练框架   | {{train_framework}} |
| 原始模型格式 | {{original_model_format}} |
| 边缘模型格式 | {{edge_model_format}} |

## 1.3 测试设备信息

| 项目     | 边缘端测试设备 |
| ------ | ------- |
| 设备名称   | {{device_name}}  |
| 芯片/SoC | {{soc}}  |
| CPU    | {{cpu}}  |
| GPU    | {{gpu}}  |
| NPU    | {{npu}}  |
| 内存     | {{memory}}  |
| 存储     | {{storage}}  |
| 操作系统   | {{os}}  |

## 1.4 测试环境信息

| 项目         | 配置信息   |
| ---------- | ------ |
| Python版本   | {{python_version}} |
| 深度学习框架     | {{dl_framework}} |
| 推理框架       | {{inference_framework}} |
| Runtime版本  | {{runtime_version}} |
| Batch Size | {{batch_size}} |
| 输入分辨率      | {{input_resolution}} |
| 其他环境配置     | {{other_env}} |"""


@dataclass
class BasicInfoSection:
    """第 1 章：报告基本信息。

    所有字段默认空值，渲染结果与模板一致；赋值后直接填入对应单元格。
    """

    # 1.1 报告基本信息
    report_name: str = ""  # 报告名称
    report_no: str = ""  # 报告编号
    model_name: str = ""  # 模型名称
    algo_type: str = ""  # 算法类型
    test_date: str = ""  # 测评日期
    conclusion: str = ""  # 测评结论

    # 1.2 算法信息
    algo_name: str = ""  # 算法名称
    algo_category: str = ""  # 算法类型
    model_arch: str = ""  # 模型架构
    model_version: str = ""  # 模型版本
    param_count: str = ""  # 参数量
    flops: str = ""  # FLOPs（浮点运算次数）
    model_size: str = ""  # 模型大小
    input_size: str = ""  # 输入尺寸
    input_format: str = ""  # 输入格式
    output_format: str = ""  # 输出格式
    num_classes: str = ""  # 类别数量
    train_framework: str = ""  # 训练框架
    original_model_format: str = ""  # 原始模型格式
    edge_model_format: str = ""  # 边缘模型格式

    # 1.3 测试设备信息（边缘端测试设备）
    device_name: str = ""  # 设备名称
    soc: str = ""  # 芯片/SoC
    cpu: str = ""  # CPU
    gpu: str = ""  # GPU
    npu: str = ""  # NPU
    memory: str = ""  # 内存
    storage: str = ""  # 存储
    os: str = ""  # 操作系统

    # 1.4 测试环境信息（配置信息）
    python_version: str = ""  # Python版本
    dl_framework: str = ""  # 深度学习框架
    inference_framework: str = ""  # 推理框架
    runtime_version: str = ""  # Runtime版本
    batch_size: str = ""  # Batch Size（批大小）
    input_resolution: str = ""  # 输入分辨率
    other_env: str = ""  # 其他环境配置

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))