"""第 2 章：测试数据集说明。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 2. 测试数据集说明

## 2.1 数据集基本信息

| 项目    | 内容           |
| ----- | ------------ |
| 数据集名称 | {{dataset_name}}       |
| 数据来源  | {{data_source}}       |
| 数据类型  | {{data_type=图片 / 视频 / 其他}} |
| 数据总量  | {{data_total}}       |
| 测试数据量 | {{test_data_count}}       |
| 类别数量  | {{num_classes}}       |
| 标注格式  | {{annotation_format}}       |
| 数据分辨率 | {{data_resolution}}       |
| 数据格式  | {{data_format}}       |

## 2.2 边缘场景数据

针对真实业务应用中的复杂环境，补充边缘场景数据，用于验证模型的实际泛化能力。

| 场景编号 | 测试场景    | 场景描述   |    数据量 |     占比 |
| ---- | ------- | ------ | -----: | -----: |
| 01   | **场景1** | {{scenario_01_desc}} | {{scenario_01_count}} | {{scenario_01_ratio}} |
| 02   | **场景2** | {{scenario_02_desc}} | {{scenario_02_count}} | {{scenario_02_ratio}} |
| 03   | **场景3** | {{scenario_03_desc}} | {{scenario_03_count}} | {{scenario_03_ratio}} |
| 04   | **场景4** | {{scenario_04_desc}} | {{scenario_04_count}} | {{scenario_04_ratio}} |
| 05   | **场景6** | {{scenario_05_desc}} | {{scenario_05_count}} | {{scenario_05_ratio}} |
| 06   | **场景6** | {{scenario_06_desc}} | {{scenario_06_count}} | {{scenario_06_ratio}} |

## 2.3 数据预处理及标准化

| 处理项目   | 原始数据   | 标准化处理方式 | 处理结果   |
| ------ | ------ | ------- | ------ |
| 数据格式转换 | {{pre_orig_format}} | {{pre_method_format}}  | {{pre_result_format}} |
| 标签格式转换 | {{pre_orig_label}} | {{pre_method_label}}  | {{pre_result_label}} |
| 图像尺寸处理 | {{pre_orig_size}} | {{pre_method_size}}  | {{pre_result_size}} |
| 数据清洗   | {{pre_orig_clean}} | {{pre_method_clean}}  | {{pre_result_clean}} |
| 异常数据处理 | {{pre_orig_anomaly}} | {{pre_method_anomaly}}  | {{pre_result_anomaly}} |"""


@dataclass
class DatasetSection:
    """第 2 章：测试数据集说明。"""

    # 2.1 数据集基本信息
    dataset_name: str = ""  # 数据集名称
    data_source: str = ""  # 数据来源
    data_type: str = ""  # 数据类型（未填时默认「图片 / 视频 / 其他」）
    data_total: str = ""  # 数据总量
    test_data_count: str = ""  # 测试数据量
    num_classes: str = ""  # 类别数量
    annotation_format: str = ""  # 标注格式
    data_resolution: str = ""  # 数据分辨率
    data_format: str = ""  # 数据格式

    # 2.2 边缘场景数据
    scenario_01_desc: str = ""  # 场景1-场景描述
    scenario_01_count: str = ""  # 场景1-数据量
    scenario_01_ratio: str = ""  # 场景1-占比
    scenario_02_desc: str = ""  # 场景2-场景描述
    scenario_02_count: str = ""  # 场景2-数据量
    scenario_02_ratio: str = ""  # 场景2-占比
    scenario_03_desc: str = ""  # 场景3-场景描述
    scenario_03_count: str = ""  # 场景3-数据量
    scenario_03_ratio: str = ""  # 场景3-占比
    scenario_04_desc: str = ""  # 场景4-场景描述
    scenario_04_count: str = ""  # 场景4-数据量
    scenario_04_ratio: str = ""  # 场景4-占比
    scenario_05_desc: str = ""  # 场景5-场景描述
    scenario_05_count: str = ""  # 场景5-数据量
    scenario_05_ratio: str = ""  # 场景5-占比
    scenario_06_desc: str = ""  # 场景6-场景描述
    scenario_06_count: str = ""  # 场景6-数据量
    scenario_06_ratio: str = ""  # 场景6-占比

    # 2.3 数据预处理及标准化
    pre_orig_format: str = ""  # 数据格式转换-原始数据
    pre_method_format: str = ""  # 数据格式转换-标准化处理方式
    pre_result_format: str = ""  # 数据格式转换-处理结果
    pre_orig_label: str = ""  # 标签格式转换-原始数据
    pre_method_label: str = ""  # 标签格式转换-标准化处理方式
    pre_result_label: str = ""  # 标签格式转换-处理结果
    pre_orig_size: str = ""  # 图像尺寸处理-原始数据
    pre_method_size: str = ""  # 图像尺寸处理-标准化处理方式
    pre_result_size: str = ""  # 图像尺寸处理-处理结果
    pre_orig_clean: str = ""  # 数据清洗-原始数据
    pre_method_clean: str = ""  # 数据清洗-标准化处理方式
    pre_result_clean: str = ""  # 数据清洗-处理结果
    pre_orig_anomaly: str = ""  # 异常数据处理-原始数据
    pre_method_anomaly: str = ""  # 异常数据处理-标准化处理方式
    pre_result_anomaly: str = ""  # 异常数据处理-处理结果

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))