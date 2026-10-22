"""第 9 章：测试数据及附件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from ..render import render

_TEMPLATE = r"""# 9. 测试数据及附件

## 9.1 原始测试数据

| 文件名称   | 文件类型   | 数据内容   |    数据量 | 文件路径   |
| ------ | ------ | ------ | -----: | ------ |
| {{raw_file_1_name}} | {{raw_file_1_type}} | {{raw_file_1_content}} | {{raw_file_1_size}} | {{raw_file_1_path}} |
| {{raw_file_2_name}} | {{raw_file_2_type}} | {{raw_file_2_content}} | {{raw_file_2_size}} | {{raw_file_2_path}} |

## 9.2 测试日志

| 日志名称   | 测试类型   | 测试时间   | 文件路径   | 备注     |
| ------ | ------ | ------ | ------ | ------ |
| {{log_1_name}} | 精度测试   | {{log_1_time}} | {{log_1_path}} | {{log_1_note}} |
| {{log_2_name}} | 性能测试   | {{log_2_time}} | {{log_2_path}} | {{log_2_note}} |
| {{log_3_name}} | 边缘场景测试 | {{log_3_time}} | {{log_3_path}} | {{log_3_note}} |

## 9.3 评测脚本输出数据

| 文件名称   | 输出内容   | 数据格式     | 文件路径   |
| ------ | ------ | -------- | ------ |
| 整体指标汇总 | 边缘端精度与速度指标汇总（mAP、Precision、Recall、端侧延迟、FPS 及各阶段耗时占比） | JSON | output/metrics_summary.json |
| 逐类别精度明细 | 精度评测逐类别明细（各类别的精确率、召回率、F1、AP50、AP50:95） | CSV | output/per_class_metrics.csv |
| 逐图耗时明细 | 性能评测逐图耗时明细（每张图的预处理、推理、后处理及总耗时） | CSV | output/per_image_timings.csv |
| 批量性能评测结果 | Batch Size 性能评测结果（不同批量下的延迟、FPS、CPU 利用率、内存占用） | JSON | output/batch_metrics.json |
| 边缘场景精度评测结果 | 各边缘场景的边缘端与云端模型精度指标（mAP、Precision、Recall） | JSON | output/scene_metrics.json |
| 大模型分析结果 | 模型对比分析、场景泛化分析、性能分析、综合判定与精度/性能优化建议 | JSON | output/llm_analysis.json |
| 设备信息采集结果 | 边缘设备的 SoC、CPU、GPU、NPU、内存、存储、操作系统信息 | JSON | output/device_info.json |
| 模型信息采集结果 | 模型名、参数量、模型大小、精度格式、输入输出格式、框架及运行时版本 | JSON | output/model_info.json |

## 9.4 标准化评测结果

| 文件名称   | 内容      | 生成时间   | 文件格式         | 文件路径   |
| ------ | ------- | ------ | ------------ | ------ |
| {{std_name}} | 标准化测评报告 | {{std_time}} | Markdown | {{std_path}} |

## 9.5 附件清单

- [ ] 原始模型
- [ ] 边缘端量化模型
- [ ] 测试数据集
- [ ] 标准化数据集
- [ ] 原始测试数据
- [ ] 测试日志
- [ ] 精度评测脚本输出
- [ ] 性能评测脚本输出
- [ ] 边缘场景测试结果
- [ ] 标准化评测结果
- [ ] 其他附件

"""


@dataclass
class AttachmentSection:
    """第 9 章：测试数据及附件。"""

    # 9.1 原始测试数据
    raw_file_1_name: str = ""  # 第1行-文件名称
    raw_file_1_type: str = ""  # 第1行-文件类型
    raw_file_1_content: str = ""  # 第1行-数据内容
    raw_file_1_size: str = ""  # 第1行-数据量
    raw_file_1_path: str = ""  # 第1行-文件路径
    raw_file_2_name: str = ""  # 第2行-文件名称
    raw_file_2_type: str = ""  # 第2行-文件类型
    raw_file_2_content: str = ""  # 第2行-数据内容
    raw_file_2_size: str = ""  # 第2行-数据量
    raw_file_2_path: str = ""  # 第2行-文件路径

    # 9.2 测试日志
    log_1_name: str = ""  # 第1行（精度测试）-日志名称
    log_1_time: str = ""  # 第1行（精度测试）-测试时间
    log_1_path: str = ""  # 第1行（精度测试）-文件路径
    log_1_note: str = ""  # 第1行（精度测试）-备注
    log_2_name: str = ""  # 第2行（性能测试）-日志名称
    log_2_time: str = ""  # 第2行（性能测试）-测试时间
    log_2_path: str = ""  # 第2行（性能测试）-文件路径
    log_2_note: str = ""  # 第2行（性能测试）-备注
    log_3_name: str = ""  # 第3行（边缘场景测试）-日志名称
    log_3_time: str = ""  # 第3行（边缘场景测试）-测试时间
    log_3_path: str = ""  # 第3行（边缘场景测试）-文件路径
    log_3_note: str = ""  # 第3行（边缘场景测试）-备注

    # 9.4 标准化评测结果
    std_name: str = ""  # 标准化测评报告-文件名称
    std_time: str = ""  # 标准化测评报告-生成时间
    std_path: str = ""  # 标准化测评报告-文件路径

    def fill_std_result(self, report_path: Path) -> None:
        """9.4 标准化评测结果：仅标准化测评报告一行，生成时间取当前日期。"""
        self.std_name = report_path.name
        self.std_time = date.today().strftime("%Y-%m-%d")
        try:
            self.std_path = report_path.resolve().relative_to(Path.cwd()).as_posix()
        except ValueError:
            self.std_path = report_path.as_posix()

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))