"""第 9 章：测试数据及附件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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

| 文件名称   | 输出内容   | 数据格式     | 脚本版本   | 文件路径   |
| ------ | ------ | -------- | ------ | ------ |
| {{out_1_name}} | 精度评测结果 | CSV/JSON | {{out_1_version}} | {{out_1_path}} |
| {{out_2_name}} | 性能评测结果 | CSV/JSON | {{out_2_version}} | {{out_2_path}} |
| {{out_3_name}} | 场景评测结果 | CSV/JSON | {{out_3_version}} | {{out_3_path}} |
| {{out_4_name}} | 综合评测结果 | CSV/JSON | {{out_4_version}} | {{out_4_path}} |

## 9.4 标准化评测结果

| 文件名称   | 内容      | 生成时间   | 文件格式         | 文件路径   |
| ------ | ------- | ------ | ------------ | ------ |
| {{std_1_name}} | 标准化精度结果 | {{std_1_time}} | {{std_1_format}}       | {{std_1_path}} |
| {{std_2_name}} | 标准化性能结果 | {{std_2_time}} | {{std_2_format}}       | {{std_2_path}} |
| {{std_3_name}} | 标准化场景结果 | {{std_3_time}} | {{std_3_format}}       | {{std_3_path}} |
| {{std_4_name}} | 标准化测评报告 | {{std_4_time}} | Markdown/PDF | {{std_4_path}} |

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

    # 9.3 评测脚本输出数据
    out_1_name: str = ""  # 第1行（精度评测结果）-文件名称
    out_1_version: str = ""  # 第1行（精度评测结果）-脚本版本
    out_1_path: str = ""  # 第1行（精度评测结果）-文件路径
    out_2_name: str = ""  # 第2行（性能评测结果）-文件名称
    out_2_version: str = ""  # 第2行（性能评测结果）-脚本版本
    out_2_path: str = ""  # 第2行（性能评测结果）-文件路径
    out_3_name: str = ""  # 第3行（场景评测结果）-文件名称
    out_3_version: str = ""  # 第3行（场景评测结果）-脚本版本
    out_3_path: str = ""  # 第3行（场景评测结果）-文件路径
    out_4_name: str = ""  # 第4行（综合评测结果）-文件名称
    out_4_version: str = ""  # 第4行（综合评测结果）-脚本版本
    out_4_path: str = ""  # 第4行（综合评测结果）-文件路径

    # 9.4 标准化评测结果
    std_1_name: str = ""  # 第1行（标准化精度结果）-文件名称
    std_1_time: str = ""  # 第1行（标准化精度结果）-生成时间
    std_1_format: str = ""  # 第1行（标准化精度结果）-文件格式
    std_1_path: str = ""  # 第1行（标准化精度结果）-文件路径
    std_2_name: str = ""  # 第2行（标准化性能结果）-文件名称
    std_2_time: str = ""  # 第2行（标准化性能结果）-生成时间
    std_2_format: str = ""  # 第2行（标准化性能结果）-文件格式
    std_2_path: str = ""  # 第2行（标准化性能结果）-文件路径
    std_3_name: str = ""  # 第3行（标准化场景结果）-文件名称
    std_3_time: str = ""  # 第3行（标准化场景结果）-生成时间
    std_3_format: str = ""  # 第3行（标准化场景结果）-文件格式
    std_3_path: str = ""  # 第3行（标准化场景结果）-文件路径
    std_4_name: str = ""  # 第4行（标准化测评报告）-文件名称
    std_4_time: str = ""  # 第4行（标准化测评报告）-生成时间
    std_4_format: str = ""  # 第4行（标准化测评报告）-文件格式
    std_4_path: str = ""  # 第4行（标准化测评报告）-文件路径

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))