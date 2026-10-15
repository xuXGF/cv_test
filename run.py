# from __future__ import annotations

from pathlib import Path
import time
from tools import *

from report_generator import TestReport
OUTPUT_DIR = Path(__file__).parent / "output"

from local_info import load_local_info

local_info = load_local_info()


from device_info import collect_device_info
device_info =  collect_device_info()


from model_info import collect_model_info, load_session, benchmark

model_info = collect_model_info()  # 缺省使用包内自带 yolov8n.onnx


from dataset_info import collect_dataset_info

dataset_info = collect_dataset_info()  # 缺省使用包内自带 data/coco128



# def build_sample_report() -> TestReport:
#     """示例：为部分参数赋值后的报告。"""
#     # 1.1
#     report = TestReport()
#     report.report_title = f"{model_info.model_name}模型测评报告"
#     report.basic_info.report_name = report.report_title
#     report.basic_info.report_no = "RPT-2026-0001"
#     report.basic_info.model_name = model_info.model_name
#     report.basic_info.algo_type = local_info.algo_type
#     report.basic_info.test_date = time.strftime("%Y/%m/%d %H:%M", time.localtime())
#     # report.basic_info.conclusion = "通过"
#
#     # 1.2
#     report.basic_info.algo_name = model_info.model_name
#     report.basic_info.algo_category = local_info.algo_type
#     report.basic_info.model_arch = model_info.model_arch
#     report.basic_info.model_version = model_info.model_version
#     report.basic_info.param_count = format_params(model_info.param_count)
#     report.basic_info.FLOPs = model_info.flops
#     report.basic_info.model_size = model_info.model_size
#     report.basic_info.input_format = model_info.input_format
#     report.basic_info.output_format = model_info.output_format
#     report.basic_info.num_classes = model_info.num_classes
#     report.basic_info.input_size = model_info.input_size
#     report.basic_info.train_framework = model_info.train_framework
#     report.basic_info.original_model_format = model_info.original_model_format
#     report.basic_info.edge_model_format = model_info.edge_model_format
#
#
#
#     # 1.3
#     report.basic_info.device_name = device_info.device_name
#     report.basic_info.soc = device_info.soc
#     report.basic_info.os = device_info.os
#     report.basic_info.cpu = device_info.cpu
#     report.basic_info.gpu = device_info.gpu
#     report.basic_info.npu = device_info.npu
#     report.basic_info.memory = device_info.memory
#     report.basic_info.storage = device_info.storage
#
#     # 1.4
#     report.basic_info.python_version = model_info.python_version
#     report.basic_info.dl_framework =  model_info.train_framework
#     report.basic_info.inference_framework = model_info.inference_framework
#     report.basic_info.runtime_version = model_info.runtime_version
#     report.basic_info.batch_size = model_info.batch_size
#     report.basic_info.input_resolution = model_info.input_size
#     report.basic_info.other_env = ''
#
#     return report


def build_sample_report() -> TestReport:
    report = TestReport()
    report.basic_info.__dict__.update(model_info.__dict__)
    report.basic_info.__dict__.update(local_info.__dict__)
    report.basic_info.__dict__.update(device_info.__dict__)

    report.report_title = f"{model_info.model_name}模型测评报告"
    report.basic_info.report_name = report.report_title
    report.basic_info.report_no = "RPT-2026-0001"
    report.basic_info.test_date = time.strftime("%Y/%m/%d %H:%M", time.localtime())
    # report.basic_info.conclusion = "通过"

    # 1.2
    report.basic_info.algo_category = local_info.algo_type
    report.basic_info.param_count = format_params(model_info.param_count)
    report.basic_info.FLOPs = model_info.flops

    # 1.3

    # 1.4
    report.basic_info.dl_framework = model_info.train_framework
    report.basic_info.input_resolution = model_info.input_size
    report.basic_info.other_env = ''
    # report.dataset.dataset_name = local_info.dataset_name

    return report

def build_sample_report() -> TestReport:
    report = TestReport()
    # 第 2 章：测试数据集说明
    # 合并顺序即优先级（后者覆盖前者）：模型侧字段 -> 数据集扫描 -> 本地手填
    report.dataset.__dict__.update(model_info.__dict__)
    report.dataset.__dict__.update(dataset_info.__dict__)  # 2.1 基本信息 / 2.3 标签格式转换
    report.dataset.__dict__.update(local_info.__dict__)    # 手填字段：数据集名称/来源/类型、2.2 边缘场景

    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # empty_path = build_empty_report().save(OUTPUT_DIR / "report_empty.md")
    sample_path = build_sample_report().save(OUTPUT_DIR / "report_sample.md")

    # print(f"空参数报告（等价模板）: {empty_path}")
    print(f"示例填充报告: {sample_path}")


if __name__ == "__main__":
    main()

# for key, value in asdict(collect_device_info()).items():
#     print(f"{key}: {value!r}")