"""示例入口：演示如何按模板生成测评报告。

* 不赋值任何参数时，输出与 ``template.md`` 等价（所有可填写单元格为空）。
* 为参数赋值后，直接输出填充好内容的报告。
"""

# from __future__ import annotations

from pathlib import Path

from report_generator import TestReport

OUTPUT_DIR = Path(__file__).parent / "output"


def build_empty_report() -> TestReport:
    """所有参数保持默认空值，用于验证输出与模板一致。"""
    return TestReport()


def build_sample_report() -> TestReport:
    """示例：为部分参数赋值后的报告。"""
    report = TestReport()
    # report.report_title = "**某检测模型边缘端部署测评报告**"
    # report.basic_info.report_name = "某检测模型边缘端部署测评报告"
    # report.basic_info.report_no = "RPT-2026-0001"
    # report.basic_info.model_name = "yolov8n"
    # report.basic_info.algo_type = "目标检测"
    # report.basic_info.test_date = "2026-10-13"
    # report.basic_info.conclusion = "通过"

    # report.basic_info.algo_name = "yolov8n-edgedet"
    # report.basic_info.model_arch = "YOLOv8"
    # report.basic_info.input_size = "640x640"
    # report.basic_info.train_framework = "PyTorch"

    # report.basic_info.device_name = "RK3588 边缘计算盒"
    # report.basic_info.soc = "RK3588"
    # report.basic_info.os = "Ubuntu 20.04"

    # report.basic_info.python_version = "3.10"
    # report.basic_info.inference_framework = "RKNN"
    # report.basic_info.runtime_version = "rknn-toolkit2 2.0"

    # report.dataset.num_classes = "80"
    # report.dataset.test_data_count = "5000"

    # report.speed.speed_edge_latency = "12.5"
    # report.speed.speed_edge_fps = "80"

    # report.verdict.verdict_overall = "通过"
    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # empty_path = build_empty_report().save(OUTPUT_DIR / "report_empty.md")
    sample_path = build_sample_report().save(OUTPUT_DIR / "report_sample.md")

    # print(f"空参数报告（等价模板）: {empty_path}")
    print(f"示例填充报告: {sample_path}")


if __name__ == "__main__":
    main()