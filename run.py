# from __future__ import annotations

import json
from pathlib import Path
import time
from tools import *

from report_generator import TestReport
OUTPUT_DIR = Path(__file__).parent / "output"

from local_info import load_local_info

local_info = load_local_info()


from device_info import collect_device_info, save_device_info
device_info =  collect_device_info()


from model_info import collect_model_info, load_session, benchmark, save_model_info

model_info = collect_model_info()  # 缺省使用包内自带 yolov8n.onnx


from dataset_info import collect_dataset_info

dataset_info = collect_dataset_info()  # 缺省使用包内自带 data/coco128


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
    report.basic_info.__dict__.update(model_info.__dict__)
    report.basic_info.__dict__.update(local_info.__dict__)
    report.basic_info.__dict__.update(device_info.__dict__)

    report.report_title = f"{model_info.model_name}模型测评报告"
    report.basic_info.report_name = report.report_title
    report.basic_info.report_no = "RPT-2026-0001"
    report.basic_info.test_date = time.strftime("%Y/%m/%d %H:%M", time.localtime())
    # report.basic_info.conclusion = "通过"

    # 1.2 模型字段与 ModelInfo 同名，由上方合并自动填充；参数量格式化为可读字符串
    report.basic_info.param_count = format_params(model_info.param_count)

    # 1.3 设备信息与 DeviceInfo 同名，由上方合并自动填充

    # 1.4 环境字段与 ModelInfo 同名（深度学习框架/输入分辨率复用
    #     train_framework/input_size），由上方合并自动填充


    # 第 2 章：测试数据集说明
    # 合并顺序即优先级（后者覆盖前者）：模型侧字段 -> 数据集扫描 -> 本地手填
    report.dataset.__dict__.update(model_info.__dict__)
    report.dataset.__dict__.update(dataset_info.__dict__)  # 2.1 基本信息 / 2.3 标签格式转换
    report.dataset.__dict__.update(local_info.__dict__)    # 手填字段：数据集名称/来源/类型、2.2 边缘场景
    report.dataset.fill_scenarios()                        # 2.2 数据量/占比：由场景地址自动统计，不手填

    # 第 3 章：测试配置
    # 合并顺序即优先级（后者覆盖前者）：边缘端量化模型（ONNX 实测） -> 本地手填
    report.config.__dict__.update(model_info.__dict__)     # 3.1 边缘端列（字段名与 ModelInfo 对齐）
    report.config.__dict__.update(local_info.__dict__)     # 3.1 云端列 + 3.2/3.3/3.4 配置值（说明列模板写死）

    # 第 4/5 章：精度与速度
    # 合并顺序即优先级（后者覆盖前者）：本地手填（云端原始模型指标） ->
    # 测评导出（metrics_summary.json，字段名与 s04/s05 对齐，覆盖边缘端列）

    # 4.3 边缘端列
    text = (OUTPUT_DIR / "metrics_summary.json").read_text(encoding="utf-8")
    eval_metrics = json.loads(text)
    report.accuracy.__dict__.update(local_info.__dict__)
    report.accuracy.__dict__.update(eval_metrics)
    # 4.5 边缘场景精度（evaluate_yolov8_onnx_scene.py 独立导出，字段名与 s04 对齐）
    text = (OUTPUT_DIR / "scene_metrics.json").read_text(encoding="utf-8")
    scene_metrics = json.loads(text)
    report.accuracy.__dict__.update(scene_metrics)
    # 按 2.2 节场景清单匹配 scenes 逐场景实测指标生成表格行，不手填
    report.accuracy.fill_scene_accuracy()

    # 5.1-5.3 边缘端列
    report.speed.__dict__.update(local_info.__dict__)
    report.speed.__dict__.update(eval_metrics)
    # 5.3 云端各阶段耗时占比：由 5.2 耗时计算，不手填
    report.speed.fill_cloud_ratios()
    # 5.4 Batch Size 性能（evaluate_yolov8_onnx_batch.py 导出，字段名与 s05 对齐）
    text = (OUTPUT_DIR / "batch_metrics.json").read_text(encoding="utf-8")
    batch_metrics = json.loads(text)
    report.speed.__dict__.update(batch_metrics)

    # 第 6 章：评测结果对比与分析
    # 6.1 / 6.3 / 6.4 小节内容（表格+分析）由大模型生成（提示词模板见
    # llm_analysis/prompts.json），原始输出保存到 output/llm_analysis.json
    report.analysis.fill_llm_analysis(
        local_info.__dict__, eval_metrics, batch_metrics, scene_metrics, OUTPUT_DIR
    )

    # 第 7 章：判定输入（云端列/目标值来自 local_info，
    # 边缘端列来自 metrics_summary.json），实测/损失/FPS/达标由公式计算；
    # 7.3 场景精度损失使用 scene_metrics.json 的逐场景指标
    report.verdict.__dict__.update(local_info.__dict__)
    report.verdict.__dict__.update(eval_metrics)
    report.verdict.__dict__.update(scene_metrics)
    report.verdict.fill_accuracy_rows()
    report.verdict.fill_perf_rows()
    report.verdict.fill_scene_loss()

    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 采集结果落盘，便于追溯报告 1.2/1.3 字段来源
    save_model_info(model_info, OUTPUT_DIR / "model_info.json")
    save_device_info(device_info, OUTPUT_DIR / "device_info.json")

    # empty_path = build_empty_report().save(OUTPUT_DIR / "report_empty.md")
    sample_path = build_sample_report().save(OUTPUT_DIR / "report_sample.md")

    # print(f"空参数报告（等价模板）: {empty_path}")
    print(f"示例填充报告: {sample_path}")


if __name__ == "__main__":
    main()

# for key, value in asdict(collect_device_info()).items():
#     print(f"{key}: {value!r}")