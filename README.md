# 边缘端模型测评工具

面向边缘端 AI 模型部署场景的自动化测评工具：自动采集**模型信息、设备硬件信息、数据集信息**，结合手动填写的本地信息，一键生成符合标准模板的 **Markdown 测评报告**。

## 功能特性

- **模型信息采集**：解析 ONNX 模型元数据（参数量、FLOPs、输入尺寸、类别数等），并基于 `onnxruntime` 完成推理基准测试（预处理/推理/后处理耗时、FPS、检测明细）
- **设备信息采集**：自动识别平台（Linux 边缘设备 / macOS），采集 SoC、CPU、GPU、NPU、内存、存储等硬件信息
- **数据集扫描与预处理**：扫描数据集目录统计基本信息；支持 COCO → YOLO 标注转换、图片格式转换、尺寸统一（letterbox）、数据清洗、异常数据处理
- **报告生成**：按 9 章模板结构渲染 Markdown 报告，未采集到的字段保持空白，不自动填充虚假值

## 环境要求

- Python 3.10+
- 依赖库：`onnxruntime`、`opencv-python`、`numpy`

```bash
pip install onnxruntime opencv-python numpy
```

## 项目结构

```text
code/
├── run.py                  # 主入口：采集全部信息并生成测评报告
├── generate_report.py      # 示例入口：演示按模板生成空报告/手填报告
├── run_test.py             # model_info 模块快速测试脚本
├── template.md             # 测评报告标准模板（9 章）
├── report_generator/       # 报告生成器
│   ├── report.py           #   TestReport 报告对象
│   ├── render.py           #   Markdown 渲染
│   └── sections/           #   按章节拆分的 9 个 Section 类
├── model_info/             # ONNX 模型信息采集与推理基准测试（自带 yolov8n.onnx、test.jpg）
├── device_info/            # 设备硬件信息采集（Linux 边缘设备 / macOS）
├── dataset_info/           # 数据集扫描、COCO->YOLO 转换、预处理（自带 data/coco128）
├── local_info/             # 本地手填信息（local_info.json）
├── tools/                  # 通用小工具（如参数量格式化 format_params）
└── output/                 # 报告输出目录
```

## 快速开始

```bash
# 一键采集模型/设备/数据集信息并生成报告 -> output/report_sample.md
python run.py
```

各模块也可单独使用：

```bash
# 查看当前设备硬件信息
python -m device_info

# 打印模型元数据并运行推理基准测试（缺省使用包内自带 yolov8n.onnx 与 test.jpg）
python -m model_info
python -m model_info <模型路径> [图片路径]   # 指定模型与测试图片

# 数据集扫描与预处理（缺省使用包内自带 data/coco128）
python -m dataset_info scan [数据集目录] [--coco-json 路径]     # 扫描并打印报告字段
python -m dataset_info convert [目录] [--coco-json 路径]        # COCO 标注 -> YOLO txt
python -m dataset_info convert-format [目录] [--to .jpg] [--replace]  # 图片格式统一
python -m dataset_info resize [目录] --size 640x640 [--letterbox]     # 尺寸统一
python -m dataset_info clean [目录] [--apply]                   # 数据清洗（缺省仅检测）
python -m dataset_info anomaly [目录] [--apply]                 # 异常数据处理（缺省仅检测）

# 查看本地手填信息
python -m local_info

# 生成空模板报告（验证与 template.md 一致）
python generate_report.py
```

## 工作原理

`run.py` 的数据流：

```text
model_info    模型元数据（1.2 算法信息 / 1.4 测试环境 / 3.1 模型配置）
device_info   设备硬件信息（1.3 测试设备信息）──┐
dataset_info  数据集扫描（2.1 / 2.3 章节）      ├──> TestReport ──> Markdown 报告
local_info    本地手填信息（算法类型、数据集名称、2.2 边缘场景等）
```

字段合并顺序即优先级（后者覆盖前者）：**模型侧字段 → 数据集扫描 → 本地手填**。

## 测评报告章节结构

报告按 `template.md` 的 9 章结构组织，每章对应 `report_generator/sections/` 下的一个类：

| 章节 | 内容 | Section 类 |
| ---- | ---- | ---- |
| 1 | 报告基本信息（1.1 基本信息 / 1.2 算法信息 / 1.3 测试设备 / 1.4 测试环境） | `BasicInfoSection` |
| 2 | 测试数据集说明（2.1 基本信息 / 2.2 边缘场景 / 2.3 标签格式与预处理） | `DatasetSection` |
| 3 | 测试配置 | `ConfigSection` |
| 4 | 精度评测 | `AccuracySection` |
| 5 | 推理速度评测 | `SpeedSection` |
| 6 | 分析 | `AnalysisSection` |
| 7 | 测评结论 | `VerdictSection` |
| 8 | 建议 | `SuggestionSection` |
| 9 | 附件 | `AttachmentSection` |

程序自动填充第 1、2、3、5 章中可采集的字段；精度评测、测评结论等需要人工介入的章节保留空白待填写。

## 本地信息配置

编辑 [local_info/local_info.json](local_info/local_info.json)，填写无法自动采集、只能人工确认的信息：

```json
{
    "algo_type": "目标检测",
    "dataset_name": "coco数据集",
    "data_source": "网络公开数据集",
    "data_type": "图像"
}
```

要求：顶层必须是 JSON 对象，每个键会成为 `LocalInfo` 对象的属性，键需为合法 Python 标识符（不能是关键字）；文件不存在或 JSON 非法时抛出明确异常，不做静默兜底。

## 设计约定

- **不填虚假值**：采集不到的字段保持空字符串，绝不自动填充占位数据
- **破坏性操作默认 dry_run**：数据清洗、异常处理等会修改文件的操作，缺省只检测报告，需显式传 `--apply` / `--replace` 才实际执行
- **字段名对齐报告**：各采集模块的字段名与 `report_generator.sections` 中对应章节字段一一对应，可直接通过 `report.xxx.__dict__.update(info.__dict__)` 合并
