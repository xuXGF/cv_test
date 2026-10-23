# 边缘端模型测评工具

面向边缘端 AI 模型部署场景的自动化测评工具：自动采集**模型信息、设备硬件信息、数据集信息**，配合独立的**精度 / 性能 / 边缘场景评测脚本**与**大模型分析**，结合手动填写的本地信息，一键生成符合标准模板的 **Markdown 测评报告**。

## 功能特性

- **模型信息采集**：解析 ONNX 模型元数据（参数量、精度、输入尺寸、类别数、opset 等），并基于 `onnxruntime` 完成单图推理基准测试（预处理/推理/后处理耗时、FPS）
- **设备信息采集**：自动识别平台（Linux 边缘设备 / macOS），采集 SoC、CPU、GPU、NPU、内存、存储等硬件信息
- **数据集扫描与预处理**：扫描数据集目录统计基本信息；支持 COCO → YOLO 标注转换、图片格式转换、尺寸统一（letterbox）、数据清洗、异常数据处理
- **精度评测**：基于 `evaluate_yolov8_onnx_manual.py` 在全量测试集上计算 mAP@0.5、mAP@0.5:0.95、Precision、Recall，导出整体指标、逐类别明细与逐图耗时
- **性能评测**：`evaluate_yolov8_onnx_batch.py` 测试 Batch Size = 1 / 4 / 8 下的延迟、FPS、CPU 利用率、GPU/NPU 利用率与内存占用
- **边缘场景评测**：`evaluate_yolov8_onnx_scene.py` 复用同一评测器，对 2.2 节各场景子集逐场景实测精度，口径与主数据集一致
- **大模型分析**：调用大模型生成 6.1 / 6.3 / 6.4 分析、7.4 综合判定、8.1 / 8.2 优化建议
- **报告生成**：按 9 章模板渲染 Markdown 报告，未采集到的字段保持空白，不自动填充虚假值

## 环境要求

- Python 3.10+
- 依赖库：`onnxruntime`、`opencv-python`、`numpy`、`requests`、`psutil`

```bash
pip install onnxruntime opencv-python numpy requests psutil
```

## 项目结构

```text
code/
├── run.py                                # 主入口：汇总采集/评测结果并生成测评报告
├── generate_report.py                    # 示例入口：演示按模板生成空报告/手填报告
├── run_test.py                           # 大模型调用示例脚本
├── template.md                           # 测评报告标准模板（9 章）
├── report_generator/                     # 报告生成器
│   ├── report.py                         #   TestReport 报告对象
│   ├── render.py                         #   Markdown 渲染
│   └── sections/                         #   按章节拆分的 9 个 Section 类
├── model_info/                           # 模型信息采集与评测（自带 yolov8n.onnx、test.jpg）
│   ├── info.py                           #   模型元数据采集
│   ├── yolov8_onnx_infer.py              #   单图推理基准测试
│   ├── evaluate_yolov8_onnx_manual.py    #   全量精度评测（4.3 / 5 章数据源）
│   ├── evaluate_yolov8_onnx_batch.py     #   Batch Size 性能评测（5.4）
│   └── evaluate_yolov8_onnx_scene.py     #   边缘场景精度评测（4.5 / 7.3）
├── device_info/                          # 设备硬件信息采集（Linux / macOS）
├── dataset_info/                         # 数据集扫描、COCO->YOLO 转换、预处理（自带 data/coco128）
├── local_info/                           # 本地手填信息（local_info.json）
├── llm_analysis/                         # 大模型调用封装（llm_config.json / prompts.json）
├── tools/                                # 通用计算与格式化工具（format_params / to_float / pct 等）
└── output/                               # 报告与评测结果输出目录
```

## 快速开始

```bash
# 1. 全量精度评测 -> output/metrics_summary.json、per_class_metrics.csv、per_image_timings.csv
cd model_info && python3 evaluate_yolov8_onnx_manual.py && cd ..

# 2. Batch Size 性能评测 -> output/batch_metrics.json
cd model_info && python3 evaluate_yolov8_onnx_batch.py && cd ..

# 3. 边缘场景精度评测 -> output/scene_metrics.json
python3 model_info/evaluate_yolov8_onnx_scene.py

# 4. 汇总采集与评测结果，调用大模型分析并生成报告 -> output/report_sample.md
python3 run.py

# 生成空模板报告（验证与 template.md 一致）
python3 generate_report.py
```

> 前两个评测脚本的路径参数按 `model_info` 目录解析，需在该目录下运行；场景脚本基于自身位置解析路径，可在任意目录运行。
> `run.py` 只负责汇总与渲染，不会自动执行评测脚本，需先完成步骤 1~3。

各信息采集模块（`model_info` / `device_info` / `dataset_info` / `local_info`）通过 Python 代码调用，用法见各包 `__init__.py` 的 docstring。

## 工作原理

`run.py` 的数据流：

```text
model_info       模型元数据（1.2 算法信息 / 1.4 测试环境 / 3.1 边缘端列）
device_info      设备硬件信息（1.3 测试设备信息）
dataset_info     数据集扫描（2.1 / 2.3）
local_info       本地手填信息（2.2 场景、3.1 云端列、3.2~3.4 配置、云端指标与目标值）
metrics_summary.json   全量精度/速度（4.3 / 5.1~5.3 / 7.1 / 7.2）
batch_metrics.json     批量性能（5.4）
scene_metrics.json     边缘场景精度（4.5 / 7.3）
        │
        ├──> TestReport（字段合并、公式计算）──> Markdown 报告
        │
        └──> 整份报告正文 ──> 大模型（6.1 / 6.3 / 6.4 / 7.4 / 8.1 / 8.2）──> output/llm_analysis.json
```

字段合并顺序即优先级（后者覆盖前者）：**模型侧字段 → 数据集扫描 → 本地手填 → 评测导出**。

> 第 6、7、8 章的大模型调用以**完整报告正文**为输入，7.4 与 8.1 / 8.2 只渲染一次，避免长输入被截断。

## 测评报告章节结构

报告按 `template.md` 的 9 章结构组织，每章对应 `report_generator/sections/` 下的一个类：

| 章节 | 内容 | 填充方式 | Section 类 |
| ---- | ---- | ---- | ---- |
| 1 | 报告基本信息（1.1 基本信息 / 1.2 算法信息 / 1.3 测试设备 / 1.4 测试环境） | 自动采集 + 手填 | `BasicInfoSection` |
| 2 | 测试数据集说明（2.1 基本信息 / 2.2 边缘场景 / 2.3 预处理） | 扫描 + 手填，2.2 数据量/占比自动计算 | `DatasetSection` |
| 3 | 测试配置（3.1 模型配置 / 3.2 图像参数 / 3.3 推理参数 / 3.4 测试次数） | 边缘端列自动采集，云端列与配置值手填 | `ConfigSection` |
| 4 | 算法精度评测（4.1 指标 / 4.2 分类 / 4.3 检测 / 4.4 分割 / 4.5 边缘场景） | 评测脚本导出 + 手填云端值 | `AccuracySection` |
| 5 | 推理速度评测（5.1 FPS / 5.2 分段耗时 / 5.3 占比 / 5.4 Batch Size） | 评测脚本导出 + 手填云端值，占比自动计算 | `SpeedSection` |
| 6 | 评测结果对比与分析（6.1 精度对比 / 6.3 场景泛化 / 6.4 推理性能） | 大模型生成 | `AnalysisSection` |
| 7 | 评测结果判定（7.1 精度达标 / 7.2 性能达标 / 7.3 场景损失 / 7.4 综合判定） | 公式计算 + 大模型生成 | `VerdictSection` |
| 8 | 量化调优建议（8.1 精度优化 / 8.2 推理性能优化） | 大模型生成 | `SuggestionSection` |
| 9 | 测试数据及附件（9.1~9.5） | 手填 + 自动生成时间/路径 | `AttachmentSection` |

所有百分比、占比、精度损失、是否达标均由公式计算生成，不手填；计算逻辑位于各 Section 类，通用方法在 `tools/tool.py`。

## 本地信息配置

编辑 [local_info/local_info.json](local_info/local_info.json)，填写无法自动采集、只能人工确认的信息：

```json
{
    "algo_type": "目标检测",
    "dataset_name": "coco数据集",
    "data_source": "网络公开数据集",
    "data_type": "图像",

    "scene": {
        "path": ["dataset_info/data/scene/scene_1", "dataset_info/data/scene/scene_2"],
        "desc": ["逆光、玻璃强反光及雪地强光等复杂光照场景", "逆光、玻璃强反光复杂光照场景"],
        "cloud_map05": ["0.7229", "1.000"],
        "cloud_map95": ["0.5701", "0.8165"],
        "cloud_precision": ["0.6375", "0.7286"],
        "cloud_recall": ["0.6375", "1.000"]
    },

    "cloud_model_name": "yolov8n",
    "cloud_model_format": "PyTorch (.pt)",
    "cloud_param_count": "3,157,200",

    "det_cloud_map05": "0.6231",
    "v_map05_target": "0.6044",
    "v_latency_target": "39.619",

    "conf_threshold": "0.25",
    "nms_threshold": "0.45"
}
```

字段分组说明：

| 分组 | 代表字段 | 用途 |
| ---- | ---- | ---- |
| 基础信息 | `algo_type` / `dataset_name` / `data_source` / `data_type` | 1.1、1.2、2.1 |
| 边缘场景 | `scene.path` / `scene.desc` / `scene.cloud_*` | 2.2、4.5、7.3（`path` 与 `desc` 按下标一一对应） |
| 云端模型 | `cloud_model_*` / `cloud_input_size` / `cloud_batch_size` | 3.1 云端原始模型列 |
| 云端精度 | `det_cloud_*` | 4.3 云端列、6.1、7.1 |
| 判定目标 | `v_*_target` | 7.1 / 7.2 达标判定 |
| 云端速度 | `speed_cloud_*` / `stage_cloud_*` | 5.1~5.3 云端列 |
| 图像/推理参数 | `img_*` / `conf_threshold` / `nms_threshold` / `iou_threshold` / `device_mode` | 3.2 / 3.3 |
| 测试次数 | `acc_*` / `perf_*` / `edge_*` | 3.4 |

要求：顶层必须是 JSON 对象，每个键会成为 `LocalInfo` 对象的属性，键需为合法 Python 标识符（不能是关键字）；文件不存在或 JSON 非法时抛出明确异常，不做静默兜底。

## 大模型配置

端点配置见 [llm_analysis/llm_config.json](llm_analysis/llm_config.json)，`provider` 在 `ollama`（`POST {base_url}/api/chat`）与 `openai`（OpenAI 兼容 `POST {base_url}/chat/completions`）之间切换；必填 `base_url` 与 `model`，OpenAI 还需 `api_key`，可选 `timeout`、`num_ctx`。

提示词模板集中在 [llm_analysis/prompts.json](llm_analysis/prompts.json)，占位符写作 `{{参数名}}`：

| 提示词 | 输出章节 |
| ---- | ---- |
| `model_compare` | 6.1 精度对比分析 |
| `scene_generalization` | 6.3 边缘场景泛化表现分析 |
| `performance_analysis` | 6.4 推理性能分析 |
| `verdict_analysis` | 7.4 评测结果综合判定 |
| `accuracy_optimization` | 8.1 精度优化建议 |
| `performance_optimization` | 8.2 推理性能优化建议 |

所有大模型原始输出汇总保存到 `output/llm_analysis.json`；接口超时、非 2xx、响应缺字段均抛出明确异常，不自动重试、不静默兜底。

## 输出文件

| 文件 | 内容 | 生成方 |
| ---- | ---- | ---- |
| `output/report_sample.md` | 标准化测评报告 | `run.py` |
| `output/model_info.json` | 模型元数据 | `run.py` |
| `output/device_info.json` | 设备硬件信息 | `run.py` |
| `output/metrics_summary.json` | 边缘端精度与速度指标汇总 | `evaluate_yolov8_onnx_manual.py` |
| `output/per_class_metrics.csv` | 逐类别精度明细 | `evaluate_yolov8_onnx_manual.py` |
| `output/per_image_timings.csv` | 逐图分段耗时明细 | `evaluate_yolov8_onnx_manual.py` |
| `output/batch_metrics.json` | Batch Size 性能评测结果 | `evaluate_yolov8_onnx_batch.py` |
| `output/scene_metrics.json` | 边缘场景精度评测结果 | `evaluate_yolov8_onnx_scene.py` |
| `output/llm_analysis.json` | 大模型分析与建议原始输出 | `run.py`（调用大模型） |

## 设计约定

- **不填虚假值**：采集不到的字段保持空字符串，绝不自动填充占位数据
- **口径统一**：精度评测、性能评测、边缘场景评测均复用同一套预处理/后处理与指标计算逻辑
- **破坏性操作默认 dry_run**：数据清洗、异常处理等会修改文件的操作，缺省只检测报告，需显式传 `--apply` / `--replace` 才实际执行
- **字段名对齐报告**：各采集模块的字段名与 `report_generator.sections` 中对应章节字段一一对应，可直接通过 `report.xxx.__dict__.update(info.__dict__)` 合并
