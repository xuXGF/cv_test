"""第 3 章：测试配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 3. 测试配置

## 3.1 模型配置

| 配置项        | 云端原始模型 | 边缘端量化模型 |
| ---------- | ------ | ------- |
| 模型名称       | {{cloud_model_name}} | {{edge_model_name}}  |
| 模型版本       | {{cloud_model_version}} | {{edge_model_version}}  |
| 模型格式       | {{cloud_model_format}} | {{edge_model_format}}  |
| 模型精度       | {{cloud_model_precision}} | {{edge_model_precision}}  |
| 模型大小       | {{cloud_model_size}} | {{edge_model_size}}  |
| 输入尺寸       | {{cloud_input_size}} | {{edge_input_size}}  |
| Batch Size | {{cloud_batch_size}} | {{edge_batch_size}}  |
| 参数量        | {{cloud_param_count}} | {{edge_param_count}}  |

## 3.2 图像参数

| 参数   | 配置值    | 说明     |
| ---- | ------ | ------ |
| 输入尺寸 | {{img_input_size}} | {{img_input_size_note}} |
| 输入通道 | {{img_channels}} | {{img_channels_note}} |
| 图像格式 | {{img_format}} | {{img_format_note}} |

## 3.3 推理参数

| 参数                   | 配置值    | 说明     |
| -------------------- | ------ | ------ |
| Confidence Threshold | {{conf_threshold}} | {{conf_threshold_note}} |
| NMS Threshold        | {{nms_threshold}} | {{nms_threshold_note}} |
| IoU Threshold        | {{iou_threshold}} | {{iou_threshold_note}} |
| Top-K                | {{top_k}} | {{top_k_note}} |
| 最大检测数量               | {{max_detections}} | {{max_detections_note}} |
| 推理线程数                | {{num_threads}} | {{num_threads_note}} |
| CPU/GPU/NPU模式        | {{device_mode}} | {{device_mode_note}} |
| 其他参数                 | {{other_infer_params}} | {{other_infer_params_note}} |

## 3.4 测试次数

| 测试项目   |   预热次数 | 正式测试次数 |  测试样本数 | 统计方式   |
| ------ | -----: | -----: | -----: | ------ |
| 精度测试   | {{acc_warmup}} | {{acc_rounds}} | {{acc_samples}} | {{acc_stat_method}} |
| 性能测试   | {{perf_warmup}} | {{perf_rounds}} | {{perf_samples}} | {{perf_stat_method}} |
| 边缘场景测试 | {{edge_warmup}} | {{edge_rounds}} | {{edge_samples}} | {{edge_stat_method}} |"""


@dataclass
class ConfigSection:
    """第 3 章：测试配置。"""

    # 3.1 模型配置
    cloud_model_name: str = ""  # 模型名称（云端原始模型）
    edge_model_name: str = ""  # 模型名称（边缘端量化模型）
    cloud_model_version: str = ""  # 模型版本（云端原始模型）
    edge_model_version: str = ""  # 模型版本（边缘端量化模型）
    cloud_model_format: str = ""  # 模型格式（云端原始模型）
    edge_model_format: str = ""  # 模型格式（边缘端量化模型）
    cloud_model_precision: str = ""  # 模型精度（云端原始模型）
    edge_model_precision: str = ""  # 模型精度（边缘端量化模型）
    cloud_model_size: str = ""  # 模型大小（云端原始模型）
    edge_model_size: str = ""  # 模型大小（边缘端量化模型）
    cloud_input_size: str = ""  # 输入尺寸（云端原始模型）
    edge_input_size: str = ""  # 输入尺寸（边缘端量化模型）
    cloud_batch_size: str = ""  # Batch Size（云端原始模型）
    edge_batch_size: str = ""  # Batch Size（边缘端量化模型）
    cloud_param_count: str = ""  # 参数量（云端原始模型）
    edge_param_count: str = ""  # 参数量（边缘端量化模型）

    # 3.2 图像参数
    img_input_size: str = ""  # 输入尺寸-配置值
    img_input_size_note: str = ""  # 输入尺寸-说明
    img_channels: str = ""  # 输入通道-配置值
    img_channels_note: str = ""  # 输入通道-说明
    img_format: str = ""  # 图像格式-配置值
    img_format_note: str = ""  # 图像格式-说明

    # 3.3 推理参数
    conf_threshold: str = ""  # Confidence Threshold-配置值
    conf_threshold_note: str = ""  # Confidence Threshold-说明
    nms_threshold: str = ""  # NMS Threshold-配置值
    nms_threshold_note: str = ""  # NMS Threshold-说明
    iou_threshold: str = ""  # IoU Threshold-配置值
    iou_threshold_note: str = ""  # IoU Threshold-说明
    top_k: str = ""  # Top-K-配置值
    top_k_note: str = ""  # Top-K-说明
    max_detections: str = ""  # 最大检测数量-配置值
    max_detections_note: str = ""  # 最大检测数量-说明
    num_threads: str = ""  # 推理线程数-配置值
    num_threads_note: str = ""  # 推理线程数-说明
    device_mode: str = ""  # CPU/GPU/NPU模式-配置值
    device_mode_note: str = ""  # CPU/GPU/NPU模式-说明
    other_infer_params: str = ""  # 其他参数-配置值
    other_infer_params_note: str = ""  # 其他参数-说明

    # 3.4 测试次数
    acc_warmup: str = ""  # 精度测试-预热次数
    acc_rounds: str = ""  # 精度测试-正式测试次数
    acc_samples: str = ""  # 精度测试-测试样本数
    acc_stat_method: str = ""  # 精度测试-统计方式
    perf_warmup: str = ""  # 性能测试-预热次数
    perf_rounds: str = ""  # 性能测试-正式测试次数
    perf_samples: str = ""  # 性能测试-测试样本数
    perf_stat_method: str = ""  # 性能测试-统计方式
    edge_warmup: str = ""  # 边缘场景测试-预热次数
    edge_rounds: str = ""  # 边缘场景测试-正式测试次数
    edge_samples: str = ""  # 边缘场景测试-测试样本数
    edge_stat_method: str = ""  # 边缘场景测试-统计方式

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))