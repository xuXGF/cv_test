"""第 3 章：测试配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 3. 测试配置

## 3.1 模型配置

| 配置项        | 云端原始模型 | 边缘端量化模型 |
| ---------- | ------ | ------- |
| 模型名称       | {{cloud_model_name}} | {{model_name}}  |
| 模型版本       | {{cloud_model_version}} | {{model_version}}  |
| 模型格式       | {{cloud_model_format}} | {{edge_model_format}}  |
| 模型精度       | {{cloud_model_precision}} | {{model_precision}}  |
| 模型大小       | {{cloud_model_size}} | {{model_size}}  |
| 输入尺寸       | {{cloud_input_size}} | {{input_size}}  |
| Batch Size | {{cloud_batch_size}} | {{batch_size}}  |
| 参数量        | {{cloud_param_count}} | {{param_count}}  |

## 3.2 图像参数

| 参数   | 配置值    | 说明     |
| ---- | ------ | ------ |
| 输入尺寸 | {{img_input_size}} | 推理前将图像缩放至模型输入尺寸 |
| 输入通道 | {{img_channels}} | RGB 三通道 |
| 图像格式 | {{img_format}} | JPEG / PNG |

## 3.3 推理参数

| 参数                   | 配置值    | 说明     |
| -------------------- | ------ | ------ |
| Confidence Threshold | {{conf_threshold}} | 置信度阈值，低于该值的检测结果会被忽略 |
| NMS Threshold        | {{nms_threshold}} | 重复检测结果的过滤强度，值越小过滤越严格 |
| IoU Threshold        | {{iou_threshold}} | 判断两个检测框是否重叠过多的标准 |
| CPU/GPU/NPU模式        | {{device_mode}} | 选择使用 CPU、显卡还是专用芯片来运行模型 |
| 其他参数                 | {{other_infer_params}} | 其他自定义推理参数 |

## 3.4 测试次数

| 测试项目   |   预热次数 | 正式测试次数 |  测试样本数 | 统计方式   |
| ------ | -----: | -----: | -----: | ------ |
| 精度测试   | {{acc_warmup}} | {{acc_rounds}} | {{acc_samples}} | {{acc_stat_method}} |
| 性能测试   | {{perf_warmup}} | {{perf_rounds}} | {{perf_samples}} | {{perf_stat_method}} |
| 边缘场景测试 | {{edge_warmup}} | {{edge_rounds}} | {{edge_samples}} | {{edge_stat_method}} |"""


@dataclass
class ConfigSection:
    """第 3 章：测试配置。

    填充方式：

    * 3.1「云端原始模型」列与 3.2/3.3/3.4 的配置值：由 ``local_info.json``
      手动填写，键名与本类字段名一致。
    * 3.1「边缘端量化模型」列：字段名与 ``ModelInfo`` 一致
      （``model_name``/``model_version``/``edge_model_format``/``model_precision``/
      ``model_size``/``input_size``/``batch_size``/``param_count``），
      可通过 ``report.config.__dict__.update(model_info.__dict__)`` 直接填充。
    * 3.2/3.3 的「说明」列为模板写死文案，不经 JSON 填充。
    """

    # 3.1 模型配置——云端原始模型（local_info.json 手动填写）
    cloud_model_name: str = ""  # 模型名称
    cloud_model_version: str = ""  # 模型版本
    cloud_model_format: str = ""  # 模型格式
    cloud_model_precision: str = ""  # 模型精度
    cloud_model_size: str = ""  # 模型大小
    cloud_input_size: str = ""  # 输入尺寸
    cloud_batch_size: str = ""  # Batch Size
    cloud_param_count: str = ""  # 参数量

    # 3.1 模型配置——边缘端量化模型（字段名与 ModelInfo 对齐，自动填充）
    model_name: str = ""  # 模型名称
    model_version: str = ""  # 模型版本
    edge_model_format: str = ""  # 模型格式（ModelInfo.edge_model_format，ONNX）
    model_precision: str = ""  # 模型精度（ModelInfo.model_precision）
    model_size: str = ""  # 模型大小
    input_size: str = ""  # 输入尺寸
    batch_size: str = ""  # Batch Size
    param_count: str = ""  # 参数量

    # 3.2 图像参数（配置值由 local_info.json 填写，说明列模板写死）
    img_input_size: str = ""  # 输入尺寸-配置值
    img_channels: str = ""  # 输入通道-配置值
    img_format: str = ""  # 图像格式-配置值

    # 3.3 推理参数（配置值由 local_info.json 填写，说明列模板写死）
    conf_threshold: str = ""  # Confidence Threshold-配置值
    nms_threshold: str = ""  # NMS Threshold-配置值
    iou_threshold: str = ""  # IoU Threshold-配置值
    device_mode: str = ""  # CPU/GPU/NPU模式-配置值
    other_infer_params: str = ""  # 其他参数-配置值

    # 3.4 测试次数（由 local_info.json 填写）
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
