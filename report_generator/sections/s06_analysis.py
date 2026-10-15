"""第 6 章：评测结果对比与分析。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..render import render

_TEMPLATE = r"""# 6. 评测结果对比与分析

## 6.1 云端原始模型与边缘端量化模型精度对比

| 指标     |   原始模型 |   量化模型 |    变化值 |    变化率 | 分析     |
| ------ | -----: | -----: | -----: | -----: | ------ |
| 主要精度指标 | {{ov_main_orig}} | {{ov_main_quant}} | {{ov_main_delta}} | {{ov_main_rate}} | {{ov_main_analysis}} |
| 其他指标   | {{ov_other_orig}} | {{ov_other_quant}} | {{ov_other_delta}} | {{ov_other_rate}} | {{ov_other_analysis}} |

**分析：**

{{ana_61=\[填写云端原始模型与边缘端量化模型的精度差异及变化原因。]}}

## 6.2 不同测试场景下的评测结果对比

| 测试场景    |    样本量 | 原始模型指标 | 量化模型指标 |   指标变化 | 是否达标   |
| ------- | -----: | -----: | -----: | -----: | ------ |
| **场景1** | {{sc6_01_samples}} | {{sc6_01_orig}} | {{sc6_01_quant}} | {{sc6_01_delta}} | {{sc6_01_ok}} |
| **场景2** | {{sc6_02_samples}} | {{sc6_02_orig}} | {{sc6_02_quant}} | {{sc6_02_delta}} | {{sc6_02_ok}} |
| **场景3** | {{sc6_03_samples}} | {{sc6_03_orig}} | {{sc6_03_quant}} | {{sc6_03_delta}} | {{sc6_03_ok}} |
| **场景4** | {{sc6_04_samples}} | {{sc6_04_orig}} | {{sc6_04_quant}} | {{sc6_04_delta}} | {{sc6_04_ok}} |
| **场景5** | {{sc6_05_samples}} | {{sc6_05_orig}} | {{sc6_05_quant}} | {{sc6_05_delta}} | {{sc6_05_ok}} |

## 6.3 精度损失分析

| 指标     |   原始模型 |   量化模型 |   精度损失 | 主要影响场景 | 原因分析   |
| ------ | -----: | -----: | -----: | ------ | ------ |
| {{loss_1_metric}} | {{loss_1_orig}} | {{loss_1_quant}} | {{loss_1_loss}} | {{loss_1_scene}} | {{loss_1_reason}} |
| {{loss_2_metric}} | {{loss_2_orig}} | {{loss_2_quant}} | {{loss_2_loss}} | {{loss_2_scene}} | {{loss_2_reason}} |

**分析：**

{{ana_63=\[填写量化过程中出现精度损失的具体表现及原因。]}}

## 6.4 推理性能分析

| 分析维度       | 测试结果   | 分析     |
| ---------- | ------ | ------ |
| 平均推理耗时     | {{perf_avg_latency}} | {{perf_avg_latency_analysis}} |
| FPS        | {{perf_fps}} | {{perf_fps_analysis}} |
| 预处理耗时      | {{perf_pre}} | {{perf_pre_analysis}} |
| 模型推理耗时     | {{perf_infer}} | {{perf_infer_analysis}} |
| 后处理耗时      | {{perf_post}} | {{perf_post_analysis}} |
| CPU利用率     | {{perf_cpu}} | {{perf_cpu_analysis}} |
| GPU/NPU利用率 | {{perf_npu}} | {{perf_npu_analysis}} |
| 内存占用       | {{perf_mem}} | {{perf_mem_analysis}} |

## 6.5 边缘场景泛化表现分析

| 边缘场景 | 标准场景指标 | 边缘场景指标 |   性能变化 | 泛化表现   |
| ---- | -----: | -----: | -----: | ------ |
| 不同光照 | {{gen_light_std}} | {{gen_light_edge}} | {{gen_light_delta}} | {{gen_light_perf}} |
| 不同角度 | {{gen_angle_std}} | {{gen_angle_edge}} | {{gen_angle_delta}} | {{gen_angle_perf}} |

**分析：**

{{ana_65=\[填写模型在真实边缘场景下的泛化表现。]}}

## 6.6 性能瓶颈分析

| 瓶颈环节 |     耗时 |     占比 | 问题描述   | 可能原因   |
| ---- | -----: | -----: | ------ | ------ |
| 预处理  | {{bn_pre_time}} | {{bn_pre_ratio}} | {{bn_pre_desc}} | {{bn_pre_reason}} |
| 模型推理 | {{bn_infer_time}} | {{bn_infer_ratio}} | {{bn_infer_desc}} | {{bn_infer_reason}} |
| 后处理  | {{bn_post_time}} | {{bn_post_ratio}} | {{bn_post_desc}} | {{bn_post_reason}} |
| 数据传输 | {{bn_data_time}} | {{bn_data_ratio}} | {{bn_data_desc}} | {{bn_data_reason}} |
| 其他   | {{bn_other_time}} | {{bn_other_ratio}} | {{bn_other_desc}} | {{bn_other_reason}} |

**性能瓶颈结论：**

{{ana_66=\[填写主要性能瓶颈及影响。]}}"""


@dataclass
class AnalysisSection:
    """第 6 章：评测结果对比与分析。"""

    # 6.1 精度对比
    ov_main_orig: str = ""  # 主要精度指标-原始模型
    ov_main_quant: str = ""  # 主要精度指标-量化模型
    ov_main_delta: str = ""  # 主要精度指标-变化值
    ov_main_rate: str = ""  # 主要精度指标-变化率
    ov_main_analysis: str = ""  # 主要精度指标-分析
    ov_other_orig: str = ""  # 其他指标-原始模型
    ov_other_quant: str = ""  # 其他指标-量化模型
    ov_other_delta: str = ""  # 其他指标-变化值
    ov_other_rate: str = ""  # 其他指标-变化率
    ov_other_analysis: str = ""  # 其他指标-分析
    ana_61: str = ""  # 6.1 分析正文

    # 6.2 不同测试场景下的评测结果对比
    sc6_01_samples: str = ""  # 场景1-样本量
    sc6_01_orig: str = ""  # 场景1-原始模型指标
    sc6_01_quant: str = ""  # 场景1-量化模型指标
    sc6_01_delta: str = ""  # 场景1-指标变化
    sc6_01_ok: str = ""  # 场景1-是否达标
    sc6_02_samples: str = ""  # 场景2-样本量
    sc6_02_orig: str = ""  # 场景2-原始模型指标
    sc6_02_quant: str = ""  # 场景2-量化模型指标
    sc6_02_delta: str = ""  # 场景2-指标变化
    sc6_02_ok: str = ""  # 场景2-是否达标
    sc6_03_samples: str = ""  # 场景3-样本量
    sc6_03_orig: str = ""  # 场景3-原始模型指标
    sc6_03_quant: str = ""  # 场景3-量化模型指标
    sc6_03_delta: str = ""  # 场景3-指标变化
    sc6_03_ok: str = ""  # 场景3-是否达标
    sc6_04_samples: str = ""  # 场景4-样本量
    sc6_04_orig: str = ""  # 场景4-原始模型指标
    sc6_04_quant: str = ""  # 场景4-量化模型指标
    sc6_04_delta: str = ""  # 场景4-指标变化
    sc6_04_ok: str = ""  # 场景4-是否达标
    sc6_05_samples: str = ""  # 场景5-样本量
    sc6_05_orig: str = ""  # 场景5-原始模型指标
    sc6_05_quant: str = ""  # 场景5-量化模型指标
    sc6_05_delta: str = ""  # 场景5-指标变化
    sc6_05_ok: str = ""  # 场景5-是否达标

    # 6.3 精度损失分析
    loss_1_metric: str = ""  # 第1行-指标
    loss_1_orig: str = ""  # 第1行-原始模型
    loss_1_quant: str = ""  # 第1行-量化模型
    loss_1_loss: str = ""  # 第1行-精度损失
    loss_1_scene: str = ""  # 第1行-主要影响场景
    loss_1_reason: str = ""  # 第1行-原因分析
    loss_2_metric: str = ""  # 第2行-指标
    loss_2_orig: str = ""  # 第2行-原始模型
    loss_2_quant: str = ""  # 第2行-量化模型
    loss_2_loss: str = ""  # 第2行-精度损失
    loss_2_scene: str = ""  # 第2行-主要影响场景
    loss_2_reason: str = ""  # 第2行-原因分析
    ana_63: str = ""  # 6.3 分析正文

    # 6.4 推理性能分析
    perf_avg_latency: str = ""  # 平均推理耗时-测试结果
    perf_avg_latency_analysis: str = ""  # 平均推理耗时-分析
    perf_fps: str = ""  # FPS-测试结果
    perf_fps_analysis: str = ""  # FPS-分析
    perf_pre: str = ""  # 预处理耗时-测试结果
    perf_pre_analysis: str = ""  # 预处理耗时-分析
    perf_infer: str = ""  # 模型推理耗时-测试结果
    perf_infer_analysis: str = ""  # 模型推理耗时-分析
    perf_post: str = ""  # 后处理耗时-测试结果
    perf_post_analysis: str = ""  # 后处理耗时-分析
    perf_cpu: str = ""  # CPU利用率-测试结果
    perf_cpu_analysis: str = ""  # CPU利用率-分析
    perf_npu: str = ""  # GPU/NPU利用率-测试结果
    perf_npu_analysis: str = ""  # GPU/NPU利用率-分析
    perf_mem: str = ""  # 内存占用-测试结果
    perf_mem_analysis: str = ""  # 内存占用-分析

    # 6.5 边缘场景泛化表现分析
    gen_light_std: str = ""  # 不同光照-标准场景指标
    gen_light_edge: str = ""  # 不同光照-边缘场景指标
    gen_light_delta: str = ""  # 不同光照-性能变化
    gen_light_perf: str = ""  # 不同光照-泛化表现
    gen_angle_std: str = ""  # 不同角度-标准场景指标
    gen_angle_edge: str = ""  # 不同角度-边缘场景指标
    gen_angle_delta: str = ""  # 不同角度-性能变化
    gen_angle_perf: str = ""  # 不同角度-泛化表现
    ana_65: str = ""  # 6.5 分析正文

    # 6.6 性能瓶颈分析
    bn_pre_time: str = ""  # 预处理-耗时
    bn_pre_ratio: str = ""  # 预处理-占比
    bn_pre_desc: str = ""  # 预处理-问题描述
    bn_pre_reason: str = ""  # 预处理-可能原因
    bn_infer_time: str = ""  # 模型推理-耗时
    bn_infer_ratio: str = ""  # 模型推理-占比
    bn_infer_desc: str = ""  # 模型推理-问题描述
    bn_infer_reason: str = ""  # 模型推理-可能原因
    bn_post_time: str = ""  # 后处理-耗时
    bn_post_ratio: str = ""  # 后处理-占比
    bn_post_desc: str = ""  # 后处理-问题描述
    bn_post_reason: str = ""  # 后处理-可能原因
    bn_data_time: str = ""  # 数据传输-耗时
    bn_data_ratio: str = ""  # 数据传输-占比
    bn_data_desc: str = ""  # 数据传输-问题描述
    bn_data_reason: str = ""  # 数据传输-可能原因
    bn_other_time: str = ""  # 其他-耗时
    bn_other_ratio: str = ""  # 其他-占比
    bn_other_desc: str = ""  # 其他-问题描述
    bn_other_reason: str = ""  # 其他-可能原因
    ana_66: str = ""  # 性能瓶颈结论正文

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))