"""第 6 章：评测结果对比与分析。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_analysis import LLMAnalyzer

from ..render import render

_TEMPLATE = r"""# 6. 评测结果对比与分析

## 6.1 云端原始模型与边缘端量化模型精度对比

{{ana_61=\[填写云端原始模型与边缘端量化模型的精度对比分析，包含指标对比表格与分析结论。]}}

## 6.2 不同测试场景下的评测结果对比

| 测试场景    |    样本量 | 原始模型指标 | 量化模型指标 |   指标变化 | 是否达标   |
| ------- | -----: | -----: | -----: | -----: | ------ |
| **场景1** | {{sc6_01_samples}} | {{sc6_01_orig}} | {{sc6_01_quant}} | {{sc6_01_delta}} | {{sc6_01_ok}} |
| **场景2** | {{sc6_02_samples}} | {{sc6_02_orig}} | {{sc6_02_quant}} | {{sc6_02_delta}} | {{sc6_02_ok}} |
| **场景3** | {{sc6_03_samples}} | {{sc6_03_orig}} | {{sc6_03_quant}} | {{sc6_03_delta}} | {{sc6_03_ok}} |
| **场景4** | {{sc6_04_samples}} | {{sc6_04_orig}} | {{sc6_04_quant}} | {{sc6_04_delta}} | {{sc6_04_ok}} |
| **场景5** | {{sc6_05_samples}} | {{sc6_05_orig}} | {{sc6_05_quant}} | {{sc6_05_delta}} | {{sc6_05_ok}} |

## 6.3 边缘场景泛化表现分析

| 边缘场景 | 标准场景指标 | 边缘场景指标 |   性能变化 | 泛化表现   |
| ---- | -----: | -----: | -----: | ------ |
| 不同光照 | {{gen_light_std}} | {{gen_light_edge}} | {{gen_light_delta}} | {{gen_light_perf}} |
| 不同角度 | {{gen_angle_std}} | {{gen_angle_edge}} | {{gen_angle_delta}} | {{gen_angle_perf}} |

**分析：**

{{ana_63=\[填写模型在真实边缘场景下的泛化表现。]}}

## 6.4 推理性能分析

{{ana_64=\[填写推理性能分析，包含性能指标表格与性能瓶颈结论。]}}"""


def _accuracy_row(label: str, metrics: Mapping[str, Any], prefix: str) -> str:
    """按指标前缀取 mAP/Precision/Recall/样本数，拼一行精度对比表格。"""
    return (
        f"| {label} | {metrics[prefix + 'map05']} | {metrics[prefix + 'map95']} | "
        f"{metrics[prefix + 'precision']} | {metrics[prefix + 'recall']} | "
        f"{metrics[prefix + 'samples']} |"
    )


def _build_compare_input(
    cloud_metrics: Mapping[str, Any], edge_metrics: Mapping[str, Any]
) -> str:
    """组装 6.1 精度对比的输入数据（Markdown 表格文本）。

    云端指标取 ``det_cloud_*`` 字段（local_info），边缘端指标取
    ``det_edge_*`` 字段（metrics_summary.json），缺字段时抛 KeyError。
    """
    return "\n".join(
        [
            "| 模型           |   mAP@0.5 |   mAP@0.5:0.95 |   Precision |   Recall |   测试样本数 |",
            "|:---------------|----------:|---------------:|------------:|---------:|-------------:|",
            _accuracy_row("云端原始模型", cloud_metrics, "det_cloud_"),
            _accuracy_row("边缘端量化模型", edge_metrics, "det_edge_"),
        ]
    )


def _build_perf_input(perf_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """组装 6.4 推理性能的输入数据（batch_metrics 字段，去除 notes 说明）。"""
    return {key: value for key, value in perf_metrics.items() if key != "notes"}


@dataclass
class AnalysisSection:
    """第 6 章：评测结果对比与分析。"""

    # 6.1 小节完整内容：大模型输出（指标对比表格 + 分析结论）
    ana_61: str = ""

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

    # 6.3 边缘场景泛化表现分析
    gen_light_std: str = ""  # 不同光照-标准场景指标
    gen_light_edge: str = ""  # 不同光照-边缘场景指标
    gen_light_delta: str = ""  # 不同光照-性能变化
    gen_light_perf: str = ""  # 不同光照-泛化表现
    gen_angle_std: str = ""  # 不同角度-标准场景指标
    gen_angle_edge: str = ""  # 不同角度-边缘场景指标
    gen_angle_delta: str = ""  # 不同角度-性能变化
    gen_angle_perf: str = ""  # 不同角度-泛化表现
    ana_63: str = ""  # 6.3 分析正文

    # 6.4 小节完整内容：大模型输出（性能指标表格 + 性能瓶颈结论）
    ana_64: str = ""

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))

    def fill_llm_analysis(
        self,
        cloud_metrics: Mapping[str, Any],
        edge_metrics: Mapping[str, Any],
        perf_metrics: Mapping[str, Any],
        output_dir: str | Path | None = None,
    ) -> None:
        """调用大模型生成 6.1 / 6.4 的完整小节内容，写入 ana_61 / ana_64。

        大模型输出即小节正文（指标表格 + 分析文字），原样填入；
        ``output_dir`` 不为 None 时，把原始输出保存到该目录下的
        ``llm_analysis.json``，以提示词名称作为字段名：
        ``model_compare`` / ``performance_analysis``。

        :param cloud_metrics: 云端原始模型精度指标（local_info，det_cloud_* 字段）
        :param edge_metrics: 边缘端量化模型精度指标（metrics_summary.json，det_edge_* 字段）
        :param perf_metrics: 性能测试数据（batch_metrics.json）
        :param output_dir: 大模型原始输出保存目录，None 不保存
        """
        analyzer = LLMAnalyzer()
        self.ana_61 = analyzer.analyze_prompt(
            "model_compare",
            params={"input_data": _build_compare_input(cloud_metrics, edge_metrics)},
        )
        self.ana_64 = analyzer.analyze_prompt(
            "performance_analysis",
            params={"input_data": _build_perf_input(perf_metrics)},
        )
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_outputs = {
                "model_compare": self.ana_61,
                "performance_analysis": self.ana_64,
            }
            (out_dir / "llm_analysis.json").write_text(
                json.dumps(raw_outputs, ensure_ascii=False, indent=4) + "\n",
                encoding="utf-8",
            )