"""第 8 章：量化调优建议。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_analysis import LLMAnalyzer

from ..render import render

_TEMPLATE = r"""# 8. 量化调优建议

## 8.1 精度优化建议

{{ana_81=\[填写精度优化建议表格。]}}

## 8.2 推理性能优化建议

{{ana_82=\[填写推理性能优化建议表格。]}}"""


@dataclass
class SuggestionSection:
    """第 8 章：量化调优建议。"""

    # 8.1 精度优化建议（由 fill_accuracy_suggestions 调用大模型生成完整表格）
    ana_81: str = ""  # 精度优化建议表格（大模型输出）

    # 8.2 推理性能优化建议（由 fill_performance_suggestions 调用大模型生成完整表格）
    ana_82: str = ""  # 推理性能优化建议表格（大模型输出）

    def fill_accuracy_suggestions(
        self, report_text: str, output_dir: str | Path | None = None
    ) -> None:
        """8.1 精度优化建议：调用大模型基于整份报告正文生成建议表格。

        输入为报告完整正文（``TestReport.render()``，直接转为字符串，
        不单独提取章节）；由提示词 ``accuracy_optimization`` 按其固定
        表格格式输出，大模型输出原样填入 ``ana_81``。
        ``output_dir`` 不为 None 时，把原始输出并入该目录下的
        ``llm_analysis.json``（字段名 ``accuracy_optimization``），
        与第 6、7 章大模型输出保存在同一文件，已有字段保留。

        :param report_text: 报告完整正文
        :param output_dir: 大模型原始输出保存目录，None 不保存
        """
        analyzer = LLMAnalyzer()
        self.ana_81 = analyzer.analyze_prompt(
            "accuracy_optimization",
            params={"input_data": report_text},
        )
        _save_llm_output(output_dir, "accuracy_optimization", self.ana_81)

    def fill_performance_suggestions(
        self, report_text: str, output_dir: str | Path | None = None
    ) -> None:
        """8.2 推理性能优化建议：调用大模型基于整份报告正文生成建议表格。

        输入为报告完整正文（``TestReport.render()``，直接转为字符串，
        不单独提取章节）；由提示词 ``performance_optimization`` 按其固定
        表格格式输出，大模型输出原样填入 ``ana_82``。
        ``output_dir`` 不为 None 时，把原始输出并入该目录下的
        ``llm_analysis.json``（字段名 ``performance_optimization``），
        与第 6、7 章大模型输出保存在同一文件，已有字段保留。

        :param report_text: 报告完整正文
        :param output_dir: 大模型原始输出保存目录，None 不保存
        """
        analyzer = LLMAnalyzer()
        self.ana_82 = analyzer.analyze_prompt(
            "performance_optimization",
            params={"input_data": report_text},
        )
        _save_llm_output(output_dir, "performance_optimization", self.ana_82)

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))


def _save_llm_output(
    output_dir: str | Path | None, prompt_name: str, content: str
) -> None:
    """把大模型原始输出并入 ``output_dir/llm_analysis.json``。

    ``output_dir`` 为 None 时不保存；文件已存在时保留其他提示词的输出。
    """
    if output_dir is None:
        return
    out_path = Path(output_dir) / "llm_analysis.json"
    outputs = {}
    if out_path.exists():
        outputs = json.loads(out_path.read_text(encoding="utf-8"))
    outputs[prompt_name] = content
    out_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
