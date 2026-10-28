"""测评报告聚合器：组合各章节并渲染为完整报告。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .render import render
from .sections import (
    AccuracySection,
    AnalysisSection,
    AttachmentSection,
    BasicInfoSection,
    ConfigSection,
    DatasetSection,
    PreviewSection,
    SpeedSection,
    SuggestionSection,
    VerdictSection,
)

#: 封面页：居中标题 + 底部日期，末尾分页符确保第 1 章另起一页
_DOC_HEADER = r"""<div align="center">

<br /><br /><br /><br /><br /><br /><br /><br /><br /><br />

<h1>{{report_title=测评报告}}</h1>

<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />

<p>{{cover_date=2026.10}}</p>

</div>

<div style="page-break-after: always;"></div>"""

#: 章节顺序
_SECTION_FIELDS = (
    "basic_info",
    "dataset",
    "config",
    "accuracy",
    "speed",
    "analysis",
    "verdict",
    "suggestion",
    # "attachment",
)


@dataclass
class TestReport:
    """一份完整测评报告，按章节聚合 9 个章节模块。

    用法::

        report = TestReport()
        report.basic_info.report_name = "某模型边缘端测评报告"
        report.render()          # 渲染为 Markdown 文本
        report.save("out.md")    # 保存到文件

    所有章节字段默认空值，此时 ``render()`` 的输出与 ``template.md`` 内容一致。
    """

    report_title: str = ""
    cover_date: str = ""
    preview: PreviewSection = field(default_factory=PreviewSection)
    basic_info: BasicInfoSection = field(default_factory=BasicInfoSection)
    dataset: DatasetSection = field(default_factory=DatasetSection)
    config: ConfigSection = field(default_factory=ConfigSection)
    accuracy: AccuracySection = field(default_factory=AccuracySection)
    speed: SpeedSection = field(default_factory=SpeedSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    verdict: VerdictSection = field(default_factory=VerdictSection)
    suggestion: SuggestionSection = field(default_factory=SuggestionSection)
    attachment: AttachmentSection = field(default_factory=AttachmentSection)

    def _render_header(self) -> str:
        return render(
            _DOC_HEADER,
            {"report_title": self.report_title, "cover_date": self.cover_date},
        )

    def render(self) -> str:
        """将整份报告渲染为 Markdown 文本。"""
        # 封面页 -> 预览表页 -> 正文（预览页自带分页符）
        blocks = [self._render_header(), self.preview.render()]
        blocks.extend(getattr(self, name).render() for name in _SECTION_FIELDS)
        return "\n\n".join(blocks)

    def render_before_verdict(self) -> str:
        """渲染封面、预览表与 1~7.3 节正文，作为 7.4 综合判定的大模型输入。

        不含 7.4 及后续章节，避免把待生成的判定内容与未填充模板传给大模型。
        """
        blocks = [self._render_header(), self.preview.render()]
        for name in _SECTION_FIELDS:
            if name == "verdict":
                break
            blocks.append(getattr(self, name).render())
        blocks.append(self.verdict.render_before_verdict())
        return "\n\n".join(blocks)

    def save(self, path: str | Path, *, encoding: str = "utf-8") -> Path:
        """将报告写入文件并返回写入路径。"""
        target = Path(path)
        target.write_text(self.render(), encoding=encoding)
        return target