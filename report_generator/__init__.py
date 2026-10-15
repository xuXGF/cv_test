"""边缘端模型测评报告生成器。"""

from __future__ import annotations

from .render import EMPTY_CELL, render
from .report import TestReport
from .sections import (
    AccuracySection,
    AnalysisSection,
    AttachmentSection,
    BasicInfoSection,
    ConfigSection,
    DatasetSection,
    SpeedSection,
    SuggestionSection,
    VerdictSection,
)

__all__ = [
    "TestReport",
    "render",
    "EMPTY_CELL",
    "BasicInfoSection",
    "DatasetSection",
    "ConfigSection",
    "AccuracySection",
    "SpeedSection",
    "AnalysisSection",
    "VerdictSection",
    "SuggestionSection",
    "AttachmentSection",
]