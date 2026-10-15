"""报告各章节模块（按章节拆分为独立的类）。"""

from __future__ import annotations

from .s01_basic_info import BasicInfoSection
from .s02_dataset import DatasetSection
from .s03_config import ConfigSection
from .s04_accuracy import AccuracySection
from .s05_speed import SpeedSection
from .s06_analysis import AnalysisSection
from .s07_verdict import VerdictSection
from .s08_suggestions import SuggestionSection
from .s09_attachments import AttachmentSection

__all__ = [
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