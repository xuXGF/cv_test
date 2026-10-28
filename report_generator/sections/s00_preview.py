"""封面后预览页：测评报告预览表。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from ..render import render

#: 预览表「测试内容」为固定文案，不随测评数据变化
_TEST_CONTENT = "模型精度、推理性能、边缘场景适应性"

#: 预览页（位于封面之后、正文之前，末尾分页符确保正文另起一页）
_TEMPLATE = r"""<p align="center"><strong>测评报告预览表</strong></p>

<!-- table-no-header -->
<!-- table-fill-page -->
| **测试基本信息** | 报告编号：{{report_no}}<br>测评对象：{{target_object}}<br>模型版本：{{model_version}}<br>测试设备：{{soc}} |
| --- | --- |
| **测试内容** | {{test_content}} |
| **测试结论** | {{test_conclusion}} |
| **测试人员** | 测试人员：<br>测试人员联系方式：<br>测试人员单位： |
| **测试确认人员** | 测试确认人员：<br>测试确认人员联系方式：<br>测试确认人员单位： |

<p align="right">日期：{{preview_date}}</p>

<div style="page-break-after: always;"></div>"""


@dataclass
class PreviewSection:
    """测评报告预览表。

    测试基本信息与 1.1/1.2/1.3 原文对应，测试内容为固定文案，
    测试结论取 7.4「综合验收结论」正文；测试人员、测试确认人员
    仅列出待填字段，留待后续人工签字。
    """

    # 测试基本信息（字段与 ModelInfo / local_info / DeviceInfo 同名，可自动填充）
    report_no: str = ""  # 报告编号
    model_arch: str = ""  # 模型架构
    algo_type: str = ""  # 算法类型
    model_version: str = ""  # 模型版本
    soc: str = ""  # 测试设备（芯片/SoC）
    test_date: str = ""  # 测评日期（用于生成预览表日期）

    # 预览表正文
    target_object: str = ""  # 测评对象（模型架构 + 算法类型 + 模型）
    test_content: str = _TEST_CONTENT  # 测试内容（固定）
    test_conclusion: str = ""  # 测试结论（7.4 综合验收结论）
    preview_date: str = ""  # 日期（右下角）

    def fill_basic_info(self) -> None:
        """填充测试基本信息与日期。

        测评对象 = 模型架构 + 算法类型 + 模型（与 1.2 原文对应）；
        日期由测评日期（``2026/10/23 14:27``）生成 ``2026年10月23日``。
        """
        if not self.model_arch:
            raise ValueError("预览表缺少模型架构（model_arch），无法生成测评对象")
        self.target_object = f"{self.model_arch} {self.algo_type}模型"
        self.preview_date = _format_date(self.test_date)

    def fill_conclusion(self, verdict_text: str) -> None:
        """从 7.4 判定正文中提取「综合验收结论」内容，作为预览表测试结论。

        找不到该标题或内容为空时抛出异常，不做静默兜底。
        """
        marker = "综合验收结论"
        position = verdict_text.find(marker)
        if position < 0:
            raise ValueError(
                "7.4 评测结果综合判定中未找到“综合验收结论”，无法填写预览表测试结论"
            )
        content = verdict_text[position + len(marker):].lstrip("*：: \t\r\n")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("7.4 综合验收结论内容为空，无法填写预览表测试结论")
        self.test_conclusion = "<br>".join(lines)

    def render(self) -> str:
        """渲染预览页为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))


def _format_date(test_date: str) -> str:
    """将测评日期 ``2026/10/23 14:27`` 转为 ``2026年10月23日``。"""
    try:
        day = datetime.strptime(test_date[:10], "%Y/%m/%d")
    except ValueError as exc:
        raise ValueError(f"预览表日期无法解析测评日期：{test_date!r}") from exc
    return f"{day.year}年{day.month}月{day.day}日"
