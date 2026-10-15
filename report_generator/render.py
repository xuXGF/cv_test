"""报告渲染引擎。

模板使用 ``{{字段名}}`` 或 ``{{字段名=默认值}}`` 形式的占位符：

* ``{{字段名}}``：字段未赋值（空字符串）时输出 ``<br />``（Markdown 空单元格占位符）。
* ``{{字段名=默认值}}``：字段未赋值时输出给定的默认值（用于模板中带有示例文案的单元格）。

因此当所有字段保持默认空值时，渲染结果即等价于原始模板；为字段赋值即可直接填入报告。
"""

from __future__ import annotations

import re
from typing import Mapping

_TOKEN_RE = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(.*?)\s*)?\}\}",
    re.DOTALL,
)

#: 空单元格在模板中的默认占位符
EMPTY_CELL = ""


def render(template: str, values: Mapping[str, str]) -> str:
    """将 ``template`` 中的占位符替换为 ``values`` 中对应字段的值。"""

    def _substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        value = values.get(name, "")
        if value:
            return str(value)
        return default if default is not None else EMPTY_CELL

    return _TOKEN_RE.sub(_substitute, template)