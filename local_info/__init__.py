"""本地信息模块：手动维护的 JSON 文件 <-> 本地信息对象。

``local_info.json`` 默认为空对象，内容（字段与取值）由使用者
手动填写；读取时将 JSON 顶层的键值转为对象属性，直接使用。

用法::

    from local_info import load_local_info

    info = load_local_info()
    info.soc   # 属性名即 JSON 中手写的键

命令行查看当前内容::

    python -m local_info

说明：

* JSON 顶层必须是对象（``{...}``），每个键成为对象的一个属性。
* 键需为合法 Python 标识符（不能是关键字），否则读取时报错。
* 文件不存在、JSON 非法、类型不符时抛出带明确信息的异常，
  不做静默兜底。
"""

from __future__ import annotations

import json
import keyword
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).parent / "local_info.json"


class LocalInfo:
    """本地信息对象，属性即 ``local_info.json`` 中手动填写的键。

    尚未填写的字段返回空字符串（与 ``DeviceInfo`` 约定一致），
    因此可以按报告所需的全部字段直接访问，填过的用实际值。
    """

    def __init__(self, **data: Any) -> None:
        invalid = [k for k in data if not k.isidentifier() or keyword.iskeyword(k)]
        if invalid:
            raise ValueError(
                f"以下键不能作为对象属性使用: {invalid}，"
                "请改用合法的 Python 标识符（字母/数字/下划线，且不以数字开头）"
            )
        self.__dict__.update(data)

    def __getattr__(self, name: str) -> Any:
        # 仅在常规属性查找失败时触发：已填写的键返回 JSON 中的值，
        # 未填写的键返回空字符串；双下划线属性按正常规则报错。
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return ""

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"LocalInfo({items})"

    def to_dict(self) -> dict[str, Any]:
        """转为字典（属性名 -> 值），与 JSON 文件结构一致。"""
        return dict(self.__dict__)


def load_local_info(path: str | Path = DEFAULT_PATH) -> LocalInfo:
    """读取 JSON 文件为 ``LocalInfo`` 对象。

    默认读取包内 ``local_info.json``；文件不存在、JSON 非法、
    顶层不是对象时抛出带明确信息的异常。
    """
    json_path = Path(path)
    try:
        text = json_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"本地信息文件不存在: {json_path}") from exc
    except OSError as exc:
        raise OSError(f"本地信息文件读取失败: {json_path}（{exc}）") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"本地信息文件不是合法 JSON: {json_path}（第 {exc.lineno} 行：{exc.msg}）"
        ) from exc

    if not isinstance(raw, dict):
        raise TypeError(
            f"本地信息文件顶层应为 JSON 对象，实际为 {type(raw).__name__}: {json_path}"
        )

    return LocalInfo(**raw)


def save_local_info(info: LocalInfo, path: str | Path = DEFAULT_PATH) -> Path:
    """将 ``LocalInfo`` 对象写回 JSON 文件，返回文件路径。"""
    json_path = Path(path)
    json_path.write_text(
        json.dumps(info.to_dict(), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return json_path
