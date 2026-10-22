from pathlib import Path


def format_params(param_str: str | int) -> str:
    """
    将参数量转为 K / M / B 可读字符串
    :param param_str: 参数字符串/数字，如 '3193909'
    :return: 格式化结果，例 3.19M
    """
    num = int(param_str)
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num}"


def to_float(value: object, field: str = "") -> float:
    """
    将值转为 float，失败时抛出带字段名的 ValueError，不做静默兜底
    :param value: 待转换值，通常为 JSON/配置中的字符串
    :param field: 字段名，用于错误提示
    :return: 转换后的浮点数
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        label = f"字段 {field}" if field else "数值"
        raise ValueError(f"{label}不是有效数值: {value!r}") from exc


def pct(part: float, total: float, label: str = "") -> str:
    """
    计算占比：部分 / 总量 × 100%，保留 1 位小数
    :param part: 部分值
    :param total: 总量，为 0 时抛出 ValueError
    :param label: 字段名，用于错误提示
    :return: 例 5.0%
    """
    try:
        return f"{part / total * 100:.1f}%"
    except ZeroDivisionError as exc:
        raise ValueError(f"{label}为 0，无法计算占比") from exc


def index_scene_metrics(scenes: object) -> dict[str, dict]:
    """
    将场景测评结果（scene_metrics.json 的 scenes 列表）按场景目录绝对路径建索引
    :param scenes: 场景列表，每项为含 path 与实测指标的字典
    :return: 场景目录绝对路径 -> 场景条目
    """
    if not isinstance(scenes, list):
        raise TypeError(f"场景测评结果 scenes 应为列表，实际为: {type(scenes).__name__}")

    indexed: dict[str, dict] = {}
    for item in scenes:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError(f"场景实测指标缺少场景路径: {item!r}")
        indexed[str(Path(item["path"]).resolve())] = item
    return indexed


def format_memory(mb: float) -> str:
    """
    将内存占用（MB）转为可读字符串：小于 1024 MB 用 MB，否则用 GB
    :param mb: 内存大小（兆字节）
    :return: 例 312.4 MB / 1.25 GB
    """
    if mb < 1024:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"