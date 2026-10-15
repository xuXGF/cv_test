"""设备信息采集公共工具。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def read_text(path: str | Path) -> str:
    """读取小文本文件；文件不存在或读取失败时返回空字符串。"""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def run_command(args: list[str], *, timeout: float = 10) -> str:
    """运行命令并返回标准输出；命令不存在或执行失败时返回空字符串。"""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def format_gb(size_bytes: int) -> str:
    """字节数格式化为 ``xx.x GB``（GiB 换算）。"""
    return f"{size_bytes / 1024**3:.1f} GB"


def storage_total() -> str:
    """根分区总容量。"""
    try:
        return format_gb(shutil.disk_usage("/").total)
    except OSError:
        return ""
