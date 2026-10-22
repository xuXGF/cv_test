"""边缘端测试设备信息采集模块。

在测试设备上运行本模块即可采集硬件信息，结果用于填写测评报告
第 1.3 节「测试设备信息」，字段名与
``report_generator.sections.BasicInfoSection`` 中的设备字段一一对应。

用法::

    from device_info import collect_device_info

    info = collect_device_info()
    print(info.soc, info.cpu, info.memory, ...)

说明：

* 支持 Linux（RK3588/Jetson 等边缘设备）与 macOS（本地开发机）。
* 采集不到的字段保持空字符串（例如无 NPU 的设备 ``npu`` 为空），
  不自动填入虚假值。
"""

from __future__ import annotations

import json
import platform
import socket
import sys
from dataclasses import asdict
from pathlib import Path

from .darwin import collect as _collect_darwin
from .linux import collect as _collect_linux
from .model import DeviceInfo

__all__ = ["DeviceInfo", "collect_device_info", "save_device_info"]


def collect_device_info() -> DeviceInfo:
    """采集当前运行设备的硬件信息。"""
    if sys.platform == "darwin":
        return _collect_darwin()
    if sys.platform.startswith("linux"):
        return _collect_linux()
    # 其他平台仅提供基础信息
    return DeviceInfo(
        device_name=socket.gethostname(),
        os=platform.platform(),
    )


def save_device_info(info: DeviceInfo, path: str | Path) -> Path:
    """将 ``DeviceInfo`` 写为 JSON 文件，返回文件路径。"""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return json_path
