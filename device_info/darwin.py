"""macOS（本地开发机，验证用）设备信息采集器。"""

from __future__ import annotations

import platform
import re
import socket

from .model import DeviceInfo
from .utils import format_gb, run_command, storage_total


def collect() -> DeviceInfo:
    """采集当前 macOS 设备的硬件信息。"""
    brand = run_command(
        ["sysctl", "-n", "machdep.cpu.brand_string"]
    )  # 例如 "Apple M3 Pro"
    is_apple = brand.startswith("Apple")
    ncpu = run_command(["sysctl", "-n", "hw.ncpu"])
    memsize = run_command(["sysctl", "-n", "hw.memsize"])
    ver = platform.mac_ver()[0]

    cpu = brand
    if brand and ncpu:
        cpu = f"{brand}（{ncpu} 核）"

    return DeviceInfo(
        device_name=socket.gethostname(),
        soc=brand if is_apple else "",
        cpu=cpu,
        gpu=_gpu(),
        npu="Apple 神经网络引擎（Neural Engine）" if is_apple else "",
        memory=format_gb(int(memsize)) if memsize.isdigit() else "",
        storage=storage_total(),
        os=f"macOS {ver}（Darwin {platform.release()}）" if ver else platform.platform(),
    )


def _gpu() -> str:
    """从 system_profiler 提取显卡型号。"""
    out = run_command(
        ["system_profiler", "SPDisplaysDataType"],
        timeout=30,
    )
    names = re.findall(r"Chipset Model:\s*(.+)", out)
    return "; ".join(dict.fromkeys(name.strip() for name in names))
