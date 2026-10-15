"""Linux（边缘端主要目标平台）设备信息采集器。"""

from __future__ import annotations

import os
import platform
import re
import socket
from pathlib import Path

from .model import DeviceInfo
from .utils import format_gb, read_text, run_command, storage_total


def collect() -> DeviceInfo:
    """采集当前 Linux 设备的硬件信息。"""
    cpuinfo = read_text("/proc/cpuinfo")
    meminfo = read_text("/proc/meminfo")
    return DeviceInfo(
        device_name=socket.gethostname(),
        soc=_soc(cpuinfo),
        cpu=_cpu(cpuinfo),
        gpu=_gpu(),
        npu=_npu(),
        memory=_memory(meminfo),
        storage=storage_total(),
        os=_os(),
    )


def _soc(cpuinfo: str) -> str:
    """优先读取设备树 model，其次 /proc/cpuinfo 的 Hardware 字段。"""
    model = read_text(
        "/sys/firmware/devicetree/base/model"
    ) or read_text("/proc/device-tree/model")
    model = model.replace("\x00", "").strip()
    if model:
        return model
    hw = re.search(r"^Hardware\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    return hw.group(1).strip() if hw else ""


def _cpu(cpuinfo: str) -> str:
    """从 /proc/cpuinfo 提取 CPU 型号，并附核心数。"""
    m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    name = m.group(1).strip() if m else platform.processor().strip()
    ncpu = os.cpu_count()
    if name and ncpu:
        return f"{name}（{ncpu} 核）"
    return name


def _gpu() -> str:
    """x86 下通过 lspci 识别显卡；ARM 板卡上多数无显卡信息则留空。"""
    out = run_command(["lspci"])
    names = re.findall(
        r"(?:VGA|3D|Display) compatible controller:\s*(.+)", out
    )
    return "; ".join(dict.fromkeys(name.strip() for name in names))


def _npu() -> str:
    """通过 NPU 设备节点、devfreq 节点与 lspci 加速器识别 NPU。"""
    found: list[str] = []

    # Rockchip NPU 设备节点（如 /dev/rknpu、/dev/rknpu_0）
    rknpu = sorted(p.name for p in Path("/dev").glob("rknpu*"))
    if rknpu:
        found.append("Rockchip RKNPU")

    # devfreq 中的 npu 节点（如 RK3588 的 fdab0000.npu）
    devfreq = Path("/sys/class/devfreq")
    if devfreq.is_dir():
        for node in devfreq.iterdir():
            if "npu" in node.name.lower():
                found.append(f"NPU devfreq 节点 {node.name}")

    # x86 平台 lspci 的加速器类别（Processing accelerators）
    out = run_command(["lspci"])
    found.extend(re.findall(r"Processing accelerators:\s*(.+)", out))

    return "; ".join(dict.fromkeys(found))


def _memory(meminfo: str) -> str:
    """从 /proc/meminfo 提取物理内存总量。"""
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    if not m:
        return ""
    return format_gb(int(m.group(1)) * 1024)


def _os() -> str:
    """发行版名称（PRETTY_NAME）+ 内核版本。"""
    os_release = read_text("/etc/os-release")
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_release, re.MULTILINE)
    if m:
        return f"{m.group(1).strip()}（内核 {platform.release()}）"
    return f"Linux（内核 {platform.release()}）"
