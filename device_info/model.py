"""设备信息数据模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """边缘端测试设备信息，字段与报告 1.3 节一一对应。"""

    device_name: str = ""  # 设备名称
    soc: str = ""  # 芯片/SoC
    cpu: str = ""  # CPU
    gpu: str = ""  # GPU
    npu: str = ""  # NPU
    memory: str = ""  # 内存
    storage: str = ""  # 存储
    os: str = ""  # 操作系统
