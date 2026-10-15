"""命令行入口：``python -m device_info`` 打印采集到的设备信息。"""

from __future__ import annotations

from dataclasses import asdict

from . import collect_device_info


def main() -> None:
    for key, value in asdict(collect_device_info()).items():
        print(f"{key}: {value!r}")


if __name__ == "__main__":
    main()
