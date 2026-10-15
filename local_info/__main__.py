"""命令行入口：``python -m local_info`` 打印本地信息文件当前内容。"""

from __future__ import annotations

from . import load_local_info


def main() -> None:
    info = load_local_info()
    data = info.to_dict()
    if not data:
        print("local_info.json 当前为空，请手动填写后重试。")
        return
    for key, value in data.items():
        print(f"{key}: {value!r}")


if __name__ == "__main__":
    main()
