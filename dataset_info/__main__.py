"""命令行入口：数据集扫描与预处理。

用法::

    python -m dataset_info                                       # 扫描包内自带 data/coco128
    python -m dataset_info scan [数据集目录] [--coco-json 路径]    # 扫描并打印报告字段
    python -m dataset_info convert [数据集目录] [--coco-json 路径] # 标签格式转换：COCO -> YOLO txt
    python -m dataset_info convert-format [目录] [--to .jpg] [--replace]  # 数据格式转换
    python -m dataset_info resize [目录] --size 640x640 [--letterbox]     # 图像尺寸处理
    python -m dataset_info clean [目录] [--apply]                # 数据清洗（缺省仅检测）
    python -m dataset_info anomaly [目录] [--apply]              # 异常数据处理（缺省仅检测）

``scan`` 之外均为实际预处理命令，打印报告 2.3 节对应行的填写内容；
预处理完成后可再次 ``scan`` 查看最新数据状态。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from . import (
    clean_dataset,
    collect_dataset_info,
    convert_image_format,
    default_dataset_path,
    handle_anomalies,
    resize_images,
)
from .coco2yolo import conversion_layout, convert_coco_to_yolo, detect_coco_jsons

#: 子命令集合（用于把旧式参数自动路由到 scan）
_COMMANDS = {"scan", "convert", "convert-format", "resize", "clean", "anomaly"}


def _parse_size(text: str) -> tuple[int, int]:
    """解析 ``宽x高`` 形式的尺寸参数。"""
    try:
        width, height = text.lower().split("x")
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"尺寸格式应为 宽x高，如 640x640: {text}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_info", description="数据集信息采集与预处理"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="扫描数据集并打印报告字段")
    p.add_argument("dataset_dir", nargs="?", default=None)
    p.add_argument("--coco-json", default=None, help="COCO 标注 JSON（提供时先转换）")

    p = sub.add_parser("convert", help="标签格式转换：COCO JSON -> YOLO txt")
    p.add_argument("dataset_dir", nargs="?", default=None)
    p.add_argument("--coco-json", default=None, help="缺省自动探测 annotations/*.json")

    p = sub.add_parser("convert-format", help="数据格式转换：统一图片格式")
    p.add_argument("dataset_dir", nargs="?", default=None)
    p.add_argument("--to", dest="target_ext", default=".jpg", help="目标格式（如 .jpg）")
    p.add_argument("--replace", action="store_true", help="转换成功后删除原文件")

    p = sub.add_parser("resize", help="图像尺寸处理：统一分辨率")
    p.add_argument("dataset_dir", nargs="?", default=None)
    p.add_argument("--size", type=_parse_size, required=True, help="目标尺寸，如 640x640")
    p.add_argument(
        "--letterbox", action="store_true",
        help="保持宽高比填充（自动同步调整标签坐标）",
    )

    p = sub.add_parser("clean", help="数据清洗：剔除损坏/不可读图片")
    p.add_argument("dataset_dir", nargs="?", default=None)
    p.add_argument("--apply", action="store_true", help="实际删除（缺省仅检测）")

    p = sub.add_parser("anomaly", help="异常数据处理：配对检查与标注修复")
    p.add_argument("dataset_dir", nargs="?", default=None)
    p.add_argument("--apply", action="store_true", help="实际修复/删除（缺省仅检测）")
    return parser


def _print_result(result) -> None:
    """打印一次处理的结果记录（即报告 2.3 节对应行）。"""
    for key, value in asdict(result).items():
        print(f"{key}: {value}")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in _COMMANDS:
        # 兼容旧用法：python -m dataset_info [数据集目录] [COCO标注JSON]
        argv = ["scan", *argv]
    args = _build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_dataset_path()

    if args.command == "scan":
        coco_json = Path(args.coco_json) if args.coco_json else None
        info = collect_dataset_info(dataset_dir, coco_json=coco_json)
        for key, value in asdict(info).items():
            print(f"{key}: {value}")
    elif args.command == "convert":
        json_paths = (
            [Path(args.coco_json)] if args.coco_json else detect_coco_jsons(dataset_dir)
        )
        if not json_paths:
            raise SystemExit(
                f"未找到 COCO 标注文件（annotations/*.json）: {dataset_dir}，"
                "可用 --coco-json 显式指定"
            )
        for json_path in json_paths:
            image_dir, labels_dir = conversion_layout(dataset_dir, json_path)
            _print_result(convert_coco_to_yolo(json_path, labels_dir, image_dir=image_dir))
    elif args.command == "convert-format":
        _print_result(
            convert_image_format(dataset_dir, args.target_ext, replace=args.replace)
        )
    elif args.command == "resize":
        _print_result(resize_images(dataset_dir, args.size, letterbox=args.letterbox))
    elif args.command == "clean":
        _print_result(clean_dataset(dataset_dir, dry_run=not args.apply))
    elif args.command == "anomaly":
        _print_result(handle_anomalies(dataset_dir, dry_run=not args.apply))


if __name__ == "__main__":
    main()
