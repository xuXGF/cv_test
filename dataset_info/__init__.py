"""数据集信息采集、COCO 标注转换与预处理模块。

输出的信息用于填写测评报告第 2 章「测试数据集说明」：

* ``collect_dataset_info()``：扫描数据集目录，采集 2.1 节数据集基本
  信息并填写 2.3 节全表；提供或检测到 COCO 标注时自动转换为
  YOLO txt，传入 ``records`` 可填写实际执行的预处理情况。
* ``convert_coco_to_yolo()``：COCO JSON 标注 -> YOLO txt 标注转换。
* ``preprocess`` 子模块：2.3 节各处理项目的实际处理程序——
  ``convert_image_format``（数据格式转换）、``resize_images``
  （图像尺寸处理）、``clean_dataset``（数据清洗）、
  ``handle_anomalies``（异常数据处理），按需手动调用。

用法::

    from dataset_info import collect_dataset_info, resize_images

    record = resize_images("data/coco128", (640, 640), letterbox=True)  # 需要时执行
    info = collect_dataset_info("data/coco128", records=[record])       # 报告记录实际处理
    print(info.data_total, info.pre_method_size)

    report.dataset.__dict__.update(info.__dict__)        # 自动采集字段
    report.dataset.__dict__.update(local_info.__dict__)  # 本地手填字段（含边缘场景）优先

说明：

* 边缘场景数据（2.2 节）由本地文件（``local_info.json``）手动填写，
  本模块不采集、不输出场景字段。
* 采集不到的字段保持空字符串，不自动填入虚假值；
  删除/修复类预处理默认只检测（dry_run），需显式开启才修改文件。
"""

from __future__ import annotations

from .coco2yolo import Coco2YoloResult, PreprocessRecord, convert_coco_to_yolo
from .info import DatasetInfo, collect_dataset_info, default_dataset_path
from .preprocess import (
    AnomalyResult,
    CleanResult,
    ImageFormatResult,
    ResizeResult,
    clean_dataset,
    convert_image_format,
    handle_anomalies,
    resize_images,
)

__all__ = [
    "Coco2YoloResult",
    "PreprocessRecord",
    "convert_coco_to_yolo",
    "DatasetInfo",
    "collect_dataset_info",
    "default_dataset_path",
    "ImageFormatResult",
    "ResizeResult",
    "CleanResult",
    "AnomalyResult",
    "convert_image_format",
    "resize_images",
    "clean_dataset",
    "handle_anomalies",
]
