"""数据集信息采集。

``collect_dataset_info()`` 扫描数据集目录，输出字段与测评报告
第 2 章「测试数据集说明」（``DatasetSection``）中可自动采集的字段一一对应：

* 2.1 数据集基本信息：数据总量、测试数据量、类别数量、标注格式、
  数据分辨率、数据格式等。
* 2.3 数据预处理及标准化：全表填写——无处理需求的行填「无需处理」；
  实际执行过预处理时（自动转换或传入 ``preprocess`` 模块的记录），
  对应行填写实际执行情况。

说明：

* 边缘场景数据（2.2）由使用者通过本地文件（``local_info.json``）
  手动填写，本模块不采集、不输出场景字段。
* 数据来源等无法自动判断的字段保持空字符串，不自动填入虚假值。
* ``num_classes`` 为标注中实际出现的类别 id 数；若数据集定义了更多
  类别（存在空类别），请通过 ``local_info.json`` 手动覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .coco2yolo import (
    Coco2YoloResult,
    PreprocessRecord,
    conversion_layout,
    convert_coco_to_yolo,
    detect_coco_jsons,
    image_root_of,
    labels_root_for,
    list_images,
)

#: 读取分辨率时的采样图片数上限
_SAMPLE_LIMIT = 64

#: 视为「测试数据」的划分目录名前缀（YOLO 布局中 images/ 的一级子目录）
_TEST_SPLIT_PREFIXES = ("val", "valid", "test")

#: 预处理记录项 -> DatasetInfo 中对应的三个字段（orig / method / result）
_RECORD_ROWS = {
    "数据格式转换": ("pre_orig_format", "pre_method_format", "pre_result_format"),
    "标签格式转换": ("pre_orig_label", "pre_method_label", "pre_result_label"),
    "图像尺寸处理": ("pre_orig_size", "pre_method_size", "pre_result_size"),
    "数据清洗": ("pre_orig_clean", "pre_method_clean", "pre_result_clean"),
    "异常数据处理": ("pre_orig_anomaly", "pre_method_anomaly", "pre_result_anomaly"),
}


def default_dataset_path() -> Path:
    """包内自带数据集的路径。"""
    return Path(__file__).resolve().parent / "data" / "coco128"


@dataclass
class DatasetInfo:
    """数据集元信息，字段与报告第 2 章「测试数据集说明」对应。

    不含 2.2 节边缘场景字段——该部分由 ``local_info.json`` 手动填写。
    """

    # 2.1 数据集基本信息
    dataset_name: str = ""       # 数据集名称（数据集根目录名）
    data_source: str = ""        # 数据来源（无法自动判断，保持为空，由 local_info 填写）
    data_type: str = ""          # 数据类型（扫描到图片时为「图片」）
    data_total: str = ""         # 数据总量（图片张数）
    test_data_count: str = ""    # 测试数据量（val/test 划分张数，无划分时与总量一致）
    num_classes: str = ""        # 类别数量（标注中实际出现的类别 id 数）
    annotation_format: str = ""  # 标注格式（YOLO txt / COCO JSON）
    data_resolution: str = ""    # 数据分辨率（采样统计，宽x高，多种时列出前几种）
    data_format: str = ""        # 数据格式（图片扩展名集合）

    # 2.3 数据预处理及标准化
    pre_orig_format: str = ""    # 数据格式转换-原始数据（现有图片格式）
    pre_method_format: str = ""  # 数据格式转换-标准化处理方式（无处理需求时为「无需转换」）
    pre_result_format: str = ""  # 数据格式转换-处理结果（无处理需求时为「无需处理」）
    pre_orig_label: str = ""     # 标签格式转换-原始数据（现有标注格式）
    pre_method_label: str = ""   # 标签格式转换-标准化处理方式
    pre_result_label: str = ""   # 标签格式转换-处理结果
    pre_orig_size: str = ""      # 图像尺寸处理-原始数据（现有分辨率）
    pre_method_size: str = ""    # 图像尺寸处理-标准化处理方式（无处理需求时为「无需处理」）
    pre_result_size: str = ""    # 图像尺寸处理-处理结果
    pre_orig_clean: str = ""     # 数据清洗-原始数据
    pre_method_clean: str = ""   # 数据清洗-标准化处理方式
    pre_result_clean: str = ""   # 数据清洗-处理结果
    pre_orig_anomaly: str = ""   # 异常数据处理-原始数据
    pre_method_anomaly: str = ""  # 异常数据处理-标准化处理方式
    pre_result_anomaly: str = ""  # 异常数据处理-处理结果


# ---------------------------------------------------------------------------
# 数据集扫描
# ---------------------------------------------------------------------------

def _collect_class_ids(label_files: list[Path]) -> set[int]:
    """统计标签文件中出现的类别 id（忽略无法解析的行）。"""
    class_ids: set[int] = set()
    for txt in label_files:
        for line in txt.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            try:
                class_ids.add(int(tokens[0]))
            except ValueError:
                continue
    return class_ids


def _is_test_split(rel: Path) -> bool:
    """图片相对路径的一级目录是否为测试划分（val/test）。"""
    first = rel.parts[0] if len(rel.parts) > 1 else ""
    return first.startswith(_TEST_SPLIT_PREFIXES)


def _image_size(path: Path) -> tuple[int, int] | None:
    """读取单张图片的 (宽, 高)，读取失败返回 None。"""
    try:
        import cv2  # 延迟导入：标注转换不依赖 OpenCV
    except ImportError as exc:
        raise ImportError(
            "读取图片分辨率需要 OpenCV，请安装: pip install opencv-python"
        ) from exc
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def _sample(images: list[Path], limit: int = _SAMPLE_LIMIT) -> list[Path]:
    """均匀采样至多 ``limit`` 张图片用于分辨率统计。"""
    if len(images) <= limit:
        return images
    step = len(images) / limit
    return [images[min(len(images) - 1, int(i * step))] for i in range(limit)]


def _resolution_summary(sizes: list[tuple[int, int]]) -> str:
    """把分辨率列表归纳为摘要文本。"""
    distinct = sorted(set(sizes))
    if not distinct:
        return ""
    if len(distinct) == 1:
        width, height = distinct[0]
        return f"{width}x{height}"
    shown = "、".join(f"{width}x{height}" for width, height in distinct[:3])
    return f"多种（{shown} 等）" if len(distinct) > 3 else f"多种（{shown}）"


def _scan_resolutions(images: list[Path]) -> str:
    """采样读取图片分辨率，全部失败时抛出异常，不静默返回空值。"""
    sizes: list[tuple[int, int]] = []
    for path in _sample(images):
        size = _image_size(path)
        if size is not None:
            sizes.append(size)
    if not sizes:
        raise ValueError(f"无法读取任何采样图片的分辨率，请检查图片文件是否损坏: {images[0].parent}")
    return _resolution_summary(sizes)


# ---------------------------------------------------------------------------
# 采集入口
# ---------------------------------------------------------------------------

def collect_dataset_info(
    dataset_dir: str | Path | None = None,
    *,
    coco_json: str | Path | None = None,
    records: Sequence[PreprocessRecord] = (),
) -> DatasetInfo:
    """扫描数据集目录，采集报告第 2 章可自动获取的字段。

    :param dataset_dir: 数据集根目录，缺省使用包内自带 ``data/coco128``。
    :param coco_json: COCO 标注 JSON 路径。提供时先转换为 YOLO txt
        （输出到对应 ``labels`` 目录）；未提供但数据集存在标准布局
        ``annotations/*.json`` 时自动执行转换。
    :param records: 预处理记录（``preprocess`` 模块各处理函数与
        ``convert_coco_to_yolo`` 的返回值），用于填写 2.3 节对应行；
        缺省时无处理需求的行填写「无需处理」。
    :return: ``DatasetInfo``，无法自动采集的字段为空字符串。
    """
    root = Path(dataset_dir) if dataset_dir is not None else default_dataset_path()
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    # 1) 按需执行 COCO -> YOLO 转换（显式指定或标准布局自动探测）
    json_paths = [Path(coco_json)] if coco_json is not None else detect_coco_jsons(root)
    record_list = list(records)
    for json_path in json_paths:
        image_dir, labels_dir = conversion_layout(root, json_path)
        record_list.append(convert_coco_to_yolo(json_path, labels_dir, image_dir=image_dir))
    # 转换时跳过的无效标注即为异常数据处理结果
    skipped_total = sum(
        r.skipped_annotations for r in record_list if isinstance(r, Coco2YoloResult)
    )
    if skipped_total:
        record_list.append(
            PreprocessRecord(
                item="异常数据处理",
                orig="COCO 标注中的无效标注",
                method=f"coco2yolo 转换时自动跳过（{skipped_total} 处）",
                result="已剔除，不写入 YOLO txt",
            )
        )

    # 2) 扫描图片与标签
    image_root = image_root_of(root)
    images = list_images(image_root)
    labels_root = labels_root_for(root, image_root)
    label_files: list[Path] = []
    for img in images:
        txt = labels_root / img.relative_to(image_root).with_suffix(".txt")
        if txt.is_file():
            label_files.append(txt)

    # 3) 测试数据量：存在 val/test 划分时取该划分，否则为全部图片
    test_images = [
        img for img in images if _is_test_split(img.relative_to(image_root))
    ]
    test_count = len(test_images) if test_images else len(images)

    # 4) 组装字段
    class_ids = _collect_class_ids(label_files)
    if label_files:
        annotation_format = "YOLO txt（class_id cx cy w h，归一化）"
    elif any(isinstance(r, Coco2YoloResult) for r in record_list):
        annotation_format = "COCO JSON"
    else:
        annotation_format = ""

    info = DatasetInfo(
        dataset_name=root.name,
        data_type="图片" if images else "",
        data_total=str(len(images)),
        test_data_count=str(test_count),
        num_classes=str(len(class_ids)),
        annotation_format=annotation_format,
        data_format="、".join(sorted({img.suffix[1:].upper() for img in images})),
    )
    info.data_resolution = _scan_resolutions(images)

    # 5) 2.3 数据预处理及标准化：各行默认「无需处理」，再由预处理记录覆盖对应行
    info.pre_orig_format = info.data_format
    info.pre_method_format = "无需转换（保持原始图片格式）"
    info.pre_result_format = "无需处理"

    info.pre_orig_size = info.data_resolution
    info.pre_method_size = "无需处理（保持原始分辨率）"
    info.pre_result_size = "无需处理"

    info.pre_orig_clean = "全量数据"
    info.pre_method_clean = "无需清洗"
    info.pre_result_clean = "无需处理"

    info.pre_orig_anomaly = "无"
    info.pre_method_anomaly = "无需处理"
    info.pre_result_anomaly = "无异常数据"

    if label_files:
        info.pre_orig_label = "YOLO txt（class_id cx cy w h，归一化）"
        info.pre_method_label = "无需转换（已为标准化格式）"
        info.pre_result_label = "无需处理"

    for record in record_list:
        fields = _RECORD_ROWS.get(record.item)
        if fields is None:
            raise ValueError(f"未知的预处理记录项: {record.item}")
        for name, value in zip(fields, (record.orig, record.method, record.result)):
            setattr(info, name, value)

    return info
