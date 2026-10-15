"""数据集预处理与标准化程序（报告 2.3 节各处理项目的实际实现）。

每个函数对应 2.3 节的一行处理项目，需要时手动调用；处理完成后可
将返回的记录传入 ``collect_dataset_info(records=...)``，由采集器把
实际执行情况填入报告对应行。

* ``convert_image_format``：数据格式转换（统一图片格式，如 BMP -> JPG）。
* ``convert_coco_to_yolo``（见 ``coco2yolo`` 模块）：标签格式转换。
* ``resize_images``：图像尺寸处理（统一分辨率，支持 letterbox）。
* ``clean_dataset``：数据清洗（剔除损坏 / 不可读图片）。
* ``handle_anomalies``：异常数据处理（图片-标签配对检查、非法标注修复）。

说明：

* YOLO txt 标签为归一化坐标：直接缩放（resize）不改变标签内容；
  letterbox 含填充偏移，会同步调整标签坐标。
* 删除/修复类操作默认只检测（dry_run），需显式关闭 dry_run 才实际
  修改文件，不做静默修改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .coco2yolo import (
    PreprocessRecord,
    _label_path,
    image_root_of,
    labels_root_for,
    list_images,
)

#: cv2 可写的图片格式
_WRITABLE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

#: letterbox 填充灰度（与 Ultralytics 一致）
_PAD_VALUE = 114

#: YOLO txt 每行保留的小数位数
_DECIMALS = 6


def _normalize_ext(ext: str) -> str:
    """扩展名规范化为 ``.jpg`` 形式（小写、带点）。"""
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def _read_image(path: Path):
    """读取图片，损坏 / 不可读返回 None。"""
    return cv2.imread(str(path))


def _imwrite(path: Path, image, quality: int) -> None:
    """写入图片文件，失败时抛出带路径的异常。"""
    params: list[int] = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(path), image, params):
        raise IOError(f"图片写入失败: {path}")


# ---------------------------------------------------------------------------
# 数据格式转换
# ---------------------------------------------------------------------------

@dataclass
class ImageFormatResult(PreprocessRecord):
    """图片格式转换结果（报告 2.3「数据格式转换」）。"""

    converted: int = 0                    # 实际转换张数
    skipped: int = 0                      # 已是目标格式跳过张数
    failed: list[str] = field(default_factory=list)  # 转换失败文件


def convert_image_format(
    dataset_dir: str | Path,
    target_ext: str = ".jpg",
    *,
    replace: bool = False,
    quality: int = 95,
) -> ImageFormatResult:
    """统一图片格式（报告 2.3「数据格式转换」）。

    :param dataset_dir: 数据集根目录（无 ``images/`` 子目录时视为图片目录）。
    :param target_ext: 目标扩展名，如 ``.jpg``。
    :param replace: 转换成功后删除原文件（缺省保留两份）。
    :param quality: JPEG 输出质量（1-100）。
    :return: ``ImageFormatResult``，含可填写报告的记录字段。
    """
    ext = _normalize_ext(target_ext)
    if ext not in _WRITABLE_EXTS:
        raise ValueError(
            f"不支持的目标格式: {ext}（支持 {', '.join(sorted(_WRITABLE_EXTS))}）"
        )

    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    images = list_images(image_root_of(root))

    result = ImageFormatResult(
        item="数据格式转换",
        orig="、".join(sorted({p.suffix[1:].upper() for p in images})) or "无",
        method=f"统一转换为 {ext[1:].upper()}",
    )
    for path in images:
        if path.suffix.lower() == ext:
            result.skipped += 1
            continue
        image = _read_image(path)
        if image is None:
            result.failed.append(str(path))
            continue
        if image.ndim == 3 and image.shape[2] == 4 and ext in (".jpg", ".jpeg", ".bmp"):
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # JPEG/BMP 不支持 alpha 通道
        _imwrite(path.with_suffix(ext), image, quality)
        if replace:
            path.unlink()
        result.converted += 1

    result.result = f"已转换 {result.converted} 张" if result.converted else "无需处理（均为目标格式）"
    if result.failed:
        result.result += f"，失败 {len(result.failed)} 张（读取失败，见 failed 列表）"
    return result


# ---------------------------------------------------------------------------
# 图像尺寸处理
# ---------------------------------------------------------------------------

@dataclass
class ResizeResult(PreprocessRecord):
    """图片尺寸调整结果（报告 2.3「图像尺寸处理」）。"""

    resized: int = 0          # 实际调整张数
    skipped: int = 0          # 已是目标尺寸跳过张数
    labels_adjusted: int = 0  # 同步调整的标签文件数（letterbox 时）
    failed: list[str] = field(default_factory=list)


def _adjust_labels_letterbox(
    label_path: Path,
    width: int,
    height: int,
    scale: float,
    dx: int,
    dy: int,
    target_w: int,
    target_h: int,
) -> int:
    """按 letterbox 变换同步调整单个 YOLO 标签文件，返回是否调整（1/0）。

    原归一化中心点像素坐标 (cx*w, cy*h) 经缩放平移后映射到画布坐标，
    再除以画布尺寸得到新归一化值。
    """
    if not label_path.is_file():
        return 0
    lines_out: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) != 5:
            lines_out.append(line)  # 非检测框行原样保留
            continue
        cls = tokens[0]
        cx, cy, nw, nh = (float(v) for v in tokens[1:])
        cx = (cx * width * scale + dx) / target_w
        cy = (cy * height * scale + dy) / target_h
        nw = nw * width * scale / target_w
        nh = nh * height * scale / target_h
        lines_out.append(
            f"{cls} {cx:.{_DECIMALS}f} {cy:.{_DECIMALS}f} {nw:.{_DECIMALS}f} {nh:.{_DECIMALS}f}"
        )
    label_path.write_text(
        "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
    )
    return 1


def resize_images(
    dataset_dir: str | Path,
    size: tuple[int, int],
    *,
    letterbox: bool = False,
    labels_dir: str | Path | None = None,
    quality: int = 95,
) -> ResizeResult:
    """统一图片分辨率（报告 2.3「图像尺寸处理」），图片原地覆盖。

    :param dataset_dir: 数据集根目录（无 ``images/`` 子目录时视为图片目录）。
    :param size: 目标尺寸 (宽, 高)。
    :param letterbox: 保持宽高比、两侧填充（Ultralytics 风格）。直接缩放
        不改变 YOLO 归一化标签；letterbox 会同步调整标签坐标，因此要求
        能定位到标签目录（``labels_dir`` 缺省按 YOLO 布局自动推断）。
    :param labels_dir: 标签目录（letterbox 时自动推断失败则需显式指定）。
    :return: ``ResizeResult``，含可填写报告的记录字段。
    """
    target_w, target_h = int(size[0]), int(size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"目标尺寸必须为正数: {size}")

    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    image_root = image_root_of(root)
    images = list_images(image_root)

    labels_root: Path | None = None
    if letterbox:
        labels_root = (
            Path(labels_dir) if labels_dir is not None else labels_root_for(root, image_root)
        )
        if not labels_root.is_dir():
            raise ValueError(
                f"letterbox 需要同步调整标签坐标，未找到标签目录: {labels_root}；"
                "请通过 labels_dir 显式指定，或改用直接缩放（letterbox=False）"
            )

    result = ResizeResult(
        item="图像尺寸处理",
        orig=f"原始分辨率（{len(images)} 张）",
        method=f"统一为 {target_w}x{target_h}"
        + ("（letterbox 保持宽高比）" if letterbox else "（直接缩放）"),
    )

    for path in images:
        image = _read_image(path)
        if image is None:
            result.failed.append(str(path))
            continue
        height, width = image.shape[:2]
        if (width, height) == (target_w, target_h):
            result.skipped += 1
            continue

        if letterbox and labels_root is not None:
            scale = min(target_w / width, target_h / height)
            new_w, new_h = round(width * scale), round(height * scale)
            dx, dy = (target_w - new_w) // 2, (target_h - new_h) // 2
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(
                resized, dy, target_h - new_h - dy, dx, target_w - new_w - dx,
                cv2.BORDER_CONSTANT, value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
            )
            result.labels_adjusted += _adjust_labels_letterbox(
                _label_path(labels_root, image_root, path),
                width, height, scale, dx, dy, target_w, target_h,
            )
        else:
            padded = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        _imwrite(path, padded, quality)
        result.resized += 1

    if result.resized:
        result.result = f"已调整 {result.resized} 张"
        if letterbox:
            result.result += f"，同步调整标签 {result.labels_adjusted} 个"
    else:
        result.result = "无需处理（均为目标尺寸）"
    if result.failed:
        result.result += f"，失败 {len(result.failed)} 张"
    return result


# ---------------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------------

@dataclass
class CleanResult(PreprocessRecord):
    """数据清洗结果（报告 2.3「数据清洗」）。"""

    checked: int = 0                              # 检查图片数
    broken: list[str] = field(default_factory=list)  # 损坏 / 不可读图片
    removed_images: int = 0                       # 实际删除图片数
    removed_labels: int = 0                       # 实际删除标签数


def clean_dataset(dataset_dir: str | Path, *, dry_run: bool = True) -> CleanResult:
    """数据清洗（报告 2.3「数据清洗」）：剔除损坏 / 不可读的图片。

    判定标准：文件为空（0 字节）或无法被 OpenCV 解码。损坏图片的
    对应标签文件一并剔除。

    :param dataset_dir: 数据集根目录（无 ``images/`` 子目录时视为图片目录）。
    :param dry_run: 仅检测不删除（缺省 True，传 False 实际清理）。
    :return: ``CleanResult``，含可填写报告的记录字段。
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    image_root = image_root_of(root)
    labels_root = labels_root_for(root, image_root)
    images = list_images(image_root)

    result = CleanResult(
        item="数据清洗",
        orig=f"全量数据（{len(images)} 张）",
        method="剔除损坏 / 不可读图片（0 字节或无法解码）",
    )

    for path in images:
        result.checked += 1
        try:
            broken = path.stat().st_size == 0
        except OSError:
            broken = True
        if not broken:
            broken = _read_image(path) is None
        if not broken:
            continue
        result.broken.append(str(path))
        if dry_run:
            continue
        path.unlink()
        result.removed_images += 1
        label_path = _label_path(labels_root, image_root, path)
        if label_path.is_file():
            label_path.unlink()
            result.removed_labels += 1

    if not result.broken:
        result.result = "未发现损坏数据"
    elif dry_run:
        result.result = f"发现损坏数据 {len(result.broken)} 张（dry_run 未删除）"
    else:
        result.result = (
            f"已剔除图片 {result.removed_images} 张、对应标签 {result.removed_labels} 个"
        )
    return result


# ---------------------------------------------------------------------------
# 异常数据处理
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult(PreprocessRecord):
    """异常数据处理结果（报告 2.3「异常数据处理」）。"""

    checked_images: int = 0  # 检查图片数
    checked_labels: int = 0  # 检查标签数
    missing_labels: list[str] = field(default_factory=list)  # 无标签图片（YOLO 视为背景图，仅提示）
    orphan_labels: list[str] = field(default_factory=list)   # 无对应图片的标签（异常）
    fixed_files: list[str] = field(default_factory=list)     # 内容有问题的标签文件
    clipped_boxes: int = 0       # 坐标越界被裁剪的框数
    dropped_lines: int = 0       # 格式非法被剔除的行数
    removed_orphan_labels: int = 0  # 实际删除的孤儿标签数


def handle_anomalies(dataset_dir: str | Path, *, dry_run: bool = True) -> AnomalyResult:
    """异常数据处理（报告 2.3「异常数据处理」）。

    检查与处理内容：

    * 图片-标签配对：图片缺标签（YOLO 视为背景图，仅提示）、标签缺图片
      （孤儿标签，``dry_run=False`` 时删除）。
    * 标签内容：每行应为 ``class_id cx cy w h`` 共 5 列、class_id 为非负
      整数、坐标在 [0, 1] 内——格式非法行剔除、坐标越界裁剪
      （``dry_run=False`` 时写回文件）。

    :param dataset_dir: 数据集根目录（无 ``images/`` 子目录时视为图片目录）。
    :param dry_run: 仅检测不修改文件（缺省 True，传 False 实际修复）。
    :return: ``AnomalyResult``，含可填写报告的记录字段。
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    image_root = image_root_of(root)
    labels_root = labels_root_for(root, image_root)
    images = list_images(image_root)

    result = AnomalyResult(
        item="异常数据处理",
        method="图片-标签配对检查；非法标注行剔除、坐标越界裁剪",
    )

    if not labels_root.is_dir():
        result.checked_images = len(images)
        result.orig = f"{len(images)} 张图片 / 0 个标签"
        result.result = "未找到标签目录（无标注数据）"
        return result

    orphan_candidates = {
        p.resolve() for p in labels_root.rglob("*.txt") if p.is_file()
    }
    for path in images:
        result.checked_images += 1
        label_path = _label_path(labels_root, image_root, path)
        if not label_path.is_file():
            result.missing_labels.append(str(path))
            continue
        result.checked_labels += 1
        orphan_candidates.discard(label_path.resolve())

        changed = False
        lines_out: list[str] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 5:
                result.dropped_lines += 1
                changed = True
                continue
            try:
                class_id = int(tokens[0])
                cx, cy, nw, nh = (float(v) for v in tokens[1:])
            except ValueError:
                result.dropped_lines += 1
                changed = True
                continue
            if class_id < 0:
                result.dropped_lines += 1
                changed = True
                continue
            fixed = tuple(min(1.0, max(0.0, v)) for v in (cx, cy, nw, nh))
            if fixed != (cx, cy, nw, nh):
                result.clipped_boxes += 1
                changed = True
            lines_out.append(
                f"{class_id} {fixed[0]:.{_DECIMALS}f} {fixed[1]:.{_DECIMALS}f} "
                f"{fixed[2]:.{_DECIMALS}f} {fixed[3]:.{_DECIMALS}f}"
            )
        if changed:
            result.fixed_files.append(str(label_path))
            if not dry_run:
                label_path.write_text(
                    "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
                )

    result.orphan_labels = [str(p) for p in sorted(orphan_candidates)]
    if not dry_run:
        for orphan in result.orphan_labels:
            Path(orphan).unlink()
            result.removed_orphan_labels += 1

    result.orig = f"{result.checked_images} 张图片 / {result.checked_labels} 个标签"
    if not (result.missing_labels or result.orphan_labels or result.fixed_files):
        result.result = "未发现异常数据"
        return result

    parts: list[str] = []
    if result.missing_labels:
        parts.append(f"{len(result.missing_labels)} 张图片无标签（背景图）")
    if result.orphan_labels:
        action = "已删除" if not dry_run else "发现"
        parts.append(f"{action}孤儿标签 {len(result.orphan_labels)} 个")
    if result.fixed_files:
        action = "已修复" if not dry_run else "待修复"
        parts.append(
            f"{action}标签 {len(result.fixed_files)} 个"
            f"（越界框 {result.clipped_boxes} 个、非法行 {result.dropped_lines} 行）"
        )
    result.result = "；".join(parts)
    return result
