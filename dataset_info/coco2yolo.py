"""COCO 标注 -> YOLO txt 标注转换。

将 COCO JSON（images / annotations / categories）中的检测框标注转换为
YOLO 检测格式：每张图片一个同名 ``.txt``，每行::

    class_id cx cy w h

其中 ``cx cy w h`` 为相对图片宽高的归一化值（0~1），
``class_id`` 按 categories 的 id 升序映射为 0 起始的连续编号。

仅使用标准库，不依赖第三方包。

用法::

    from dataset_info.coco2yolo import convert_coco_to_yolo

    result = convert_coco_to_yolo(
        "data/coco/annotations/instances_train2017.json",
        "data/coco/labels/train2017",
        image_dir="data/coco/images/train2017",  # 可选：仅转换磁盘上实际存在的图片
    )
    print(result.image_count, result.annotation_count)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: 每行坐标保留的小数位数（与常见 YOLO 标注一致）
_DECIMALS = 6

#: 视为图片的扩展名（小写）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class PreprocessRecord:
    """一次预处理操作的记录，对应报告 2.3 节某一行（原始数据 / 处理方式 / 处理结果）。

    ``preprocess`` 模块中各处理函数的返回值均继承本类，可传入
    ``collect_dataset_info(records=...)`` 填入报告对应行。
    """

    item: str = ""    # 处理项目（数据格式转换 / 标签格式转换 / 图像尺寸处理 / 数据清洗 / 异常数据处理）
    orig: str = ""    # 原始数据
    method: str = ""  # 标准化处理方式
    result: str = ""  # 处理结果


@dataclass
class Coco2YoloResult(PreprocessRecord):
    """一次 COCO -> YOLO 转换的统计结果（报告 2.3「标签格式转换」）。"""

    coco_json: str = ""           # 输入的 COCO JSON 路径
    output_dir: str = ""          # 标签输出目录
    image_count: int = 0          # 写出标签文件的图片数
    annotation_count: int = 0     # 写出的有效标注框数
    skipped_annotations: int = 0  # 跳过的无效标注框数（缺 bbox / 尺寸非法 / 类别未知）
    empty_label_count: int = 0    # 无标注图片数（写出空 txt，作为背景图）
    missing_image_count: int = 0  # JSON 中记录但磁盘上缺失的图片数（仅提供 image_dir 时统计）
    class_names: list[str] = field(default_factory=list)  # 类别名（按 class_id 升序）


def _clip(value: float) -> float:
    """把归一化坐标夹取到 [0, 1]，规避浮点误差导致的越界。"""
    return min(1.0, max(0.0, value))


def _load_coco_json(json_path: Path) -> dict:
    """读取并校验 COCO JSON，失败时抛出带明确信息的异常。"""
    try:
        text = json_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"COCO 标注文件不存在: {json_path}") from exc
    except OSError as exc:
        raise OSError(f"COCO 标注文件读取失败: {json_path}（{exc}）") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"COCO 标注文件不是合法 JSON: {json_path}（第 {exc.lineno} 行：{exc.msg}）"
        ) from exc
    if not isinstance(data, dict):
        raise TypeError(f"COCO 标注文件顶层应为 JSON 对象，实际为 {type(data).__name__}: {json_path}")
    return data


# ---------------------------------------------------------------------------
# 数据集布局（info / preprocess 共用）
# ---------------------------------------------------------------------------

def image_root_of(root: Path) -> Path:
    """图片根目录：标准 YOLO/COCO 布局为 ``root/images``，否则为数据集根目录。"""
    image_root = root / "images"
    return image_root if image_root.is_dir() else root


def labels_root_for(root: Path, image_root: Path) -> Path:
    """与图片目录对应的标签目录（YOLO 布局中 ``images`` 平级替换为 ``labels``）。"""
    rel = image_root.relative_to(root)
    if rel.parts and rel.parts[0] == "images":
        return root.joinpath("labels", *rel.parts[1:])
    return root / "labels"


def list_images(image_root: Path) -> list[Path]:
    """递归列出图片文件（按路径排序），目录不存在或无图片时抛出异常。"""
    if not image_root.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_root}")
    images = sorted(
        p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise ValueError(
            f"未找到图片文件（支持 {', '.join(sorted(IMAGE_EXTS))}）: {image_root}"
        )
    return images


def detect_coco_jsons(root: Path) -> list[Path]:
    """按标准 COCO 布局探测标注文件（``annotations/*.json``）。"""
    annotations_dir = root / "annotations"
    if not annotations_dir.is_dir():
        return []
    return sorted(annotations_dir.glob("*.json"))


def conversion_layout(root: Path, json_path: Path) -> tuple[Path, Path]:
    """确定转换的图片目录与标签输出目录（标准 COCO/YOLO 布局）。

    split 名取自 JSON 文件名（去掉 ``instances_`` 等常见前缀），
    与 ``images/`` 下的同名子目录匹配；无同名子目录但恰有一个子目录时
    使用该子目录；否则直接使用 ``images/``。
    """
    image_root = root / "images"
    if not image_root.is_dir():
        raise ValueError(
            f"数据集目录下未找到 images 子目录，无法自动定位图片: {root}；"
            "请手动调用 convert_coco_to_yolo(json, 输出目录, image_dir=图片目录)"
        )
    stem = json_path.stem
    for prefix in ("instances_", "labels_", "annotations_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    split_dir = image_root / stem
    if split_dir.is_dir():
        image_dir = split_dir
    else:
        subdirs = sorted(d for d in image_root.iterdir() if d.is_dir())
        image_dir = subdirs[0] if len(subdirs) == 1 else image_root
    return image_dir, labels_root_for(root, image_dir)


def _label_path(labels_root: Path, image_root: Path, image_path: Path) -> Path:
    """图片对应的标签文件路径（YOLO 布局镜像关系）。"""
    return labels_root / image_path.relative_to(image_root).with_suffix(".txt")


def convert_coco_to_yolo(
    coco_json: str | Path,
    output_dir: str | Path,
    *,
    image_dir: str | Path | None = None,
) -> Coco2YoloResult:
    """将 COCO JSON 标注转换为 YOLO txt 标注。

    :param coco_json: COCO 标注 JSON 路径。
    :param output_dir: 标签输出目录（不存在时自动创建）。
    :param image_dir: 图片目录。提供时以磁盘上实际存在的图片为准
        （JSON 中有标注但磁盘缺失的图片跳过；磁盘有图但 JSON 未标注的
        图片写出空 txt 作为背景图）；缺省时按 JSON 中的 images 列表写出。
    :return: ``Coco2YoloResult`` 转换统计。
    """
    json_path = Path(coco_json)
    data = _load_coco_json(json_path)

    images = data.get("images") or []
    annotations = data.get("annotations") or []
    categories = data.get("categories") or []
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise TypeError(
            f"COCO 标注文件的 images/annotations 应为列表: {json_path}"
        )

    # 类别映射：按 category id 升序映射为 0 起始的连续编号
    sorted_categories = sorted(categories, key=lambda c: c["id"])
    class_map = {int(c["id"]): i for i, c in enumerate(sorted_categories)}
    class_names = [str(c.get("name", c["id"])) for c in sorted_categories]

    images_by_id = {int(img["id"]): img for img in images if img.get("id") is not None}

    # 按图片聚合有效标注框
    anns_by_image: dict[int, list[dict]] = {}
    skipped = 0
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id not in images_by_id or ann.get("category_id") not in class_map:
            skipped += 1
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            skipped += 1
            continue
        anns_by_image.setdefault(int(image_id), []).append(ann)

    # 确定待写出标签的图片清单：(相对路径, JSON 图片记录或 None)
    out_root = Path(output_dir)
    if image_dir is not None:
        img_root = Path(image_dir)
        if not img_root.is_dir():
            raise FileNotFoundError(f"图片目录不存在: {img_root}")
        json_by_name = {
            Path(img.get("file_name", str(img_id))).name: img
            for img_id, img in images_by_id.items()
        }
        disk_files = list_images(img_root)
        targets = [(p.relative_to(img_root), json_by_name.get(p.name)) for p in disk_files]
        missing = len(images_by_id) - sum(1 for _, img in targets if img is not None)
    else:
        targets = [
            (Path(img.get("file_name", str(img_id))), img)
            for img_id, img in images_by_id.items()
        ]
        missing = 0

    out_root.mkdir(parents=True, exist_ok=True)
    result = Coco2YoloResult(
        item="标签格式转换",
        orig="COCO JSON 标注（bbox 为 x, y, w, h 绝对像素坐标）",
        method=(
            "coco2yolo 转换：检测框中心化并按图片宽高归一化，"
            "类别按 category id 升序重编号"
        ),
        coco_json=str(json_path),
        output_dir=str(out_root),
        skipped_annotations=skipped,
        missing_image_count=missing,
        class_names=class_names,
    )

    for rel, img in targets:
        lines: list[str] = []
        if img is not None:
            img_w = int(img.get("width") or 0)
            img_h = int(img.get("height") or 0)
            for ann in anns_by_image.get(int(img["id"]), []):
                x, y, w, h = (float(v) for v in ann["bbox"])
                # 像素框夹取到图片范围内，完全出界的框跳过
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
                if img_w <= 0 or img_h <= 0 or x2 <= x1 or y2 <= y1:
                    result.skipped_annotations += 1
                    continue
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                nw = (x2 - x1) / img_w
                nh = (y2 - y1) / img_h
                lines.append(
                    f"{class_map[int(ann['category_id'])]} "
                    f"{_clip(cx):.{_DECIMALS}f} {_clip(cy):.{_DECIMALS}f} "
                    f"{_clip(nw):.{_DECIMALS}f} {_clip(nh):.{_DECIMALS}f}"
                )
                result.annotation_count += 1
        # img 为 None 时：磁盘存在但 JSON 未标注，写出空 txt 作为背景图

        if not lines:
            result.empty_label_count += 1

        target_txt = out_root / Path(rel).with_suffix(".txt")
        target_txt.parent.mkdir(parents=True, exist_ok=True)
        target_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        result.image_count += 1

    result.result = f"YOLO txt（{result.image_count} 张图片 / {result.annotation_count} 个标注框）"
    if result.skipped_annotations:
        result.result += f"，跳过无效标注 {result.skipped_annotations} 处"
    if result.missing_image_count:
        result.result += f"，JSON 记录但磁盘缺失图片 {result.missing_image_count} 张"
    return result
