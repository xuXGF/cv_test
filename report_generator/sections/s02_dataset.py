"""第 2 章：测试数据集说明。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from tools import pct

from ..render import render

#: 计入数据量的图片扩展名
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

#: 项目根目录（sections -> report_generator -> 项目根），用于解析场景相对路径
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _count_images(scene_dir: Path) -> int:
    """统计场景目录下的图片张数。

    标准布局为 ``<场景目录>/images/``；不存在该子目录时统计场景目录本身。
    目录不存在时抛出带路径的异常，不做静默兜底。
    """
    image_dir = scene_dir / "images" if (scene_dir / "images").is_dir() else scene_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"2.2 边缘场景目录不存在: {scene_dir}")
    return sum(
        1 for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES
    )


_TEMPLATE = r"""# 2. 测试数据集说明

## 2.1 数据集基本信息

| 项目    | 内容           |
| ----- | ------------ |
| 数据集名称 | {{dataset_name}}       |
| 数据来源  | {{data_source}}       |
| 数据类型  | {{data_type=图片 / 视频 / 其他}} |
| 数据总量  | {{data_total}}       |
| 测试数据量 | {{test_data_count}}       |
| 类别数量  | {{num_classes}}       |
| 标注格式  | {{annotation_format}}       |
| 数据分辨率 | {{data_resolution}}       |
| 数据格式  | {{data_format}}       |

## 2.2 边缘场景数据

针对真实业务应用中的复杂环境，补充边缘场景数据，用于验证模型的实际泛化能力。

| 场景编号 | 测试场景    | 场景描述   |    数据量 |     占比 |
| ---- | ------- | ------ | -----: | -----: |
{{scene_rows}}

## 2.3 数据预处理及标准化

| 处理项目   | 原始数据   | 标准化处理方式 | 处理结果   |
| ------ | ------ | ------- | ------ |
| 数据格式转换 | {{pre_orig_format}} | {{pre_method_format}}  | {{pre_result_format}} |
| 标签格式转换 | {{pre_orig_label}} | {{pre_method_label}}  | {{pre_result_label}} |
| 图像尺寸处理 | {{pre_orig_size}} | {{pre_method_size}}  | {{pre_result_size}} |
| 数据清洗   | {{pre_orig_clean}} | {{pre_method_clean}}  | {{pre_result_clean}} |
| 异常数据处理 | {{pre_orig_anomaly}} | {{pre_method_anomaly}}  | {{pre_result_anomaly}} |"""


@dataclass
class DatasetSection:
    """第 2 章：测试数据集说明。"""

    # 2.1 数据集基本信息
    dataset_name: str = ""  # 数据集名称
    data_source: str = ""  # 数据来源
    data_type: str = ""  # 数据类型（未填时默认「图片 / 视频 / 其他」）
    data_total: str = ""  # 数据总量
    test_data_count: str = ""  # 测试数据量
    num_classes: str = ""  # 类别数量
    annotation_format: str = ""  # 标注格式
    data_resolution: str = ""  # 数据分辨率
    data_format: str = ""  # 数据格式

    # 2.2 边缘场景数据（由 fill_scenarios 按 local_info.scene 循环生成，不手填）
    scene_rows: str = ""  # 场景表格行（每行：编号/场景/描述/数据量/占比）

    # 2.3 数据预处理及标准化
    pre_orig_format: str = ""  # 数据格式转换-原始数据
    pre_method_format: str = ""  # 数据格式转换-标准化处理方式
    pre_result_format: str = ""  # 数据格式转换-处理结果
    pre_orig_label: str = ""  # 标签格式转换-原始数据
    pre_method_label: str = ""  # 标签格式转换-标准化处理方式
    pre_result_label: str = ""  # 标签格式转换-处理结果
    pre_orig_size: str = ""  # 图像尺寸处理-原始数据
    pre_method_size: str = ""  # 图像尺寸处理-标准化处理方式
    pre_result_size: str = ""  # 图像尺寸处理-处理结果
    pre_orig_clean: str = ""  # 数据清洗-原始数据
    pre_method_clean: str = ""  # 数据清洗-标准化处理方式
    pre_result_clean: str = ""  # 数据清洗-处理结果
    pre_orig_anomaly: str = ""  # 异常数据处理-原始数据
    pre_method_anomaly: str = ""  # 异常数据处理-标准化处理方式
    pre_result_anomaly: str = ""  # 异常数据处理-处理结果

    def fill_scenarios(self) -> None:
        """2.2 边缘场景：按 ``local_info.scene`` 的 path/desc 列表循环生成表格行。

        场景地址与描述由 ``local_info.json`` 的 ``scene`` 字段手动填写，
        ``scene.path`` 与 ``scene.desc`` 按下标一一对应；数据量为场景目录下
        图片张数（相对路径基于项目根），占比 = 场景数据量 / 测试数据量（2.1 节）。
        未配置场景时表格为空。
        """
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        if not isinstance(scene, dict):
            raise TypeError(
                f"local_info 的 scene 应为对象（含 path/desc 列表），实际为: {type(scene).__name__}"
            )
        paths = scene.get("path", [])
        descs = scene.get("desc", [])
        if not isinstance(paths, list) or not isinstance(descs, list):
            raise TypeError("scene.path 与 scene.desc 应为列表")
        if len(paths) != len(descs):
            raise ValueError(f"scene.path 与 scene.desc 数量不一致: {len(paths)} != {len(descs)}")
        if not paths:
            return

        try:
            total = int(self.test_data_count)
        except ValueError as exc:
            raise ValueError(
                f"计算 2.2 边缘场景占比需要测试数据量为整数，实际为: {self.test_data_count!r}"
            ) from exc

        rows: list[str] = []
        for i, (path, desc) in enumerate(zip(paths, descs), start=1):
            scene_dir = Path(path)
            if not scene_dir.is_absolute():
                scene_dir = _PROJECT_ROOT / scene_dir
            count = _count_images(scene_dir)
            ratio = pct(count, total, f"场景{i}占比")
            rows.append(f"| {i:02d} | **场景{i}** | {desc} | {count} | {ratio} |")
        self.scene_rows = "\n".join(rows)

    def render(self) -> str:
        """渲染本章节为 Markdown 文本。"""
        return render(_TEMPLATE, asdict(self))