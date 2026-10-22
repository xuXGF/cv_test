"""ONNX 模型元数据采集。

``collect_model_info()`` 输出的字段与测评报告 1.2 节「算法信息」、
1.4 节「测试环境信息」、3.1 节「模型配置」中的模型字段一一对应。

采集通道：

* ONNX Runtime 会话：输入/输出形状、执行提供者、以及 Ultralytics 导出模型
  携带的自定义元数据（任务类型、类别名称、输入尺寸等）。
* 纯标准库的 ONNX protobuf 解析：initializer 张量的参数量、精度推断、
  opset 版本，避免对 ``onnx`` 包的硬依赖。

说明：

* 采集不到的字段保持空字符串，不自动填入虚假值。
* FLOPs 需要图级算子计算，本模块不提供，保持为空。
* protobuf 解析仅覆盖顶层 graph 的 initializer，不含稀疏 initializer
  与子图（If/Loop）内部的 initializer。
"""

from __future__ import annotations

import ast
import json
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import onnxruntime as ort

# 包内自带的数据文件
_BUNDLED_MODEL = "yolov8n.onnx"
_BUNDLED_TEST_IMAGE = "test.jpg"


def default_model_path() -> Path:
    """包内自带模型的路径。"""
    return Path(__file__).resolve().parent / _BUNDLED_MODEL


def default_test_image_path() -> Path:
    """包内自带测试图片的路径。"""
    return Path(__file__).resolve().parent / _BUNDLED_TEST_IMAGE


# ---------------------------------------------------------------------------
# 极简 ONNX protobuf 解析（仅统计 initializer 与 opset，不构建完整对象模型）
# ---------------------------------------------------------------------------

# TensorProto data_type 编号 -> 是否计入参数量。
# float 系列与 8-bit 整数（量化权重）计入参数量；
# int32/int64 通常是形状/索引常量，不计入。
_COUNTABLE_DTYPES = {1, 10, 16, 2, 3, 9}  # float32, float16, bfloat16, uint8, int8, bool

_DTYPE_NAMES = {
    0: "undefined", 1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
    5: "int16", 6: "int32", 7: "int64", 9: "bool", 10: "float16",
    11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16",
}


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """读取一个 varint，返回 (值, 新位置)。"""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("protobuf 数据被截断")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    """迭代 protobuf 字段，产出 (field_number, wire_type, value)。"""
    pos, end = 0, len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_num, wire_type = tag >> 3, tag & 0x7
        if wire_type == 0:  # varint
            value, pos = _read_varint(buf, pos)
        elif wire_type == 1:  # 64-bit
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire_type == 5:  # 32-bit
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"不支持的 wire type: {wire_type}")
        yield field_num, wire_type, value


def _parse_tensor(buf: bytes) -> tuple[list[int], int]:
    """解析 TensorProto 的 dims 与 data_type，返回 (dims, data_type)。"""
    dims: list[int] = []
    dtype = 0
    for field_num, wire_type, value in _iter_fields(buf):
        if field_num == 1:  # dims（可为 packed 或逐个 varint）
            if wire_type == 2:
                pos, end = 0, len(value)
                while pos < end:
                    dim, pos = _read_varint(value, pos)
                    dims.append(dim)
            elif wire_type == 0:
                dims.append(value)
        elif field_num == 2 and wire_type == 0:  # data_type
            dtype = value
    return dims, dtype


def parse_onnx_static(model_path: Path) -> dict:
    """解析 ONNX 文件，返回参数量、dtype 分布与 opset 列表。"""
    data = model_path.read_bytes()
    param_count = 0
    dtype_counts: dict[str, int] = {}
    opsets: list[str] = []

    for field_num, wire_type, value in _iter_fields(data):
        if field_num == 8 and wire_type == 2:  # ModelProto.opset_import
            domain, version = "", 0
            for sub_num, sub_wtype, sub_value in _iter_fields(value):
                if sub_num == 1 and sub_wtype == 2:  # domain
                    domain = sub_value.decode("utf-8", "replace")
                elif sub_num == 2 and sub_wtype == 0:  # version
                    version = sub_value
            opsets.append(f"{domain or 'ai.onnx'}:{version}")
        elif field_num == 7 and wire_type == 2:  # ModelProto.graph
            for sub_num, sub_wtype, sub_value in _iter_fields(value):
                if sub_num == 5 and sub_wtype == 2:  # GraphProto.initializer
                    dims, dtype = _parse_tensor(sub_value)
                    dtype_name = _DTYPE_NAMES.get(dtype, str(dtype))
                    dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
                    if dtype in _COUNTABLE_DTYPES:
                        elements = 1
                        for dim in dims:
                            elements *= dim
                        param_count += elements

    return {"param_count": param_count, "dtype_counts": dtype_counts, "opsets": opsets}


# ---------------------------------------------------------------------------
# 模型信息数据模型与采集
# ---------------------------------------------------------------------------

def _infer_precision(dtype_counts: dict[str, int]) -> str:
    """根据 initializer dtype 分布推断模型精度。"""
    def has(name: str) -> bool:
        return dtype_counts.get(name, 0) > 0

    if has("int8") or has("uint8"):
        precision = "INT8 量化"
        if has("float32"):
            precision += "（混合 FP32）"
        return precision
    if has("float16") and not has("float32"):
        return "FP16"
    if has("bfloat16"):
        return "BF16"
    if has("float32"):
        return "FP32"
    return ""


_PRODUCER_NAMES = {"pytorch": "PyTorch", "tensorflow": "TensorFlow"}

_TASK_NAMES = {
    "detect": "目标检测",
    "segment": "实例分割",
    "classify": "图像分类",
    "pose": "关键点检测",
    "obb": "旋转框检测",
}

_ELEMENT_TYPE_NAMES = {"float": "float32", "double": "float64"}


def _element_type(tensor_type: str) -> str:
    """'tensor(float)' -> 'float32'。"""
    inner = tensor_type.removeprefix("tensor(").removesuffix(")")
    return _ELEMENT_TYPE_NAMES.get(inner, inner)


def _parse_class_names(raw: str) -> list[str] | None:
    """解析 Ultralytics 自定义元数据中的 names 字段，失败返回 None。"""
    if not raw:
        return None
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(names, dict):
        return [str(v) for v in names.values()]
    if isinstance(names, list):
        return [str(v) for v in names]
    return None


def _static_hw(session: ort.InferenceSession, custom: dict[str, str]) -> tuple[int, int]:
    """输入尺寸 (H, W)：优先取会话静态形状，其次取元数据 imgsz。"""
    shape = session.get_inputs()[0].shape
    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return int(shape[2]), int(shape[3])
    imgsz = custom.get("imgsz", "")
    try:
        parsed = ast.literal_eval(imgsz)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return int(parsed[0]), int(parsed[1])
    except (ValueError, SyntaxError):
        pass
    return 0, 0


@dataclass
class ModelInfo:
    """ONNX 模型元数据，字段与报告 1.2「算法信息」/ 1.4「测试环境信息」对应。"""

    model_name: str = ""  # 模型名称
    model_path: str = ""  # 模型文件路径
    algo_category: str = ""  # 算法类型（目标检测等）
    model_arch: str = ""  # 模型架构（如 YOLOv8n）
    model_version: str = ""  # 模型版本
    param_count: str = ""  # 参数量
    flops: str = ""  # FLOPs（图级计算不提供，保持为空）
    model_size: str = ""  # 模型大小
    model_precision: str = ""  # 模型精度（FP32/FP16/INT8 量化）
    input_size: str = ""  # 输入尺寸（HxW）
    batch_size: str = ""  # Batch Size
    input_format: str = ""  # 输入格式（布局 / 元素类型）
    output_format: str = ""  # 输出格式（形状 / 元素类型）
    num_classes: str = ""  # 类别数量
    train_framework: str = ""  # 训练框架
    original_model_format: str = ""  # 原始模型格式
    edge_model_format: str = ""  # 边缘模型格式（ONNX）
    onnx_opset: str = ""  # ONNX opset
    inference_framework: str = ""  # 推理框架
    runtime_version: str = ""  # Runtime 版本
    providers: str = ""  # 执行提供者
    python_version: str = ""  # Python 版本


def collect_model_info(model_path: str | Path | None = None) -> ModelInfo:
    """采集 ONNX 模型的静态元数据与运行环境信息。

    :param model_path: 模型文件路径，缺省使用包内自带的 ``yolov8n.onnx``。
    :return: ``ModelInfo``，采集不到的字段为空字符串。
    """
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta()
    custom = meta.custom_metadata_map
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    static = parse_onnx_static(path)

    task = custom.get("task", "")
    class_names = _parse_class_names(custom.get("names", ""))
    producer = meta.producer_name.strip()
    producer_display = _PRODUCER_NAMES.get(producer.lower(), producer)

    # 输入尺寸与 batch：静态形状优先，动态形状回退到元数据。
    input_h, input_w = _static_hw(session, custom)
    batch = input_info.shape[0] if input_info.shape else ""
    if not isinstance(batch, int):
        batch = custom.get("batch", "")

    # 类别数量：优先元数据 names，其次按检测输出布局推断（4 + N）。
    if class_names is not None:
        num_classes = str(len(class_names))
    elif len(output_info.shape) == 3 and isinstance(output_info.shape[1], int) \
            and output_info.shape[1] > 4:
        num_classes = str(output_info.shape[1] - 4)
    else:
        num_classes = ""

    # 模型架构：从描述中提取（如 "Ultralytics YOLOv8n model ..." -> YOLOv8n）。
    description = custom.get("description", "") or meta.description or ""
    arch_match = re.search(r"Ultralytics (\S+) model", description)
    model_arch = arch_match.group(1) if arch_match else ""

    input_h_str = str(input_h) if input_h else "动态"
    input_w_str = str(input_w) if input_w else "动态"

    return ModelInfo(
        model_name=path.stem,
        model_path=str(path),
        algo_category=_TASK_NAMES.get(task.lower(), task),
        model_arch=model_arch,
        model_version=custom.get("version", ""),
        param_count=str(static["param_count"]) if static["param_count"] else "",
        flops="",
        model_size=f"{path.stat().st_size / 1024 / 1024:.2f} MB",
        model_precision=_infer_precision(static["dtype_counts"]),
        input_size=f"{input_h_str}x{input_w_str}",
        batch_size=str(batch) if batch != "" else "",
        input_format=f"{'NCHW' if len(input_info.shape) == 4 else ''} / "
                     f"{_element_type(input_info.type)}".strip(" /"),
        output_format=f"{list(output_info.shape)} / {_element_type(output_info.type)}",
        num_classes=num_classes,
        train_framework=producer_display,
        original_model_format=producer_display,
        edge_model_format="ONNX",
        onnx_opset=", ".join(static["opsets"]),
        inference_framework="onnxruntime",
        runtime_version=ort.__version__,
        providers=", ".join(session.get_providers()),
        python_version=platform.python_version(),
    )


def save_model_info(info: ModelInfo, path: str | Path) -> Path:
    """将 ``ModelInfo`` 写为 JSON 文件，返回文件路径。"""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return json_path
