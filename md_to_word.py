#!/usr/bin/env python3
"""将 Markdown 测评报告转换为 Word(.docx)，版式尽量与 Markdown 预览保持一致。

用法：python3 md_to_word.py input.md output.docx
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.table import (
    WD_ALIGN_VERTICAL,
    WD_ROW_HEIGHT_RULE,
    WD_TABLE_ALIGNMENT,
)
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_FONT = "微软雅黑"
LATIN_FONT = "Arial"
CODE_FONT = "Courier New"
BODY_SIZE = Pt(10.5)
TABLE_SIZE = Pt(9)
PAGE_TABLE_SIZE = Pt(11)
TEXT_COLOR = RGBColor(0x1F, 0x23, 0x28)
MUTED_COLOR = RGBColor(0x57, 0x60, 0x6A)
BORDER_COLOR = "BFBFBF"
HEADER_FILL = "F2F4F7"

# 页面正文可用宽度（A4 21cm 减去左右各 2cm 页边距）
CONTENT_WIDTH_CM = 17.0

# 整页表格高度：A4 29.7cm 减去上下页边距各 2cm 后，扣除标题、日期与分页段落占用，
# 余量用于让日期留在表格下方同页
PAGE_TABLE_HEIGHT_CM = 20.0

ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!@~>|])")
INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")
COVER_OPEN_RE = re.compile(r"^<div\s+align=\"center\"\s*>$", re.I)
PAGE_BREAK_RE = re.compile(r"page-break-(?:after|before)\s*:\s*always", re.I)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)
BR_RE = re.compile(r"<br\s*/?>", re.I)
HTML_STRONG_RE = re.compile(r"</?(?:strong|b)>", re.I)
HTML_EM_RE = re.compile(r"</?(?:em|i)>", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CENTER_P_RE = re.compile(r"^<p\s+align=\"center\"\s*>(.*?)</p>$", re.I)
RIGHT_P_RE = re.compile(r"^<p\s+align=\"right\"\s*>(.*?)</p>$", re.I)

#: 表格无表头标记：其后紧跟的表格不渲染表头行样式（首行按数据行处理）
NO_HEADER_TABLE_MARK = "<!-- table-no-header -->"
#: 表格铺满整页标记：其后紧跟的表格行高等分页面正文高度
PAGE_FILL_TABLE_MARK = "<!-- table-fill-page -->"


def display_width(text: str) -> int:
    """按显示宽度估算字符数，中文按 2 个宽度计算。"""
    total = 0
    for ch in text:
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def unescape_markdown(text: str) -> str:
    return ESCAPE_RE.sub(r"\1", text)


def apply_run_font(run, size=None, bold=None, italic=None, color=None, code=False):
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if size is not None:
        run.font.size = size
    if color is not None:
        run.font.color.rgb = color

    font_name = CODE_FONT if code else LATIN_FONT
    run.font.name = font_name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)
    rfonts.set(qn("w:eastAsia"), CODE_FONT if code else BODY_FONT)


def strip_html_inline(text: str) -> str:
    """将行内 HTML 转成等效 Markdown，并移除其余标签。

    ``<strong>/<b>`` 转为 ``**``，``<em>/<i>`` 转为 ``*``，避免标签以字面量
    出现在 Word 中（如 ``<p align="center">`` 内的标题）。
    """
    text = HTML_STRONG_RE.sub("**", text)
    text = HTML_EM_RE.sub("*", text)
    return HTML_TAG_RE.sub("", text)


def add_inline(paragraph, text, size=None, bold=False, color=None):
    """渲染行内的 **粗体**、*斜体*、`代码`。"""
    text = strip_html_inline(unescape_markdown(text))
    for token in INLINE_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            run = paragraph.add_run(token[2:-2])
            apply_run_font(run, size=size, bold=True, color=color)
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            run = paragraph.add_run(token[1:-1])
            apply_run_font(run, size=size, color=color, code=True)
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            run = paragraph.add_run(token[1:-1])
            apply_run_font(run, size=size, bold=bold, italic=True, color=color)
        else:
            run = paragraph.add_run(token)
            apply_run_font(run, size=size, bold=bold, color=color)


def set_style_font(style, size=None, bold=None):
    style.font.name = LATIN_FONT
    if size is not None:
        style.font.size = size
    if bold is not None:
        style.font.bold = bold
    style.font.color.rgb = TEXT_COLOR
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), LATIN_FONT)
    rfonts.set(qn("w:hAnsi"), LATIN_FONT)
    rfonts.set(qn("w:eastAsia"), BODY_FONT)


def add_paragraph_bottom_border(paragraph, color="D0D7DE", size=6):
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    ppr.append(pbdr)


def add_paragraph_left_border(paragraph, color="D0D7DE", size=18):
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), "8")
    left.set(qn("w:color"), color)
    pbdr.append(left)
    ppr.append(pbdr)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F6F8FA")
    ppr.append(shd)


def tighten_break_paragraph(paragraph) -> None:
    """压缩分页段落自身高度，避免分页后在下一页顶部多占一行空白。"""
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = Pt(1)


def init_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = LATIN_FONT
    normal.font.size = BODY_SIZE
    normal.font.color.rgb = TEXT_COLOR
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    set_style_font(normal, size=BODY_SIZE)

    heading_sizes = {1: 18, 2: 14, 3: 12, 4: 11}
    for level, size in heading_sizes.items():
        style = doc.styles[f"Heading {level}"]
        set_style_font(style, size=Pt(size), bold=True)
        fmt = style.paragraph_format
        fmt.space_before = Pt(14 if level == 1 else 10)
        fmt.space_after = Pt(6 if level == 1 else 4)
        fmt.keep_with_next = True
        fmt.line_spacing = 1.15
    return doc


def add_heading(doc, level, text):
    level = min(level, 4)
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    add_inline(paragraph, text, bold=True, color=TEXT_COLOR)
    if level == 1:
        add_paragraph_bottom_border(paragraph)


def add_horizontal_rule(doc):
    paragraph = doc.add_paragraph()
    add_paragraph_bottom_border(paragraph, color="D0D7DE", size=8)


def is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:]


def is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return "-" in stripped and "|" in stripped and all(c in "|-: \t" for c in stripped)


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def parse_alignment(separator: str) -> list[str]:
    aligns = []
    for cell in split_table_row(separator):
        cell = cell.strip()
        if cell.startswith(":") and cell.endswith(":"):
            aligns.append("center")
        elif cell.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), BORDER_COLOR)
        borders.append(element)
    tbl_pr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook"
    )

    cell_mar = OxmlElement("w:tblCellMar")
    for side, width in (("top", "40"), ("bottom", "40"), ("left", "80"), ("right", "80")):
        element = OxmlElement(f"w:{side}")
        element.set(qn("w:w"), width)
        element.set(qn("w:type"), "dxa")
        cell_mar.append(element)
    tbl_pr.insert_element_before(cell_mar, "w:tblLook")


def set_table_full_width(table):
    tbl_pr = table._tbl.tblPr
    width = tbl_pr.find(qn("w:tblW"))
    if width is None:
        width = OxmlElement("w:tblW")
        tbl_pr.insert(0, width)
    width.set(qn("w:type"), "dxa")
    width.set(qn("w:w"), str(int(CONTENT_WIDTH_CM / 2.54 * 1440)))

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.insert_element_before(layout, "w:tblCellMar", "w:tblLook")
    layout.set(qn("w:type"), "fixed")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.insert_element_before(
        shd, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark"
    )


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def add_table(
    doc, block: list[str], header: bool = True, fill_page: bool = False
):
    rows = []
    aligns = None
    for index, line in enumerate(block):
        if is_separator_row(line):
            if aligns is None:
                aligns = parse_alignment(line)
            continue
        if index == 0 and is_separator_row(line):
            continue
        rows.append(split_table_row(line))

    if not rows:
        return

    col_count = max(len(row) for row in rows)
    for row in rows:
        while len(row) < col_count:
            row.append("")
    if aligns is None:
        aligns = ["left"] * col_count
    while len(aligns) < col_count:
        aligns.append("left")

    table = doc.add_table(rows=len(rows), cols=col_count)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_full_width(table)
    set_table_borders(table)

    # 整页表格：行高均分页面正文高度，使表格上下铺满整页
    cell_size = PAGE_TABLE_SIZE if fill_page else TABLE_SIZE
    if fill_page:
        row_height = Cm(PAGE_TABLE_HEIGHT_CM / len(rows))
        for row in table.rows:
            row.height = row_height
            row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST

    # Markdown 表头行 = 首行 + 分隔行；无表头表格（如预览表）由标记关闭该样式
    looks_like_header = bool(
        header
        and len(block) > 1
        and is_table_row(block[0])
        and is_separator_row(block[1])
    )

    weights = []
    for col in range(col_count):
        widest = max(display_width(row[col]) for row in rows)
        weights.append(min(max(widest, 6), 40))
    total_weight = sum(weights) or 1
    widths = [CONTENT_WIDTH_CM * w / total_weight for w in weights]

    for col_index, width in enumerate(widths):
        table.columns[col_index].width = Cm(width)

    align_map = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }

    for row_index, row in enumerate(rows):
        is_header = row_index == 0 and looks_like_header
        if is_header:
            set_repeat_header(table.rows[row_index])
        for col_index, text in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.width = Cm(widths[col_index])
            if is_header:
                set_cell_shading(cell, HEADER_FILL)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

            paragraph = cell.paragraphs[0]
            paragraph.alignment = align_map[aligns[col_index]]
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.line_spacing = 1.1
            # 单元格内 <br> 转为换行，支持多行内容
            for line_index, segment in enumerate(BR_RE.split(text)):
                if line_index:
                    paragraph.add_run().add_break()
                add_inline(paragraph, segment, size=cell_size, bold=is_header)


def add_quote(doc, lines: list[str]):
    for text in lines:
        paragraph = doc.add_paragraph()
        add_paragraph_left_border(paragraph)
        fmt = paragraph.paragraph_format
        fmt.left_indent = Cm(0.5)
        fmt.space_before = Pt(2)
        fmt.space_after = Pt(4)
        fmt.line_spacing = 1.2
        add_inline(paragraph, text, color=MUTED_COLOR)


def add_list(doc, items: list[str], ordered: bool):
    for index, item in enumerate(items, start=1):
        prefix = f"{index}. " if ordered else "• "
        if item.startswith("[ ] "):
            prefix, item = "□ ", item[4:]
        elif item.lower().startswith("[x] "):
            prefix, item = "☑ ", item[4:]

        paragraph = doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.left_indent = Cm(0.6)
        fmt.first_line_indent = Cm(-0.3)
        fmt.space_before = Pt(1)
        fmt.space_after = Pt(2)
        fmt.line_spacing = 1.2
        bullet = paragraph.add_run(prefix)
        apply_run_font(bullet, size=BODY_SIZE, color=TEXT_COLOR)
        add_inline(paragraph, item)


def add_body_paragraph(doc, text: str):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(6)
    add_inline(paragraph, text)


def add_cover(doc, lines: list[str]):
    """渲染封面：居中标题 + 底部日期。"""
    block = "\n".join(lines)
    title_match = H1_RE.search(block)
    date_match = P_RE.search(block)

    if title_match:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Cm(7.5)
        paragraph.paragraph_format.space_after = Cm(11.0)
        add_inline(paragraph, title_match.group(1).strip(), size=Pt(24), bold=True, color=TEXT_COLOR)

    if date_match:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        add_inline(paragraph, date_match.group(1).strip(), size=Pt(12), color=TEXT_COLOR)


def convert(input_path: Path, output_path: Path):
    doc = init_document()
    lines = input_path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    index = 0
    next_table_header = True
    next_table_fill_page = False

    while index < total:
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped == NO_HEADER_TABLE_MARK:
            next_table_header = False
            index += 1
            continue

        if stripped == PAGE_FILL_TABLE_MARK:
            next_table_fill_page = True
            index += 1
            continue

        if PAGE_BREAK_RE.search(stripped):
            tighten_break_paragraph(doc.add_page_break())
            index += 1
            continue

        if COVER_OPEN_RE.match(stripped):
            block = []
            index += 1
            while index < total and lines[index].strip().lower() != "</div>":
                block.append(lines[index])
                index += 1
            if index < total:
                index += 1
            add_cover(doc, block)
            continue

        center_match = CENTER_P_RE.match(stripped)
        if center_match:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(12)
            add_inline(
                paragraph,
                center_match.group(1).strip(),
                size=Pt(18),
                bold=True,
                color=TEXT_COLOR,
            )
            index += 1
            continue

        right_match = RIGHT_P_RE.match(stripped)
        if right_match:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(0)
            add_inline(
                paragraph,
                right_match.group(1).strip(),
                size=Pt(12),
                color=TEXT_COLOR,
            )
            index += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading_match:
            add_heading(doc, len(heading_match.group(1)), heading_match.group(2))
            index += 1
            continue

        if re.fullmatch(r"\s*([-*_])(\s*\1){2,}\s*", line):
            add_horizontal_rule(doc)
            index += 1
            continue

        if is_table_row(line):
            block = []
            while index < total and is_table_row(lines[index]):
                block.append(lines[index])
                index += 1
            add_table(
                doc,
                block,
                header=next_table_header,
                fill_page=next_table_fill_page,
            )
            next_table_header = True
            next_table_fill_page = False
            continue

        if stripped.startswith(">"):
            block = []
            while index < total and lines[index].strip().startswith(">"):
                block.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            add_quote(doc, block)
            continue

        if re.match(r"^\s*[-*+]\s+", line):
            block = []
            while index < total and re.match(r"^\s*[-*+]\s+", lines[index]):
                block.append(re.sub(r"^\s*[-*+]\s+", "", lines[index]).strip())
                index += 1
            add_list(doc, block, ordered=False)
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            block = []
            while index < total and re.match(r"^\s*\d+\.\s+", lines[index]):
                block.append(re.sub(r"^\s*\d+\.\s+", "", lines[index]).strip())
                index += 1
            add_list(doc, block, ordered=True)
            continue

        add_body_paragraph(doc, stripped)
        index += 1

    doc.save(str(output_path))
    print(f"已生成：{output_path}")


def main():
    # if len(sys.argv) < 3:
    #     print("用法：python3 md_to_word.py 输入.md 输出.docx")
    #     sys.exit(1)
    argv1 = "../output/report_sample.md"
    argv2 = '测试报告.docx'
    convert(Path(argv1), Path(argv2))


if __name__ == "__main__":
    main()
