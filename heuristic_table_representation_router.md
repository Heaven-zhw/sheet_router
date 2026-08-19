# 启发式单模态表格表示路由

目标：对每个样本只选择一种表格输入格式，不依赖具体大模型，只根据表格文件和查询内容进行路由。当前先覆盖 RealHiTBench 的 QA/CoT 和 SpreadsheetBench 的 Manipulation/PoT，不考虑文本+图像混合输入。

## 1. 经验结论

现有统计给出的强信号如下。

- QA 任务中，文本格式整体比纯图像稳定。跨模型平均看，`latex` 是最稳默认；`json_cells` 和 `markdown` 紧随其后。
- QA 任务的 oracle 明显高于任何单一格式，说明格式选择有空间；但纯图像不是通用默认，只应在视觉/样式信息重要时触发。
- Manipulation/PoT 中，`json_cells` 是最稳默认：平均 hard_all 最高、平均排名第一，并且 execution/valid 也更稳。
- Manipulation 中 `json_rows` 是 `json_cells` 后最有价值的文本候选，适合行记录式操作；图像格式主要用于样式、布局、格式参考等文本序列化丢失的信息。
- `default_image` 只作为兜底渲染图像，不应优先于数据集原始图像或 Excel 风格截图。

因此路由思想是：先判断任务是否需要视觉样式；否则 Manipulation 默认走坐标友好的 JSON，QA 默认走结构友好的 LaTeX/JSON。

## 2. 候选格式优先级

QA/CoT 候选：

```text
latex > json_cells > markdown > json_rows > html > dataframe > csv > image > excel_1_image > default_image
```

Manipulation/PoT 候选：

```text
json_cells > json_rows > markdown > latex > html > dataframe > csv > excel_1_image > image > default_image
```

图像内部优先级：

```text
RealHiTBench: image > excel_1_image > default_image
SpreadsheetBench: excel_1_image > image > default_image
```

解释：RealHiTBench 的 `image` 是原始图片，保留真实视觉线索；SpreadsheetBench 中 `excel_1_image` 更接近 Excel 风格和格式保真，`default_image` 是运行时兜底。

## 3. 可抽取特征

路由器只需要实现轻量特征，不需要训练模型。

### 3.1 查询特征

从 question/instruction 中用关键词和正则抽取：

- `style_query`：是否问颜色、背景、加粗、斜体、边框、合并单元格、格式、样式、宽度、高度、截图、图表、图片、format reference。
- `coordinate_query`：是否出现 A1、B2:C10、row 3、column B、sheet name、指定单元格/范围、answer_position、answer_sheet。
- `record_query`：是否是按行记录处理，如 match、duplicate、lookup、filter、sort、group、sum by、date/ref、for each row。
- `numeric_query`：是否涉及 max/min/sum/average/count/rate/percentage/exceed/below/between/top/bottom。
- `structure_query`：是否涉及 header、row header、column header、multi-level、nested table、sub-table、above/below、left/right、parent/child、跨表 join。
- `manipulation_type`：SpreadsheetBench 直接读 `instruction_type`，分为 Cell-Level Manipulation 和 Sheet-Level Manipulation。

### 3.2 表格特征

从 xlsx 或已有元数据抽取：

- `n_sheets`
- 每个 sheet 的 `used_range_rows`、`used_range_cols`
- `non_empty_cells` 和 `density = non_empty_cells / used_range_area`
- `merged_range_count`、`max_merged_row_span`、`max_merged_col_span`
- `has_style_signal`：是否存在非默认填充色、字体加粗/斜体、边框、合并、列宽/行高异常、条件格式。
- `is_sparse`：density < 0.35 或 used range 中空白块很多。
- `is_wide`：列数 >= 12。
- `is_long`：行数 >= 80。
- `is_large`：估计文本 token 超过预算的 70%。
- `available_formats`：当前样本实际可用的格式。

## 4. 主决策规则

按以下顺序执行，命中高优先级规则后直接返回格式。

### Rule A: 视觉/样式优先

如果满足任一条件：

- `style_query = true`
- instruction 明确要求“保持/参考格式”
- 查询答案依赖颜色、背景、边框、加粗、合并、图表、图片、视觉位置
- 表格 `has_style_signal = true` 且查询不是纯数值/纯文本检索

则选择图像格式：

```text
if dataset == RealHiTBench: choose first available of [image, excel_1_image, default_image]
if dataset == SpreadsheetBench: choose first available of [excel_1_image, image, default_image]
```

注意：如果只是普通 merged headers，不算强视觉样式；优先交给 `latex` 或 `json_cells`。

### Rule B: Manipulation/PoT 默认

SpreadsheetBench 默认选择：

```text
json_cells
```

因为 PoT 需要写代码，`json_cells` 同时保留 cell address、sheet name、used range 和 merged ranges，最适合生成可执行的 openpyxl/pandas 代码。

从 `json_cells` 切到 `json_rows` 的条件：

- `record_query = true`
- 操作主要是按行匹配、分组、聚合、过滤、排序、去重、追加行
- 表格不是强稀疏表：`density >= 0.35`
- 查询不强依赖精确单元格样式或零散坐标

保留 `json_cells` 的条件：

- Cell-Level Manipulation
- 需要写入指定 cell/range
- 多 sheet、稀疏表、大量空白或合并单元格
- 指令要求复制/清空/插入/删除具体区域
- 需要根据 answer_position/answer_sheet 精确定位输出

### Rule C: QA/CoT 默认

RealHiTBench 默认选择：

```text
latex
```

从 `latex` 切到 `json_cells` 的条件：

- `structure_query = true` 且问题强调具体行列位置、层级表头、单元格定位
- `coordinate_query = true`
- 表格很稀疏，或 used range 很大但非空单元格较少
- LaTeX 不可用或估计 token 明显超预算

从 `latex` 切到 `json_rows` 的条件：

- `record_query = true`
- 问题更像数据库行检索/过滤，而不是表头层级理解
- 表格是规则矩形表，第一行/前几行表头清晰

从 `latex` 切到 `markdown` 的条件：

- 表格很小：行数 <= 20 且列数 <= 8
- 无合并单元格或复杂层级表头
- 问题是直接 lookup、简单比较、简单 fact checking

### Rule D: 大表 token 保护

如果候选文本格式估计 token 超过预算的 70%，先改选更紧凑的格式：

```text
if is_sparse: json_cells
elif task == manipulation: json_rows
elif latex available and not too long: latex
else: csv
```

如果所有文本格式都会严重截断，并且图像可用，选择对应任务的首选图像格式。这个规则只作为避免截断的兜底，不作为常规视觉偏好。

## 5. 打分实现

也可以把以上规则实现为可解释打分器。先给格式基础分，再按特征加减分，最后取最高分。

### 5.1 基础分

QA/CoT:

```python
base = {
    "latex": 3.0,
    "json_cells": 2.6,
    "markdown": 2.4,
    "json_rows": 2.0,
    "html": 1.7,
    "dataframe": 1.3,
    "csv": 1.2,
    "image": 0.8,
    "excel_1_image": 0.6,
    "default_image": 0.3,
}
```

Manipulation/PoT:

```python
base = {
    "json_cells": 4.0,
    "json_rows": 3.2,
    "markdown": 2.4,
    "latex": 2.3,
    "html": 2.0,
    "dataframe": 1.8,
    "csv": 1.6,
    "excel_1_image": 1.3,
    "image": 1.0,
    "default_image": 0.7,
}
```

### 5.2 特征加分

通用：

```python
if style_query:
    image_like += 4.0
    text_like -= 1.0
if has_style_signal and not numeric_query:
    image_like += 1.5
if is_sparse:
    json_cells += 1.5
    json_rows -= 0.5
if is_large:
    json_cells += 0.8
    markdown -= 1.0
    html -= 0.8
    dataframe -= 0.8
```

QA/CoT：

```python
if numeric_query:
    latex += 1.0
    json_cells += 0.6
if structure_query:
    json_cells += 1.2
    latex += 0.8
    markdown -= 0.3
if coordinate_query:
    json_cells += 1.5
if record_query:
    json_rows += 1.2
    markdown += 0.5
if small_simple_table:
    markdown += 1.0
```

Manipulation/PoT：

```python
if manipulation_type == "Cell-Level Manipulation":
    json_cells += 1.5
if manipulation_type == "Sheet-Level Manipulation":
    json_cells += 0.8
    json_rows += 0.8
if coordinate_query:
    json_cells += 1.5
if record_query and density >= 0.35 and not style_query:
    json_rows += 1.5
if n_sheets >= 2:
    json_cells += 1.0
if instruction_mentions_format_reference:
    excel_1_image += 2.5
```

### 5.3 图像分数映射

不要让所有 image-like 格式一起无差别上升，应按数据集映射：

```python
if dataset == "realhitbench":
    image += image_bonus
    excel_1_image += image_bonus * 0.8
    default_image += image_bonus * 0.5
else:
    excel_1_image += image_bonus
    image += image_bonus * 0.8
    default_image += image_bonus * 0.5
```

### 5.4 Tie-break

同分时按候选优先级排序：

```text
QA: latex > json_cells > markdown > json_rows > html > dataframe > csv > image > excel_1_image > default_image
Manipulation: json_cells > json_rows > markdown > latex > html > dataframe > csv > excel_1_image > image > default_image
```

如果最高格式不可用，选择下一个可用格式。

## 6. 推荐接口

```python
def route_table_format(
    dataset: str,
    task_mode: str,          # "qa_cot" or "manipulation_pot"
    query: str,
    xlsx_path: str,
    available_formats: set[str],
    metadata: dict | None = None,
) -> dict:
    """
    Returns:
        {
            "format": "json_cells",
            "score": 6.3,
            "reason": [
                "Manipulation/PoT default prefers json_cells.",
                "Detected coordinate_query and Cell-Level Manipulation.",
                "No style_query; image not needed."
            ],
            "features": {...},
            "scores": {...}
        }
    """
```

实现时必须返回 `reason`，方便后续分析路由是否符合预期。

## 7. 预期错误与诊断

- 图像误触发：如果只因为“format”一词就选图像，可能会损失大量 PoT 代码可执行性。需要区分 output format/answer format 和 spreadsheet visual format。
- JSON 过度触发：如果 QA 的普通数值题全部走 JSON，可能损失 `latex` 对复杂表头的表达优势。QA 默认仍应是 `latex`。
- `json_rows` 误用于稀疏表：行向 JSON 会保留大量空字符串，干扰代码生成；稀疏表优先 `json_cells`。
- `default_image` 误优先：它是兜底渲染，不是优先视觉源。
- 大表截断：如果路由选中会被截断的文本格式，应该在调用模型前重新路由，而不是静默截断。

## 8. 最小消融建议

后续实现后，先比较三条线：

- `best_single_default`：QA 固定 `latex`，Manipulation 固定 `json_cells`。
- `heuristic_router`：本文规则。
- `oracle`：任一格式正确即正确。

重点看 router 相对 default 的提升，以及它靠近 oracle 的比例。另需输出每个格式被选择的次数，防止路由坍缩到单一格式。
