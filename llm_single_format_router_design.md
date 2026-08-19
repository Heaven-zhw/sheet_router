# LLM 单格式表格表示路由设计

目标：对每个样本只选择一种表格输入格式，不使用文本+图像混合输入；路由只依赖表格文件和查询，不依赖下游求解模型名称、大小或历史表现。

适用任务：

- RealHiTBench：QA / CoT。
- SpreadsheetBench：Manipulation / PoT。

## 1. 核心想法

让一个轻量 LLM-router 先阅读“查询 + 表格摘要”，只判断当前样本最需要哪类表格信息；程序再根据这些需求标签、表格统计和格式先验映射到一个单一格式。

它不直接回答问题，也不生成操作代码。它只判断：

- 任务是否需要视觉样式信息。
- 是否需要精确单元格坐标。
- 是否更像按行记录处理。
- 是否依赖复杂表头、合并单元格或层级结构。
- 文本表示是否可能过长或过稀疏。

然后程序用固定映射和后处理规则检查格式是否可用、是否会截断、是否误触发图像。

## 2. 统计先验

现有 Qwen/Gemma 统计给出三个稳定结论。

1. RealHiTBench 中，oracle 明显高于单一格式，说明“按样本选格式”有空间。跨 Qwen/Gemma 平均看，`latex` 与 `json_cells` 是最稳的文本候选。
2. SpreadsheetBench 中，`json_cells` 是最稳默认格式：在已有 Qwen/Gemma 统计里平均 hard_all 最高，且每个模型上都是第一。
3. 图像格式有独占正确样例，但整体不适合作为默认。只有查询明确需要颜色、边框、合并视觉布局、格式参考、截图对象等信息时才优先考虑图像。

聚合数字：

```text
RealHiTBench / EM:
  oracle avg = 82.16
  latex default avg = 64.24
  oracle gap = +17.92
  only-one-format-correct avg = 5.58

SpreadsheetBench / hard_all:
  oracle avg = 63.05
  json_cells default avg = 40.30
  oracle gap = +22.75
  only-one-format-correct avg = 10.45
```

推荐默认：

```text
RealHiTBench / QA: latex
SpreadsheetBench / Manipulation: json_cells
```

最终偏离默认时必须在 `evidence` 和格式分数中留下明确原因。

## 3. 候选格式

当前只考虑单格式：

```text
markdown
latex
html
csv
dataframe
json_rows
json_cells
image
excel_1_image
default_image
```

不要选择 `structure`，因为当前 RealHiTBench 中结构文件覆盖不足。不要选择任何 `text+image` 混合格式。

## 4. 格式能力卡

下面内容作为程序侧格式映射依据。当前实现不要求 LLM 直接选择格式。

```text
latex:
  good_for: complex headers, merged headers, compact QA reasoning, table hierarchy
  weak_for: exact sparse coordinate tracing, very large tables

json_cells:
  good_for: exact cell addresses, sparse tables, multi-sheet workbooks, PoT code generation, non-empty-cell tracing
  weak_for: row-wise record semantics if headers must be inferred from dense rows

json_rows:
  good_for: row filtering, matching, grouping, sorting, aggregation, database-like records
  weak_for: sparse sheets, many empty cells, precise scattered coordinate edits

markdown:
  good_for: small simple rectangular tables, direct lookup, simple comparison
  weak_for: large/wide tables, complex merged headers

html:
  good_for: preserving table cell order with explicit table tags, occasional layout cues
  weak_for: token cost, noisy reasoning

csv:
  good_for: compact dense numeric tables
  weak_for: merged headers, styles, formulas, complex layouts

dataframe:
  good_for: visual fixed-width text for small/medium regular tables
  weak_for: long/wide tables and precise coordinates

image:
  good_for: original visual evidence when available, charts, colors, borders, screenshots, visual layout
  weak_for: code generation and exact value extraction on long tables

excel_1_image:
  good_for: Excel-style visual rendering, style/layout reference, formatting tasks
  weak_for: long-table reasoning and exact computation

default_image:
  good_for: last-resort screenshot when no better image is available
  weak_for: should not be preferred over image/excel_1_image
```

## 5. 路由输入摘要

不要把完整大表直接丢给 router。先用程序从 xlsx 和数据集字段抽一个短摘要。

建议 schema：

```json
{
  "dataset": "realhitbench | spreadsheetbench",
  "task_mode": "qa_cot | manipulation_pot",
  "query": "...",
  "instruction_type": "Cell-Level Manipulation | Sheet-Level Manipulation | null",
  "answer_position": "A1:B5 | null",
  "answer_sheet": "Sheet1 | null",
  "available_formats": ["latex", "json_cells", "..."],
  "token_budget": 100000,
  "workbook_profile": {
    "num_sheets": 1,
    "sheets": [
      {
        "name": "Sheet1",
        "used_range": "A1:H40",
        "rows": 40,
        "cols": 8,
        "non_empty_cells": 210,
        "density": 0.66,
        "merged_ranges": ["A1:H1"],
        "has_style_signal": true,
        "style_signal_summary": ["fill_color", "bold_font", "border"],
        "header_preview": ["Year", "Agriculture", "Manufacturing", "..."],
        "first_rows_preview": [
          ["1950", "7123", "..."],
          ["1951", "6901", "..."]
        ]
      }
    ],
    "estimated_tokens": {
      "latex": 3200,
      "markdown": 5100,
      "json_rows": 6200,
      "json_cells": 2800,
      "html": 7900,
      "csv": 4100,
      "dataframe": 5600
    }
  }
}
```

摘要中必须包含表头/前几行预览，因为 router 需要判断查询词是否对应列名、行名、sheet 名或视觉样式。

## 6. Router Prompt

System prompt：

```text
You are a model-independent spreadsheet representation router.

Your job is to tag what evidence a downstream spreadsheet solver needs.
Do not choose the table format directly. Do not answer the spreadsheet question. Do not write code.

The tags must depend only on the spreadsheet file summary and the user query.
Never use the downstream solver model name, model family, or per-model performance.

Be conservative about visual/style tags: words like "answer format" or "output format" do not mean spreadsheet visual style.
Return only valid JSON.
```

User prompt：

```text
# Task
Tag the evidence needs for this spreadsheet task.

# Synthetic Examples
Example A:
Query: "Which product has the largest Q3 margin under the Retail group?"
Workbook clue: merged group headers and quarter subcolumns.
Output tags: complex_headers=true, numeric_reasoning=true, exact_coordinates=false, row_records=false, visual_style=false.

Example B:
Query: "Filter orders after 2024-03-01 by customer ID, sort them by date, and add a subtotal row."
Workbook clue: dense table with clear row records.
Output tags: row_records=true, numeric_reasoning=true, exact_coordinates=false, complex_headers=false, visual_style=false.

Example C:
Query: "Fill the target block using the sample block as a formatting reference and keep the same borders and colors."
Workbook clue: styled cells and target range.
Output tags: visual_style=true, format_reference=true, exact_coordinates=true.

Example D:
Query: "What value appears at cell F12?"
Workbook clue: sparse sheet with many blank cells.
Output tags: exact_coordinates=true, sparse_or_large=true.

# Input
{routing_input_json}

# Tag Meanings
- visual_style: the solver needs colors, fills, borders, bold/italic, row height, column width, charts, images, or screenshot layout.
- format_reference: the query asks to preserve/copy/use visual formatting from a sample range.
- exact_coordinates: exact cells, ranges, output positions, or sheet names are central.
- row_records: the task is like filtering, matching, grouping, sorting, deduplicating, appending rows, or aggregating by keys.
- complex_headers: merged headers, multi-level headers, nested sections, row/column header hierarchy, or cross-header lookup matter.
- sparse_or_large: the workbook is sparse, wide, long, or likely to be hard to serialize densely.
- multi_sheet: reasoning across multiple worksheets matters.
- numeric_reasoning: arithmetic, max/min, count, comparison, percentage, or totals matter.

# Output JSON Schema
{
  "needs": {
    "visual_style": false,
    "format_reference": false,
    "exact_coordinates": false,
    "row_records": false,
    "complex_headers": false,
    "sparse_or_large": false,
    "multi_sheet": false,
    "numeric_reasoning": false
  },
  "default_risk": "none | low | medium | high",
  "evidence": ["short evidence 1", "short evidence 2"]
}
```

## 7. 输出约束

Router 输出必须满足：

- `needs` 必须是 JSON object，所有标签解析为 boolean。
- `default_risk` 只能是 `none | low | medium | high`。
- `evidence` 只解释需求标签，不推理答案。
- LLM 不输出 `selected_format`；单一格式由程序侧映射得到。

## 8. 程序后处理

LLM-router 后面必须有 deterministic mapper 和 guardrails。

```python
def route_from_needs(llm_decision, routing_input):
    available = set(routing_input["available_formats"])
    task_mode = routing_input["task_mode"]
    dataset = routing_input["dataset"]

    default = "latex" if task_mode == "qa_cot" else "json_cells"
    needs = merge_llm_and_program_tags(llm_decision["needs"], routing_input)
    scores = base_scores(task_mode)

    if needs["visual_style"] or needs["format_reference"]:
        boost_image_scores(scores, dataset)
    if needs["exact_coordinates"]:
        scores["json_cells"] += 2.2
    if needs["row_records"]:
        scores["json_rows"] += 2.0
    if needs["complex_headers"]:
        scores["latex"] += 1.3
        scores["json_cells"] += 1.0
    if needs["sparse_or_large"] or needs["multi_sheet"]:
        scores["json_cells"] += 1.0

    fmt = highest_scoring_available_format(scores, available, task_mode)
    if fmt in {"image", "excel_1_image", "default_image"}:
        visual = needs["visual_style"] or needs["format_reference"]
        too_long = all_text_formats_too_long(routing_input)
        if not visual and not too_long:
            fmt = default if default in available else compact_text_fallback(task_mode, routing_input, available)

    if fmt == "default_image":
        fmt = first_available(image_priority(dataset), available)

    if will_truncate(fmt, routing_input):
        fmt = compact_text_fallback(task_mode, routing_input, available)

    return fmt
```

Image priority：

```text
RealHiTBench: image > excel_1_image > default_image
SpreadsheetBench: excel_1_image > image > default_image
```

## 9. 典型选择规则

这些规则是程序侧把 `needs` 标签映射到格式时的主要检查项。

选择 `latex`：

- QA 任务。
- 查询需要理解层级表头、合并表头、跨列/跨行结构。
- 表格中等大小，LaTeX 不会明显截断。

选择 `json_cells`：

- Manipulation/PoT 默认。
- 需要定位具体单元格、range、sheet；SpreadsheetBench 中 Cell-Level 的 `answer_position` 是强坐标信号。
- 表格稀疏、多 sheet、合并单元格多，或只需要非空单元格。
- QA 中问题强调坐标、上下左右位置、某行某列交叉点。

选择 `json_rows`：

- 操作像数据库任务：match、filter、sort、group、deduplicate、sum by、lookup by key、append rows。
- 表格密度较高，行记录结构清楚。
- 不强依赖样式、零散坐标或复杂合并区域。

选择 `markdown`：

- 小型规则矩形表。
- 简单 lookup、比较、计数。
- 无明显复杂表头或样式依赖。

选择图像格式：

- 查询要求颜色、背景、边框、加粗、缩进、图表、图片、截图可见位置。
- Manipulation 指令要求“参考已有格式”“保持原格式”“按样式填充”。
- 文本序列化会丢失关键视觉布局。

## 10. 实现接口

当前实现模块：

```text
single_resp_llm_router.py
```

核心接口：

```python
def route_table_format(
    *,
    dataset: str,
    task_mode: str,
    query: str,
    xlsx_path: str,
    available_formats: set[str],
    instruction_type: str | None = None,
    answer_position: str | None = None,
    answer_sheet: str | None = None,
    token_budget: int | None = None,
    router_model: str | None = None,
) -> dict:
	    """
	    Returns:
	        {
	            "selected_format": "json_cells",
	            "llm_needs": {...},
	            "program_needs": {...},
	            "scores": {...},
	            "reason": [...]
	        }
	    """
```

Solver 集成方式：

- RealHiTBench：在 `RealHiTCoTSolver.build_prompt()` 调用 `TableInputBuilder` 前决定 `table_format`。
- SpreadsheetBench：在 `SpreadSheetPoTSolver.build_prompt()` 调用 `SpreadsheetTableInputBuilder` 前决定 `table_format`。
- 每个样本保存 `router_decision` 到结果 JSONL，方便后续分析路由失败原因。

## 11. 最小评测

先跑三条线：

```text
default_single:
  RealHiTBench = latex
  SpreadsheetBench = json_cells

llm_router_single:
  本文方法，每个样本选一个格式

oracle:
  任一格式正确即正确
```

记录：

- 主指标：RealHiTBench 用 EM，SpreadsheetBench 用 hard_all。
- router 相对 default_single 的提升。
- router 达到 oracle gap 的比例。
- 每种格式被选择次数，避免坍缩到默认格式。
- LLM `needs` 标签分布，以及最终格式分布。
- 图像格式触发样例中，真正因样式/视觉触发的比例。

## 12. 失败诊断

需要重点看五类错误：

1. 图像误触发：查询里的 “format” 只是输出格式，不是表格视觉格式。
2. `json_rows` 误用于稀疏表：行 JSON 被大量空值污染。
3. QA 过度切到 JSON：失去 LaTeX 对复杂表头的表达优势。
4. Manipulation 过度切到图像：PoT 代码缺少可解析的单元格和值。
5. 文本截断：映射器选了理论合适但实际超预算的格式。

## 13. 推荐第一版策略

第一版不要训练分类器，直接实现：

```text
xlsx profiler + query/task metadata
-> LLM-router JSON evidence tags
-> calibrated format mapper + deterministic guardrails
-> selected single table_format
-> 原 CoT/PoT solver
```

这样贡献点清楚：不是为某个模型调参，而是用表格/查询本身推断“当前任务最需要的信息类型”，再映射到最合适的输入表示。

## References

- FLEXTAF: Flexible Table Reasoning for Textual and Visual Modalities, https://arxiv.org/pdf/2408.08841
- SpreadsheetAgent environment paper, https://arxiv.org/pdf/2604.12282
