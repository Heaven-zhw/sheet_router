# dynamic_router.py 代码说明

本文档说明 SheetRouter 当前实现的特征提取、路由策略、fallback 与输出格式。当前版本是第一版可运行的 rule-based SheetRouter，目标是优先跑通 RealHiTBench 和 SpreadsheetBench 的动态路由实验，为后续接入 LLM router、学习型 router、区域裁剪、结构摘要等模块预留接口。

代码已按最小职责拆分：

```text
dynamic_router.py:
  统一入口，只负责读数据、调用 profiler/router/solver/verifier、fallback、保存结果

core/routing/profile.py:
  workbook 静态特征、query/instruction 文本特征、可用表示检测

core/routing/router.py:
  RouteDecision、heuristic、blackbox_heuristic、router_profile 输出清理
```

## 1. 总体流程

当前 `dynamic_router.py` 的执行流程如下：

```text
读取样本
  -> SpreadsheetProfiler 提取 workbook/query 静态特征
  -> HeuristicRouter 基于规则选择 route
  -> 调用已有 solver
       RealHiTBench: RealHiTCoTSolver
       SpreadsheetBench: SpreadSheetPoTSolver
  -> ObservableVerifier 检查可观察失败
  -> 必要时尝试 fallback route
  -> 保存结果与 router_decisions.jsonl
```

其中 route 目前主要包含：

```text
solver_mode + table_format + fallback_formats
```

例如：

```json
{
  "solver_mode": "cot_qa",
  "table_format": "latex+image",
  "fallback_formats": ["latex", "markdown+image", "markdown"]
}
```

或：

```json
{
  "solver_mode": "pot_code",
  "table_format": "markdown+excel_1_image",
  "fallback_formats": ["markdown", "markdown+default_image"]
}
```

## 2. Workbook 静态特征

Workbook 特征由 `profile_workbook()` 提取，主要使用 `openpyxl` 直接读取 Excel 文件，不调用大模型。

### 2.1 每个 sheet 的特征

对每个 worksheet，当前保存到 `sheet_summaries`：

```text
sheet_name: 工作表名称
used_range: 非空区域范围，如 A1:K81
used_rows: 使用区域行数
used_cols: 使用区域列数
used_cells: used_rows * used_cols
nonempty_cells: 非空单元格数量
merged_ranges: 合并单元格区域数量
formula_cells: 公式单元格数量
distinct_fill_colors: 背景填充颜色种类数
bold_cells: 加粗单元格数量
bordered_cells: 有可见边框的单元格数量
hidden_rows: 隐藏行数量
hidden_cols: 隐藏列数量
charts: 图表数量
embedded_images: 嵌入图片数量
```

### 2.2 workbook 汇总特征

在所有 sheet 上进一步汇总：

```text
num_sheets: 工作表数量
sheet_names: 工作表名称列表
max_rows / max_cols: 最大 sheet 尺寸
max_sheet_cells: 最大 sheet 的 used_cells
total_used_cells: 所有 sheet 的 used_cells 总和
total_nonempty_cells: 所有 sheet 的非空单元格总数
nonempty_ratio: total_nonempty_cells / total_used_cells
total_text_chars: 非空单元格文本长度总和
estimated_text_tokens: 根据字符数粗略估算 token 数，约 len / 3.5
num_merged_ranges: 合并区域总数
merged_cell_signal: 是否存在合并单元格
num_formulas: 公式单元格总数
formula_ratio: 公式单元格占非空单元格比例
num_distinct_fill_colors: 背景色种类数
has_background_color: 是否存在明显背景色
num_bold_cells: 加粗单元格总数
num_bordered_cells: 有边框单元格总数
has_hidden_rows_or_cols: 是否存在隐藏行列
hidden_rows / hidden_cols: 隐藏行列数量
has_charts_or_images: 是否存在图表或嵌入图片
num_charts / num_embedded_images: 图表和嵌入图片数量
numeric_cell_ratio: 数值单元格比例
date_cell_ratio: 日期或疑似日期单元格比例
long_text_cell_ratio: 长文本单元格比例
```

这些特征对应研究设想中的低成本 workbook profile，用来粗略判断：

```text
表格规模
结构复杂度
样式/视觉信号
公式密度
多 sheet 风险
文本截断风险
```

## 3. Query / Instruction 文本特征

文本特征由 `query_features()` 提取。当前是 keyword/rule-based，不调用模型。

### 3.1 基础长度特征

```text
query_chars: 问题或指令字符数
query_tokens_est: 粗略 token 估计
```

### 3.2 语义触发特征

当前通过英文关键词触发以下布尔特征：

```text
mentions_color_or_style:
  是否提到 color, background, fill, font, bold, border, highlight, style, format 等视觉/样式词

mentions_header_or_structure:
  是否提到 header, merged, hierarchy, category, section, layout, structure 等结构词

mentions_formula:
  是否提到 formula, equation, computed

mentions_sort_filter_pivot:
  是否提到 sort, filter, pivot, rank, ascending, descending, highest, lowest, order 等操作词

mentions_insert_delete:
  是否提到 insert, delete, remove, clear, create, add row, add column 等编辑词

mentions_cross_sheet:
  是否提到 sheet/worksheet，或 match, lookup, duplicate, merge, combine, join, cross sheet 等跨表/匹配词

mentions_total_sum_average_rank:
  是否提到 sum, total, average, mean, median, max, min, count, calculate 等聚合词

mentions_date_time:
  是否提到 date, month, year, day, time

mentions_exact_cell_or_range:
  是否出现类似 A1、B2:C10 的单元格或区域引用

mentions_sheet_name:
  是否显式提到 sheet/worksheet，或出现 'Sheet1'!A1 这类格式
```

### 3.3 任务需求派生特征

```text
num_operations_in_instruction:
  基于逗号、and 和操作关键词粗略估计指令复杂度

requires_lookup:
  是否可能需要查找、匹配或 join

requires_aggregation:
  是否可能需要求和、平均、计数等聚合

requires_comparison:
  是否出现 highest, lowest, largest, smallest, greater, less 等比较需求

requires_formatting:
  是否可能需要格式/样式信息
```

注意：当前关键词主要是英文，因为两个目标 benchmark 的题目/指令基本是英文。后续如果接入中文任务，需要扩展这些 keyword 列表，或改成 LLM/classifier router。

## 4. 数据集级 profile 补充

`SpreadsheetProfiler` 会在通用 workbook profile 之外，为不同数据集加入额外字段。

### 4.1 RealHiTBench

从样本字段补充：

```text
dataset = realhitbench
sample_id
file_name
question_type: Fact Checking / Numerical Reasoning / Structure Comprehending
sub_question_type
complex_structure: CompStrucCata
question_features
latex_available
```

同时检测可用表示：

```text
available_text_formats:
  latex, markdown, html, csv, tsv, dataframe, json_rows, json_cells
  如果 latex 文件不存在，则不会优先加入 latex

available_image_formats:
  image: 数据集原始图片
  excel_1_image: 预生成 Excel 风格图片
  default_image: 可以运行时从 xlsx 渲染

image_counts:
  每种图片表示可用的图片数量
```

此外，如果 `estimated_text_tokens > max_text_tokens`，设置：

```text
truncation_risk = true
```

### 4.2 SpreadsheetBench

从样本字段补充：

```text
dataset = spreadsheetbench
sample_id
input_file
instruction_type: Cell-Level Manipulation / Sheet-Level Manipulation
answer_position
answer_sheet
question_features
```

检测可用表示：

```text
available_text_formats:
  markdown, html, csv, tsv, dataframe, json_rows, json_cells, latex

available_image_formats:
  image
  excel_1_image
  default_image
```

额外判断：

```text
truncation_risk:
  estimated_text_tokens > max_text_tokens

multi_image_risk:
  num_sheets >= 3
```

## 5. 当前路由策略

当前支持两套可独立运行的规则路由器：

```text
heuristic:
  第一版 oracle-leaning heuristic。
  会使用 benchmark 自带的任务类型字段，例如 RealHiTBench 的 QuestionType、CompStrucCata，
  以及 SpreadsheetBench 的 instruction_type、answer_position 等。
  适合作为有标签分析 baseline，不适合作为真实黑盒样例的通用路由规则。

blackbox_heuristic:
  新增的黑盒启发式路由器。
  只假设真实运行时知道：
  - 任务族：QA 或 spreadsheet operation
  - 用户自然语言问题/指令
  - 表格文件路径
  - 从 xlsx 直接统计出的 workbook profile
  - 当前样例可用哪些输入表示
  不使用 benchmark 标签、不使用答案位置、不调用 LLM router。
```

可以通过参数选择：

```bash
--router_policy heuristic
--router_policy blackbox_heuristic
```

### 5.1 表示选择优先级

`heuristic` 的文本表示选择：

```text
RealHiTBench 默认优先 latex
SpreadsheetBench 默认强制 markdown
```

`blackbox_heuristic` 的文本表示选择：

```text
RealHiTBench 默认优先 markdown
SpreadsheetBench 默认强制 markdown
```

图片表示选择：

```text
RealHiTBench 默认顺序:
  image -> excel_1_image -> default_image

SpreadsheetBench 默认顺序:
  excel_1_image -> image -> default_image
```

可以通过参数覆盖：

```bash
--qa_text_format latex|markdown|html|auto
--operation_text_format markdown|html|latex
--image_preference auto|image|excel_1_image|default_image|none
```

## 6. heuristic 路由规则

这一节说明旧的 `heuristic`。它会利用 benchmark 元信息，因此更适合做分析 baseline。

### 6.1 RealHiTBench

RealHiTBench 固定使用：

```text
solver_mode = cot_qa
```

即调用 `RealHiTCoTSolver` 直接回答问题。

#### 6.1.1 内部信号

路由时会构造以下信号：

```text
large_table:
  total_nonempty_cells >= large_cell_threshold
  默认阈值 1500

small_table:
  total_nonempty_cells <= small_cell_threshold
  默认阈值 400

structure_signal:
  满足以下任一条件：
  - QuestionType == Structure Comprehending
  - query 提到 header/merged/structure 等结构词
  - workbook 存在 merged_cell_signal
  - CompStrucCata 属于:
      ColumnHeaderMerge
      MultiColumnClassified
      SingleRowClassified
      ContentCompound
      StructureCompound

style_signal:
  满足以下任一条件：
  - query 提到样式/颜色词
  - CompStrucCata == BackgroundColor
  - workbook 有背景色

reasoning_signal:
  满足以下任一条件：
  - QuestionType == Numerical Reasoning
  - query 需要 aggregation
  - SubQType 中包含 Reasoning

retrieval_like:
  QuestionType == Fact Checking
  且没有 reasoning_signal、structure_signal、style_signal
```

#### 6.1.2 主 route 选择规则

规则顺序如下：

```text
1. 如果有 text+image，且 structure_signal 或 style_signal 为 true:
     选择 text+image

2. 否则，如果有 text+image，且 reasoning_signal 为 true，
   且表格不是 large table:
     选择 text+image

3. 否则，如果有 text+image，且不是 retrieval_like:
     选择 text+image

4. 否则:
     选择 text only
```

直观解释：

```text
结构理解、样式理解、非大表数值推理优先使用文本+图片；
大表或简单 fact-checking 更倾向保守使用文本，避免图像和冗余上下文干扰。
```

#### 6.1.3 fallback route

RealHiTBench 当前 fallback 候选：

```text
如果 primary 是 text+image:
  fallback 到 text only

如果 primary 是 text only，且 text+image 可用:
  fallback 到 text+image

如果首选文本不是 markdown 且 image 可用:
  额外加入 markdown+image 和 markdown
```

举例：

```text
primary = latex+image
fallback = latex, markdown+image, markdown
```

实际最多尝试多少个 fallback 由参数控制：

```bash
--fallback / --no-fallback
--max_fallbacks
```

默认 `--max_fallbacks 1`，即最多在 primary 后再试 1 条 fallback route。

### 6.2 SpreadsheetBench

SpreadsheetBench 固定使用：

```text
solver_mode = pot_code
```

即调用 `SpreadSheetPoTSolver` 生成 Python 代码、执行并输出修改后的 Excel 文件。

#### 6.2.1 内部信号

```text
style_signal:
  query 提到样式/颜色/格式，或 requires_formatting 为 true

workbook_style:
  workbook 有背景色，或有边框单元格

truncation_risk:
  estimated_text_tokens > max_text_tokens

multi_image_risk:
  num_sheets >= 3

exact_cell_signal:
  query 出现单元格/区域引用，或样本有 answer_position
```

#### 6.2.2 主 route 选择规则

规则顺序如下：

```text
1. 如果 text+image 可用，且 style_signal 和 workbook_style 都为 true:
     选择 markdown+image

2. 否则，如果 text+image 可用，且存在 truncation_risk，
   且不是 multi_image_risk:
     选择 markdown+image

3. 否则:
     选择 markdown
```

直观解释：

```text
SpreadsheetBench 是代码生成和文件修改任务。
默认优先 markdown，因为它对坐标、值、代码生成最稳定。
只有当任务明显需要视觉/样式信息，或文本太长且图片数量风险不高时，才加图片。
不会把 image-only 作为主 route。
```

#### 6.2.3 fallback route

SpreadsheetBench 当前 fallback 候选：

```text
如果 primary 是 markdown+image:
  fallback 到 markdown

如果 primary 是 markdown，且 markdown+image 可用:
  fallback 到 markdown+image

如果 default_image 可用:
  额外加入 markdown+default_image
```

举例：

```text
primary = markdown+excel_1_image
fallback = markdown, markdown+default_image
```

## 7. blackbox_heuristic 路由规则

`blackbox_heuristic` 是根据输入格式实验规律新增的通用规则。设计原则是：

```text
RealHiTBench / QA:
  文本+图像整体收益稳定，因此默认 markdown+image。
  只有在“大表 + 简单检索型问题”这类图像可能稀释上下文的场景，才退回 markdown。

SpreadsheetBench / operation:
  PoT 代码生成依赖坐标、单元格值和可解析文本，因此默认 markdown。
  只有强视觉/布局需求、文本截断风险低且图片风险可控时，才加 excel_1_image。
```

### 7.1 不使用的字段

该策略不使用以下 benchmark-only 信息：

```text
RealHiTBench:
  QuestionType
  SubQType
  CompStrucCata

SpreadsheetBench:
  instruction_type
  answer_position
  answer_sheet
```

输出到 `router_profile` 时也会对这些字段做过滤，并在 `router_decision.stages.blackbox_constraints` 中记录：

```json
{
  "uses_benchmark_labels": false,
  "uses_answer_position": false,
  "uses_answer_sheet": false,
  "uses_llm_router": false
}
```

### 7.2 QA 特征与打分

QA 场景重新从自然语言问题中提取 `query_features(question, "qa")`，同时结合 workbook profile 构造三个分数。

```text
structure_need_score:
  +2 query 提到 header/merged/layout/structure 等结构词
  +2 workbook 存在合并单元格
  +1 合并区域数量 >= 3
  +1 sheet 数量 >= 2
  +1 非空密度较低 nonempty_ratio < 0.55
  +1 表格同时满足 max_cols >= 10 且 max_rows >= 20

visual_need_score:
  +3 query 提到颜色/样式/格式词
  +2 workbook 有背景色
  +2 workbook 有图表或嵌入图片
  +1 workbook 有边框单元格

reasoning_need_score:
  +2 聚合需求
  +1 比较需求
  +1 lookup/match/join 需求
  +1 sort/filter/pivot/rank 需求
  +1 日期时间需求
  +0~2 多操作复杂度
```

另外定义：

```text
retrieval_like:
  没有聚合、lookup、比较需求
  structure_need_score <= 1
  visual_need_score == 0
  query_tokens_est <= 50

large_table:
  total_nonempty_cells >= large_cell_threshold
```

### 7.3 QA route 选择

图片优先级：

```text
image -> excel_1_image -> default_image
```

主 route：

```text
如果有图片，且不是 large_table + retrieval_like:
  primary = markdown+image
否则:
  primary = markdown
```

fallback 候选：

```text
markdown
latex+同类图片 / latex
markdown+excel_1_image
markdown+default_image
```

这对应 Qwen3.5-9B 实验中 RealHiTBench 的主要规律：`markdown+image` 整体最强，结构/样式类任务尤其受益；但黑盒规则不能直接读取 `Structure Comprehending` 或 `BackgroundColor`，因此改用 query 关键词和 workbook 结构/样式统计近似判断。

### 7.4 Spreadsheet operation 特征与打分

operation 场景重新从自然语言指令中提取 `query_features(instruction, "operation")`，构造四类信号。

```text
visual_need_score:
  +3 query 提到颜色/样式/格式词
  +2 workbook 有背景色
  +2 workbook 有图表或嵌入图片
  +1 workbook 有边框单元格

layout_need_score:
  +2 query 提到 header/merged/layout/structure 等结构词
  +2 query 提到 sheet/worksheet/cross-sheet
  +1 lookup/match/join 需求
  +1 sort/filter/pivot/rank 需求
  +1 insert/delete/remove/create 需求
  +1 workbook 存在合并单元格
  +1 sheet 数量 >= 2
  +1 非空密度较低

operation_complexity_score:
  +2 lookup/match/join 需求
  +2 聚合需求
  +1 sort/filter/pivot/rank 需求
  +1 insert/delete/remove/create 需求
  +0~3 多操作复杂度

image_risk_score:
  +2 sheet 数量 >= 3
  +1 当前图片数量 >= 3
  +1 large_table
  +1 只有 default_image
```

### 7.5 Spreadsheet operation route 选择

图片优先级：

```text
excel_1_image -> image -> default_image
```

主 route：

```text
默认:
  primary = markdown

如果 excel_1_image 可用，且满足任一条件:
  visual_need_score >= 4
  或 layout_need_score >= 5 且 operation_complexity_score >= 3 且 image_risk_score <= 3
  或 truncation_risk 且 image_risk_score <= 2
则:
  primary = markdown+excel_1_image

如果没有 excel_1_image，但存在非 default 图片，
且 visual_need_score >= 5 且 image_risk_score <= 2:
  primary = markdown+image

如果只有 default_image，
且 visual_need_score >= 6 且 image_risk_score <= 1:
  primary = markdown+default_image
```

fallback 候选：

```text
如果 primary 是 markdown+image:
  fallback 到 markdown

如果 primary 是 markdown:
  fallback 到 markdown+excel_1_image / markdown+image / markdown+default_image

始终谨慎加入 markdown+default_image 作为后备，而不是主 route
```

这对应 SpreadsheetBench 实验规律：`markdown` 在整体 hard_all 上最稳定，图像加入后可能提升部分 sheet-level/布局任务，但会降低代码执行稳定性。因此黑盒策略默认保守，只在强视觉/布局证据足够时使用 `markdown+excel_1_image`。

## 8. ObservableVerifier 与 fallback 触发

当前 verifier 只使用可观察失败信号，不使用 gold answer 或 evaluator 的语义正确性作为重路由依据。

### 8.1 RealHiTBench

失败条件：

```text
format_valid == false
model_answer 为空
```

不会因为 EM/F1 低而 fallback，因为那属于 gold/evaluator 信号。

### 8.2 SpreadsheetBench

失败条件：

```text
format_valid == false
execution_success == false
solution 为空
```

不会因为 `total_hard_restriction == 0` 自动 fallback，除非对应失败能被执行错误、未生成文件、空代码等可观察信号解释。

## 9. 输出文件

完整运行结束后只保留 canonical 结果文件和路由决策文件。

### 9.1 RealHiTBench

```text
realhit_cot.jsonl
realhit_cot_eval.json
realhit_cot_score.json
router_decisions.jsonl
```

### 9.2 SpreadsheetBench

```text
spreadsheet_pot.jsonl
spreadsheet_pot_eval.json
spreadsheet_pot_accuracy.json
router_decisions.jsonl
```

运行过程中会定期写入 partial 文件。完整运行结束后会删除：

```text
realhit_cot.partial.jsonl
realhit_cot_eval.partial.json
realhit_cot_score.partial.json
spreadsheet_pot.partial.jsonl
spreadsheet_pot_eval.partial.json
spreadsheet_pot_accuracy.partial.json
```

同时也会清理旧版本遗留的 `dynamic_router*` 文件。

## 10. 每条结果中的路由字段

每个样本结果会附加：

```text
router_policy:
  当前路由器名称，例如 heuristic 或 blackbox_heuristic

router_decision:
  最终路由决策，包括 solver_mode、table_format、fallback_formats、reason、stages

router_profile:
  当前样本的 workbook/query 静态特征

router_attempts:
  实际尝试过的 route，包括 primary 和 fallback

router_verifier:
  最终 route 的可观察验证结果

router_fallback_used:
  是否实际使用了 fallback

router_failed_attempt_results:
  如果开启 save_all_route_attempts，保存失败 route 的完整结果
```

`router_decisions.jsonl` 是精简版，只保存样本 id、router decision、verifier 和是否使用 fallback，便于快速统计 route 分布。

## 11. 当前实现的局限

当前版本是为了先跑通实验，因此有以下局限：

```text
1. 路由器是 heuristic，不是训练出来的 classifier/ranker。
2. query keyword 主要覆盖英文，中文任务需要扩展关键词或接入 LLM router。
3. 当前没有做 sheet selection、region crop、公式依赖图、结构摘要接入。
4. token 数是粗略估计，真正 prompt token 数以 solver build_prompt 后的 metadata 为准。
5. fallback 只基于可观察失败，不基于 evaluator/gold 结果做 oracle reroute。
6. SpreadsheetBench 当前不选择 image-only route，因为代码生成稳定性通常依赖文本坐标和值。
```

## 12. 后续可扩展方向

建议后续按以下顺序扩展：

```text
1. 加入结构摘要 route:
   markdown+structure, structure_only

2. 加入局部证据 route:
   selected_sheet_markdown, region_crop, selected_sheet_image

3. 加入 LLM prompt router:
   输入 router_profile + query，输出 route JSON

4. 加入学习型 router:
   使用已有多表示实验结果训练 route scorer

5. 加入成本统计:
   text_tokens, image_count, image_pixels, latency proxy

6. 加入 route 分析脚本:
   按 question_type、CompStrucCata、sheet_count、style_signal 统计 route 分布和性能
```
