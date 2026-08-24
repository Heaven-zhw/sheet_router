# 表格输入格式说明

本文档说明项目通过 `table_format` 向模型提供电子表格内容时支持的格式、数据来源和主要特点。

## 数据集支持总览

| 格式类别 | RealHiTBench | SpreadsheetBench | 说明 |
| --- | --- | --- | --- |
| 官方文本 | `official_latex` | 不支持 | 直接读取 RealHiTBench 官方提供的 LaTeX 文本文件 |
| 动态文本 | `latex`、`csv`、`tsv`、`markdown`、`dataframe`、`json_rows`、`json_cells`、`html` | 同左 | 运行时从 XLSX 的有效数据区域生成 |
| 纯图片 | `image`、`excel_1_image`、`default_image` | 同左 | 使用数据集图片或运行时渲染的工作表截图 |
| 文本与图片组合 | `official_latex`、`latex`、`markdown`、`html` 可以与三种图片格式组合 | `latex`、`markdown`、`html` 可以与三种图片格式组合 | 同时向多模态模型提供文本序列化和工作表截图 |

## 文本格式

| 格式名称 | 数据来源 | 内容结构 | 主要特点 | 适用场景与限制 |
| --- | --- | --- | --- | --- |
| `official_latex` | RealHiTBench 官方 `.txt` 文件 | 官方预生成的 LaTeX 表格文本 | 保留官方基准使用的表格表示；与历史 RealHiTBench `latex` 实验含义一致 | 仅 RealHiTBench 支持；不受运行时坐标和合并单元格参数控制 |
| `latex` | XLSX | 每个工作表包含名称、有效区域和 `tabular` 内容 | 两个数据集使用相同转换逻辑；合并单元格使用 `\multirow` 和 `\multicolumn` 表示；LaTeX 特殊字符会被转义 | 适合复杂表头和合并结构；不保留颜色；不受 `include_coordinates` 和 `fill_merged` 控制 |
| `csv` | XLSX | 逗号分隔的二维矩阵 | 简洁、通用、相对节省 token；每个工作表前记录名称和有效区域 | 单元格内的逗号、引号和换行由 CSV 规则转义；结构和视觉信息较弱 |
| `tsv` | XLSX | 制表符分隔的二维矩阵 | 与 CSV 类似，包含大量逗号的表格通常更易读 | 单元格中的制表符和换行仍需要转义；结构和视觉信息较弱 |
| `markdown` | XLSX | Markdown 管道表格 | 对模型和人工都较易阅读，行列边界直观 | 宽表会产生较多填充字符；复杂合并结构会被展平 |
| `dataframe` | XLSX | 按列宽对齐的定宽纯文本 | 类似终端中打印 DataFrame 的效果，便于快速浏览 | 名称表示展示风格，并不创建 pandas DataFrame；宽表的空格填充会增加 token |
| `json_rows` | XLSX | 工作表对象中包含 `sheet_name`、`used_range`、`columns`、`rows` 和 `merged_ranges` | 按行组织数据，始终保留行号和列字母；便于筛选、匹配和逐行计算 | 有效区域内的空单元格也会以空字符串保留，稀疏表可能较冗长 |
| `json_cells` | XLSX | 工作表对象中只列出非空单元格，并记录 `cell`、`row`、`column`、`value` 和 `merged_ranges` | 坐标最明确；稀疏表紧凑；适合精确查找和生成单元格操作代码 | 行记录关系不如 `json_rows` 直观；不会把合并单元格左上角的值复制到覆盖区域 |
| `html` | XLSX | 带 `data-sheet` 和 `data-used-range` 属性的 HTML `<table>` | 标签明确，表格边界稳定，并对文本进行 HTML 转义 | 标签开销较大；复杂合并结构会被展平，不生成 `rowspan` 或 `colspan` |

除 `official_latex` 外，动态文本格式均读取 XLSX 中公式的缓存计算结果，而不是公式字符串。多工作表文件会按工作表顺序拼接，各部分以工作表名称分隔。

## 图片格式

| 格式名称 | 数据来源 | 主要特点 | 适用场景与限制 |
| --- | --- | --- | --- |
| `image` | 数据集 `image` 目录中的预生成图片 | RealHiTBench 中是优先级最高的原始视觉来源；能够保留颜色、布局、图表等文本格式难以表达的信息 | 依赖数据集已有图片；纯图片不提供可直接复制的单元格文本和坐标 |
| `excel_1_image` | 数据集 `excel_1_images` 目录中的预生成图片 | 更接近 Excel 工作表的显示效果；SpreadsheetBench 中通常比 `image` 更适合作为格式和布局证据 | 依赖预生成图片是否存在；大工作表可能缩放明显 |
| `default_image` | 运行时从 XLSX 有效区域渲染并缓存到结果目录 | 不依赖数据集预生成图片，可作为统一的图片兜底方案 | 渲染结果受本地 LibreOffice/字体环境影响；首次生成有额外开销 |

如果工作簿包含多个工作表，图片会按工作表顺序附加；文本部分同时列出图片文件名以说明顺序。

## 文本与图片组合格式

组合格式使用 `文本格式+图片格式` 命名。模型会同时收到完整文本序列化、图片顺序说明和实际图片。

| 文本基格式 | `+image` | `+excel_1_image` | `+default_image` | 数据集支持 |
| --- | --- | --- | --- | --- |
| `official_latex` | `official_latex+image` | `official_latex+excel_1_image` | `official_latex+default_image` | 仅 RealHiTBench |
| `latex` | `latex+image` | `latex+excel_1_image` | `latex+default_image` | 两个数据集 |
| `markdown` | `markdown+image` | `markdown+excel_1_image` | `markdown+default_image` | 两个数据集 |
| `html` | `html+image` | `html+excel_1_image` | `html+default_image` | 两个数据集 |

组合格式适合同时需要精确数值和视觉证据的任务，例如颜色、边框、图表、复杂布局或仅靠文本难以判断的表头结构。代价是输入更长，并且要求模型支持图片输入。

## 坐标与合并单元格参数

| 格式 | `include_coordinates` | `fill_merged` | 合并单元格表示 |
| --- | --- | --- | --- |
| `csv`、`tsv`、`markdown`、`dataframe`、`html` | 生效；默认添加行号和列字母 | 生效；启用后把左上角值复制到合并区域 | 默认展平；是否复制值由 `fill_merged` 决定 |
| `json_rows` | 不使用该开关；格式本身始终记录行号和列字母 | 生效 | 额外记录 `merged_ranges`，可选择复制左上角值 |
| `json_cells` | 不使用该开关；每个非空单元格始终带坐标 | 不生效 | 只保留真实非空单元格，并额外记录 `merged_ranges` |
| `latex` | 不生效 | 不生效 | 使用 `\multirow`、`\multicolumn` 和占位单元格保留合并结构 |
| `official_latex` | 不生效 | 不生效 | 由 RealHiTBench 官方文件决定 |
| 图片格式 | 不适用 | 不适用 | 通过工作表截图直接呈现 |

## 简要选择建议

| 任务特点 | 建议优先考虑 |
| --- | --- |
| RealHiTBench 的通用问答基线 | `official_latex` |
| SpreadsheetBench 单元格操作或精确定位 | `json_cells` |
| 按行筛选、匹配、统计记录 | `json_rows` |
| 小型、规则、便于阅读的表格 | `markdown` |
| 复杂表头或大量合并单元格 | `official_latex`（RealHiTBench）或 `latex` |
| 颜色、边框、图表、版式相关问题 | 数据集图片，必要时使用文本与图片组合格式 |
| 没有预生成图片但需要视觉输入 | `default_image` 或相应组合格式 |

