# SheetFlex 多表示推理结果聚合：整体设计说明

> 供 coding agent 在本地项目 `/mnt/data/zhw/sheet_router` 中实现。当前仓库名和包名暂不修改；不要把 `sheet_router` 批量重命名为 `SheetFlex`。

## 1. 目标与范围

SheetFlex 面向同一个电子表格样例在多种结构等价输入表示下得到的推理结果，进行测试时聚合。

固定使用六种表示：

```text
latex, markdown, json_cells, json_rows, image, excel_1_image
```

当前只实现：

- `SheetFlex-vote`：六种表示等权聚合，作为 baseline；生成累计对数概率只用于平票。
- `SheetFlex-agg`：在 `SheetFlex-vote` 完成并验收后，再用样例级自验证置信度进行加权聚合。

暂不实现：

- `SheetFlex-route`；
- 训练式路由器或训练式聚合器；
- 逐单元格拼接新的 Excel 文件；
- 使用测试集准确率设定格式权重。

## 2. 固定实验设定

候选生成保持现有设置：

```text
temperature = 0
top_p = 1
```

自验证置信度生成也使用：

```text
temperature = 0
top_p = 1
```

加权聚合中的 softmax 温度是独立参数，记为 `fusion_temperature`，默认：

```text
fusion_temperature = 1.0
```

不要把模型生成温度与融合温度混为同一个参数。

## 3. 两种方法的统一定义

对样例 `x` 的有效候选集合记为 `V_x`。候选 `i` 的输出为 `o_i`，候选之间的一致性为 `kappa(o_i, o_j)`。

### 3.1 SheetFlex-vote

所有有效候选等权：

```text
w_i = 1,  i in V_x
```

候选分数：

```text
S_i = sum_j kappa(o_i, o_j)
```

### 3.2 SheetFlex-agg

对每个有效候选独立获得样例级自验证置信度 `q_i`，其中 `q_i` 为 0–100 的整数。转换为：

```text
p_i = clamp(q_i / 100, 0.01, 0.99)
c_i = log(p_i / (1 - p_i))
w_i = softmax(c_i / fusion_temperature)
```

候选分数：

```text
S_i = sum_j w_j * kappa(o_i, o_j)
```

无效候选权重严格为 0。缺失但本应存在的置信度默认按 50 分处理，同时必须在输出中记录 fallback；正式实验应尽量做到无 fallback。

固定格式权重只作为消融接口，不进入主方法。若提供外部格式先验 `pi_format`：

```text
w_i ∝ exp(c_i / fusion_temperature
          + format_weight_strength * log(pi_format_i))
```

主实验固定 `format_weight_strength = 0`。

## 4. RealHiTBench 聚合

### 4.1 有效候选

候选满足以下条件才参与聚合：

- `format_valid == True`；
- `model_answer` 非空。

### 4.2 答案等价关系

必须复用现有评测逻辑：

```python
normalize_answer(process_decimal(model_answer))
```

两个规范化答案完全相同时，视为同一答案组：

```text
kappa(o_i, o_j) = 1 if normalized_i == normalized_j else 0
```

`SheetFlex-vote` 选择成员数最多的答案组；`SheetFlex-agg` 选择候选权重和最大的答案组。

### 4.3 Structure Comprehending

该类型不能直接聚合顶层 `model_answer`。必须分别聚合：

- `structure_reference_run`：原始工作簿结果；
- `structure_swap_run`：swap 工作簿结果。

得到两个聚合答案后，再沿用现有 RealHiTBench 指标逻辑比较二者。

## 5. SpreadsheetBench verified_400 聚合

当前阶段允许聚合器读取：

- `answer_position`；
- `answer_sheet`；
- 输入工作簿；
- 六个候选输出工作簿。

选择阶段严禁读取 golden 工作簿。golden 只能在最终评测阶段使用。

### 5.1 有效候选

一个候选必须同时满足：

- `execution_success == True`；
- 对应输出文件存在；
- 文件可被 `openpyxl` 打开；
- 目标工作表存在；
- `answer_position` 的全部区域均能成功解析和读取。

无输出文件、损坏文件、缺失目标 sheet 或无法读取关键区域的候选均弃权。六个候选全部无效时，不生成聚合输出文件，该样例直接失败。

### 5.2 关键区域表示

从每个有效输出文件中提取 `answer_position` 覆盖的全部单元格，使用与评测器一致的值规范化和比较规则。

必须支持：

- 单个单元格；
- 普通矩形区域；
- 多个逗号分隔区域；
- 显式 sheet 名；
- `answer_position` 不含 sheet 时使用 `answer_sheet`；
- 带不规范引号的 sheet 名；
- `A:G` 一类整列范围；
- 重叠区域去重。

### 5.3 候选一致性与选择

对关键区域内全部坐标计算精确单元格一致率：

```text
kappa(o_i, o_j)
  = matched_target_cells(i, j) / total_target_cells
```

值是否相同必须调用与评测一致的比较逻辑，而不是直接使用 Python `==`。

- `SheetFlex-vote`：等权计算每个候选与所有有效候选的一致性总分。
- `SheetFlex-agg`：使用自验证置信度权重计算加权一致性总分。

最终只选择并复制六个候选中的一个完整输出工作簿。禁止逐单元格投票后拼接新工作簿。

## 6. FLEXTAF 风格平票规则

候选生成时可选保存完整 assistant response 的 token log-probability 摘要：

```text
sequence_logprob_sum
sequence_logprob_mean
sequence_token_count
logprob_available
```

正式平票使用累计值：

```text
sequence_logprob_sum = sum_t log P(token_t | previous tokens, input)
```

数值越大越优先。

### RealHiTBench

若多个答案组聚合分数相同：

1. 对每个答案组取组内最大的 `sequence_logprob_sum`；
2. 选择该值最大的答案组；
3. 在获胜组内选择 `sequence_logprob_sum` 最大的原始候选作为最终原始答案。

### SpreadsheetBench

若多个候选工作簿的聚合分数相同，选择 `sequence_logprob_sum` 最大的候选文件。

### 缺失概率时

只有当所有待比较的平票对象都具有有效累计 log-probability 时，才使用该规则；否则按固定格式顺序稳定打破平局：

```text
latex > markdown > json_cells > json_rows > image > excel_1_image
```

输出 trace 中必须记录平票是否发生，以及实际使用的是 `logprob` 还是 `format_order`。

## 7. 样例级自验证置信度

`SheetFlex-agg` 的权重来自候选级、样例级自验证，不来自格式的总体准确率，也不直接使用候选生成 log-probability 作为权重。

### RealHiTBench

对每个有效候选独立验证。验证输入包括：

- 与候选对应的原始表格表示和问题；
- 候选最终 response、reasoning 和 final answer；
- 任务原有答案格式约束。

不得提供金标准答案，也不得同时展示其他格式候选。

### SpreadsheetBench

只验证有效输出候选。验证输入包括：

- 与候选对应的原始表格表示和操作指令；
- `answer_position` 和 `answer_sheet`；
- 最终成功执行的代码；
- 执行成功信息；
- 输入文件到候选输出文件在关键区域内的紧凑变化摘要。

不得读取或提供 golden 工作簿。

验证模型只返回：

```json
{"confidence": 0}
```

其中数值为 0–100 的整数。主实验默认使用与候选生成相同的模型；代码允许显式指定其他 verifier，但不得悄悄更换。

## 8. 推荐代码结构

不要把所有逻辑塞进一个脚本。推荐：

```text
core/eval/spreadsheet_regions.py       # 关键区域解析、规范化和读取
core/sheetflex/common.py               # 候选结构、稳定平票、通用工具
core/sheetflex/realhit.py              # RealHiT 聚合
core/sheetflex/spreadsheet.py          # SpreadsheetBench 聚合
sheetflex_vote.py                      # 等权聚合 CLI
sheetflex_confidence.py                # 自验证置信度采集 CLI
sheetflex_agg.py                       # 加权聚合 CLI
```

文件名可根据仓库现状小幅调整，但需保持职责分离。

## 9. 输出与可审计性

聚合结果至少保存：

- 样例 ID；
- 六个候选的格式、有效性、生成累计 log-probability；
- RealHiT 的规范化答案，或 SpreadsheetBench 的关键区域摘要/hash；
- 每个候选的聚合分数；
- `SheetFlex-agg` 中每个候选的置信度和最终权重；
- 最终选择格式；
- 是否平票；
- 平票处理来源；
- 无效原因和 fallback 情况。

SpreadsheetBench 的 trace 不应保存巨大的完整关键区域数组，可保存 cell 数、hash、两两相似度和必要的诊断摘要。

## 10. 实现顺序

采用五个编码阶段，并在 `SheetFlex-vote` 全量运行后设置一个独立验收关口：

1. 修复并统一 SpreadsheetBench 关键区域解析与评测。
2. 增加候选生成累计 log-probability 的可选记录。
3. 实现并运行 `SheetFlex-vote`。
4. 实现样例级自验证置信度采集。
5. 实现 `SheetFlex-agg` 和固定格式权重消融接口。

在阶段 3 与阶段 4 之间，必须先完成 `SheetFlex-vote` 的全量结果检查和错误诊断。该验收关口不引入新方法。全量 vote 结果未经检查，不进入阶段 4、5。

## 11. 工程约束

- 每次只实现当前阶段，不提前加入后续功能。
- 先阅读本地代码和实际输出文件，再修改；不要仅凭本文猜测路径或 schema。
- 所有聚合按样例 ID 对齐，不能依赖六个 JSON 文件的列表下标一致。
- 新功能默认不覆盖历史结果；使用新目录或新后缀。
- 不删除现有 solver、router 或运行脚本。
- 保持候选生成 `temperature=0, top_p=1`。
- 选择阶段不得访问测试标签或 golden 文件。
- 为核心纯函数编写小型单元测试，并提供最小 smoke-test 命令。
- 每个阶段结束时报告：修改文件、关键设计、测试结果、运行命令、尚未解决的问题。
