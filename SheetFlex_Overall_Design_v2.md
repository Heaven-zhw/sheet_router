# SheetFlex 多表示推理结果聚合：整体设计说明

> 供 coding agent 在本地项目 `/mnt/data/zhw/sheet_router` 中实现。当前仓库名和包名暂不修改，不要把 `sheet_router` 批量重命名为 `SheetFlex`。

## 1. 目标与范围

SheetFlex 对同一个电子表格样例构造六种结构等价的输入表示，让同一个下游模型分别推理，再聚合六个候选结果。

固定使用六种表示：

```text
latex, markdown, json_cells, json_rows, image, excel_1_image
```

本设计包含两种多表示聚合方法：

- `SheetFlex-vote`：六个有效候选等权，作为基础方法。
- `SheetFlex-LPVote`：使用已有候选的 `sequence_logprob_mean` 为每个候选分配软权重，再进行加权投票或加权工作簿选择。

现有单格式 `Self-Consistency` 保留为主要对比基线。

`SheetFlex-LPVote` 必须满足：

- 不额外请求 LLM；
- 不重新生成六格式候选；
- 不要求逐 token logprob；
- 只读取现有 `lp_outs` 中已经保存的 `sequence_logprob_mean`；
- 不使用固定格式权重；
- 不使用测试集准确率设置权重；
- 不使用 SpreadsheetBench 的 `test_case_results`、golden workbook 或其他金标准信息参与选择；
- 不逐单元格拼接新的 Excel 文件。

暂不实现：

- 任何需要额外请求 LLM 的候选置信度判断；
- 训练式路由器或训练式聚合器；
- 输入—输出 edit signature 聚合；
- 逐单元格合成工作簿；
- 默认格式软权重或格式可靠性权重。

## 2. 与当前代码的衔接

开始编码前必须阅读本地实际代码和真实输出，不得只根据本文猜测接口。优先复用：

```text
core/eval/spreadsheet_regions.py   # SpreadsheetBench 关键区域解析与值比较
core/sheetflex/common.py           # run-map、格式顺序、logprob 检查、平票工具
core/sheetflex/realhit.py          # RealHiT 候选构造、答案规范化、等权投票
core/sheetflex/spreadsheet.py      # Spreadsheet 候选校验、相似度矩阵、现有一致性选择
sheetflex_vote.py                  # SheetFlex-vote CLI
self_consistency.py                # Self-Consistency 聚合基线
```

实现要求：

1. 不复制第二套答案规范化、候选有效性、关键区域解析、工作簿相似度或评测逻辑。
2. 如需把现有等权函数改造成可接收权重，只做最小重构。
3. 重构后，现有 `SheetFlex-vote` 和 `Self-Consistency` 的结果必须保持不变。
4. 所有候选按 `sample_id` 和 `format` 对齐，不能依赖六个 JSONL 的行号一致。
5. 六格式候选目录视为只读，不修改、不补写、不覆盖。
6. 聚合选择阶段不读取 gold answer 或 golden workbook；只有最终评测阶段可以读取。

## 3. 固定实验输入

候选来自已经完成的六格式单次推理：

```text
temperature = 0
top_p = 1
```

每个候选已经保存：

```text
logprob_available
sequence_logprob_sum
sequence_logprob_mean
sequence_token_count
```

`SheetFlex-LPVote` 只使用：

```text
sequence_logprob_mean
```

不使用 `sequence_logprob_sum` 计算权重。`sum` 仅保留在原始记录中，方便历史结果复现和诊断。

`sequence_logprob_mean` 表示完整模型回复中，每个生成 token 的平均 logprob：

```text
sequence_logprob_mean
  = sequence_logprob_sum / sequence_token_count
```

它在两个数据集中的范围是：

- RealHiTBench：完整回答序列，包括 reasoning、答案和输出格式；
- SpreadsheetBench：完整代码回复，包括 Python 代码和可能存在的代码围栏或解释。

本方法把它视为模型生成当前候选时的“相对确定程度”，而不是严格的答案正确概率。

## 4. 格式顺序和平票顺序

代码保留两套固定顺序：

### 4.1 Recommend 顺序

```text
json_cells > latex > json_rows > markdown > excel_1_image > image
```

### 4.2 Legacy 顺序

```text
latex > markdown > json_cells > json_rows > image > excel_1_image
```

默认使用：

```text
recommend
```

固定顺序只在最终分数仍然完全相同时使用，不作为候选软权重。

## 5. SheetFlex-vote

### 5.1 RealHiTBench

1. 保留有效候选：

```text
format_valid == True
model_answer 非空
```

2. 复用现有逻辑规范化答案：

```python
normalize_answer(process_decimal(model_answer))
```

3. 相同规范化答案属于同一答案组。
4. 每个有效候选计 1 票。
5. 选择票数最多的答案组。
6. 若答案组平票，可以：
   - 使用 `sequence_logprob_mean` 选择；或
   - 直接使用固定 recommend 顺序。

### 5.2 SpreadsheetBench

1. 保留有效候选：

```text
execution_success == True
输出工作簿存在
工作簿可以打开
answer_position 的全部区域可以读取
```

2. 复用当前实现，计算任意两个候选工作簿在目标区域中的单元格一致率：

```text
similarity(i, j)
  = 相同目标单元格数 / 目标单元格总数
```

3. 当前等权分数：

```text
score(i) = sum_j similarity(i, j)
```

4. 选择分数最高的已有工作簿。
5. 若分数平票，可以使用 `sequence_logprob_mean` 或固定 recommend 顺序。
6. 最终只复制一个已有候选工作簿，不生成拼接工作簿。

## 6. SheetFlex-LPVote 的核心思想

当前 `SheetFlex-vote + mean` 只在普通票数或工作簿一致性分数完全平票时，才使用 `sequence_logprob_mean`。

`SheetFlex-LPVote` 的不同之处是：

> 从聚合开始时，就让 mean logprob 较高的候选拥有更大的票，而不是等到最后平票时才使用它。

该方法没有默认格式加成。六种格式的权重只由当前样例中各自的 `sequence_logprob_mean` 决定。

## 7. 候选权重计算

对当前样例的有效候选集合 `V_x`，候选 `i` 的 mean logprob 记为：

```text
l_i = sequence_logprob_mean_i
```

设置一个非负参数：

```text
lp_weight_strength = alpha
```

默认：

```text
alpha = 1.0
```

首先计算未归一化权重：

```text
raw_weight_i = exp(alpha * (l_i - max_l))
```

其中：

```text
max_l = max(l_j for j in V_x)
```

减去 `max_l` 只为避免数值溢出，不改变最终权重比例。

再归一化：

```text
weight_i = raw_weight_i / sum_j raw_weight_j
```

因此：

- mean logprob 越大，候选权重越大；
- 所有有效候选权重之和为 1；
- 无效候选权重为 0；
- 若所有 mean logprob 相同，则所有有效候选等权；
- 若 `alpha=0`，所有有效候选等权，结果应退化为 `SheetFlex-vote + fixed-recommend`；
- `alpha` 越大，最高 mean logprob 候选的影响越强；
- `alpha` 越小，方法越接近等权投票。

### 7.1 简单例子

假设六个候选得到：

| 格式 | 答案 | mean logprob | 归一化权重示例 |
|---|---|---:|---:|
| latex | A | -2.120 | 0.12 |
| markdown | A | -2.040 | 0.13 |
| json_rows | A | -1.897 | 0.15 |
| json_cells | B | -1.204 | 0.30 |
| image | B | -1.386 | 0.25 |
| excel_1_image | C | -2.996 | 0.05 |

普通投票：

```text
A = 3 票
B = 2 票
C = 1 票
```

普通 `SheetFlex-vote` 选择 A。

加权投票：

```text
A = 0.12 + 0.13 + 0.15 = 0.40
B = 0.30 + 0.25 = 0.55
C = 0.05
```

`SheetFlex-LPVote` 选择 B。

## 8. 缺失 mean logprob 的处理

提供两个模式：

```text
missing_logprob_policy = vote | error
```

默认：

```text
vote
```

### 8.1 vote

若某个样例中任一有效候选缺失或具有非法 `sequence_logprob_mean`：

- 不删除该候选；
- 不伪造 0；
- 不用其他样例的均值补齐；
- 整个样例退回等权 `SheetFlex-vote + fixed-recommend`；
- trace 中记录 `lpvote_fallback=True` 和具体原因。

### 8.2 error

只要任一有效候选缺失合法 mean logprob，就终止并报错。该模式用于正式运行前检查 `lp_outs` 的完整性。

若当前样例只有一个有效候选，其权重直接设为 1，不需要其他候选 logprob。

## 9. RealHiTBench 的 LPVote

### 9.1 普通样例

对每个规范化答案组 `G`：

```text
answer_score(G) = sum_{i in G} weight_i
```

选择 `answer_score` 最大的答案组。

获胜答案组中：

1. 优先选择权重最大的候选作为原始答案来源；
2. 若候选权重完全相同，再使用固定 recommend 顺序。

最终输出仍使用现有 `model_answer` 字段和现有 RealHiTBench 评测。

### 9.2 Structure Comprehending

必须分别处理：

```text
structure_reference_run
structure_swap_run
```

两个分支各自：

1. 构造有效候选集合；
2. 读取各自的 `sequence_logprob_mean`；
3. 独立计算候选权重；
4. 独立进行答案组加权投票。

不能使用顶层 `model_answer` 替代两个分支结果。

## 10. SpreadsheetBench 的 LPVote

SpreadsheetBench 保留现有目标区域相似度，不引入新的工作簿 diff 或 edit signature。

设候选 `i` 与候选 `j` 的目标区域一致率为：

```text
similarity(i, j)
```

候选 `i` 的加权一致性分数为：

```text
score(i) = sum_j weight_j * similarity(i, j)
```

含义是：

> 与高权重候选更相似的工作簿，获得更多支持。

注意权重放在提供支持的候选 `j` 上。候选与自身的相似度仍为 1，这样当 `alpha=0` 时，结果可以严格退化为现有等权方法。

最终选择 `score(i)` 最高的已有工作簿并完整复制。

禁止：

- 逐单元格投票后合成新文件；
- 从不同候选拼接 sheet；
- 读取 golden workbook 参与选择；
- 使用 `test_case_results`、`total_hard_restriction` 或其他最终评测结果计算权重。

## 11. LPVote 的最终平票规则

`sequence_logprob_mean` 已经用于主权重，因此 LPVote 的最终平票不再重复使用 mean logprob。

若多个答案组或候选工作簿的加权分数仍然相同：

1. 按用户指定的固定格式顺序选择；
2. 默认使用 recommend 顺序；
3. 可使用 legacy 顺序复现实验。

trace 中必须记录：

```text
tie
tied_count
tie_break_order
tie_break_reason
```

## 12. 推荐代码结构

建议新增：

```text
core/sheetflex/lp_vote.py       # mean logprob 权重计算和公共检查
sheetflex_lp_vote.py            # RealHiT / SpreadsheetBench 离线聚合 CLI
```

也可以把通用权重函数放入：

```text
core/sheetflex/common.py
```

并最小修改：

```text
core/sheetflex/realhit.py
core/sheetflex/spreadsheet.py
```

推荐核心函数：

```python
def compute_lp_weights(candidates, strength, logprob_field="sequence_logprob_mean"):
    ...

def aggregate_answer_lp_vote(candidates, ...):
    ...

def select_spreadsheet_lp_candidate(candidates, similarity_matrix, ...):
    ...
```

要求：

- 现有等权函数保持可用；
- 不把 LPVote 逻辑混入模型请求代码；
- 不修改候选生成 solver；
- 不新增模型服务依赖。

## 13. CLI 设计

新增统一入口：

```text
sheetflex_lp_vote.py
```

至少支持：

```text
realhit / spreadsheet 子命令
--run_map
--output_dir
--ids
--limit
--lp_weight_strength
--missing_logprob_policy vote|error
--tie_break_order recommend|legacy
```

方法固定读取：

```text
sequence_logprob_mean
```

不要把 sum 和 mean 同时输入方法。若以后需要比较 sum，只能作为单独消融，不能改变 `SheetFlex-LPVote` 的主定义。

## 14. 输出文件

### 14.1 RealHiTBench

```text
sheetflex_lp_vote.jsonl
sheetflex_lp_vote_eval.json
sheetflex_lp_vote_score.json
sheetflex_lp_vote_diagnostics.json
```

### 14.2 SpreadsheetBench

```text
sheetflex_lp_vote.jsonl
spreadsheet_pot_eval.json
spreadsheet_pot_accuracy.json
sheetflex_lp_vote_diagnostics.json
spreadsheet/
```

不同模型、数据集和 `lp_weight_strength` 使用不同输出目录，不覆盖：

```text
单格式结果
Self-Consistency
SheetFlex-vote mean
SheetFlex-vote fixed-recommend
其他 LPVote strength 结果
```

## 15. Trace 要求

每个样例至少保存：

```text
sample_id
method = SheetFlex-LPVote
lp_weight_strength
missing_logprob_policy
valid_candidate_count
lpvote_fallback
lpvote_fallback_reason
```

每个候选至少保存：

```text
format
valid
invalid_reason
logprob_available
sequence_logprob_mean
lp_raw_weight
lp_weight
aggregation_score
selected
```

RealHiT 还要保存：

```text
normalized_answer
answer_groups
每个答案组的 weighted_score
```

SpreadsheetBench 还要保存：

```text
region_hash
similarity_matrix
每个候选的 weighted_score
selected_source_file
```

## 16. 诊断统计

至少输出：

```text
有效候选数分布
mean logprob 完整率
LPVote fallback 数和比例
最终选择格式分布
LPVote 与 vote 选择不同的样例数
最终加权分数平票率
每个样例的最大候选权重
最大权重与第二大权重之差
权重接近等权的样例比例
```

RealHiT 额外统计：

```text
答案组数量和大小分布
普通多数答案被 LPVote 改写的次数
```

SpreadsheetBench 额外统计：

```text
Cell-Level / Sheet-Level 分开统计
原一致性分数平票样例数
LPVote 后仍平票的样例数
只有两个有效候选的样例数
平均两两目标区域一致率
```

## 17. 测试要求

### 17.1 权重计算

至少测试：

- mean logprob 越大，权重越大；
- 权重和为 1；
- 无效候选权重为 0；
- 只有一个有效候选时权重为 1；
- 所有 mean 相同时等权；
- `alpha=0` 时等权；
- 极大或极小 logprob 时数值稳定；
- 缺失 mean 时 `vote/error` 两种策略正确。

### 17.2 RealHiTBench

至少测试：

- 普通加权答案组选择；
- 少数高权重答案可以超过多数低权重答案；
- 所有权重相同时与 fixed-recommend vote 一致；
- 获胜组内代表候选选择稳定；
- Structure reference/swap 分别计算权重和聚合；
- 全部候选无效。

### 17.3 SpreadsheetBench

至少测试：

- 已知相似度矩阵下的加权分数正确；
- 两个候选在 mean 不同时能够解除原等权平票；
- 所有权重相同时与现有等权工作簿选择一致；
- 全部 region 相同时按固定顺序；
- 无效、损坏、缺失工作簿不参与；
- 只复制已有候选工作簿；
- 选择阶段不读取 golden。

### 17.4 回归测试

必须确认：

- 现有 `SheetFlex-vote` 结果不变；
- 现有 `Self-Consistency` 结果不变；
- `alpha=0` 的 LPVote 与 `SheetFlex-vote + fixed-recommend` 逐样例一致。

## 18. 实验比较

至少比较：

```text
六种单格式单次推理
六种单格式 Self-Consistency
Best Self-Consistency
SheetFlex-vote + mean
SheetFlex-vote + fixed-recommend
SheetFlex-LPVote
Oracle
```

LPVote 默认报告：

```text
lp_weight_strength = 1.0
```

离线敏感性分析建议：

```text
0, 0.5, 1, 2, 5, 10
```

其中：

- `0` 用于验证退化到等权投票；
- `1` 是未额外放大的主设置；
- 更大的值用于检查是否需要更明显地区分候选权重。

不得把在测试集上表现最好的 strength 自动改称为预先设定的主结果。若以后需要选择非 1 的固定值，应使用独立开发集确定。

## 19. 结果分析

相对于 `SheetFlex-vote + mean` 和 `SheetFlex-vote + fixed-recommend`，至少统计：

```text
rescue：原方法错误，LPVote 正确
harm：原方法正确，LPVote 错误
keep-correct：两者都正确
keep-wrong：两者都错误
```

同时报告：

```text
选择变化率
平票解除率
LPVote fallback 率
按最终选择格式分组的准确率
```

SpreadsheetBench 必须按：

```text
Hard all
Hard sheet
Hard cell
```

分别报告，并重点检查 Cell-Level 平票样例。

使用 gold 的 rescue/harm 和准确率只允许在聚合完成后的评测脚本中计算，不能反馈到候选权重或选择过程。

## 20. 实现顺序

采用五个阶段：

1. 统一并修复 SpreadsheetBench 关键区域解析与评测。
2. 记录候选生成的 sequence logprob sum、mean 和 token count。
3. 实现并运行 `SheetFlex-vote` 与 Self-Consistency 基线。
4. 实现 `SheetFlex-LPVote`，完成单元测试和小规模 smoke test。
5. 对现有 `lp_outs` 离线运行 LPVote，完成 strength 敏感性、结果验收和统一比较。

阶段 4 和阶段 5 均不得请求大模型或重新生成候选。

## 21. 工程约束

- 每次只实现当前阶段，不提前加入后续方法。
- 先阅读本地代码、测试和实际输出，再修改。
- 新功能默认写入新目录，不覆盖历史结果。
- 候选生成设置保持不变。
- 所有选择按样例 ID 对齐。
- 聚合选择函数不得读取测试标签、`test_case_results` 或 golden workbook。
- `test_case_results` 只能用于最终评测和事后分析。
- 不实现格式软权重，不根据模型或任务类型设置格式权重。
- 不增加任何额外 LLM 置信度判断请求。
- 不逐单元格合成工作簿。
- 为核心纯函数编写单元测试，并提供可复现的离线运行命令。
- coding agent 每阶段结束时报告：修改文件、公式与代码对应、测试结果、smoke test、运行命令和未解决问题。
