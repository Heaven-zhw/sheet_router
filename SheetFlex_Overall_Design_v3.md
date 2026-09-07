# SheetFlex 多表示结果聚合：整体设计 v3

> 本文供 coding agent 在 `/mnt/data/zhw/sheet_router` 中实现后续方法。开始编码前必须核对本地代码、测试和真实结果 schema；本文描述设计约束，不替代代码事实。

## 1. 目标与范围

SheetFlex 为同一个电子表格样例构造六种结构等价的输入表示，由同一模型分别生成候选，再离线聚合。

固定格式及规范顺序：

```text
latex, markdown, json_cells, json_rows, image, excel_1_image
```

本设计覆盖四类方法：

1. `Self-Consistency`：同一格式的六条独立轨迹，作为单格式基线。
2. `SheetFlex-vote`：六格式等权聚合。
3. `SheetFlex-LPVote`：使用 `sequence_logprob_mean` 对六格式候选加权。
4. `SheetFlex-CGVote`：在 LPVote 的跨候选支持之外，再用输出等价类自身置信度进行门控。

本阶段新增的是：

```text
SheetFlex-CGVote = Confidence-Gated Vote
```

CGVote 必须满足：

- 完全离线，不请求 LLM，不重新生成候选；
- 只读取已有六格式候选和 `sequence_logprob_mean`；
- 不使用测试集准确率、gold answer、golden workbook 或评测结果参与选择；
- 不使用固定格式软权重；
- 不训练路由器或聚合器；
- SpreadsheetBench 只选择并复制一个已有工作簿，不逐单元格合成；
- 暂不评价或优化同一 `region_hash` 类内的完整文件质量。

## 2. 现有代码与复用边界

实现前至少核对：

```text
core/eval/spreadsheet_regions.py
core/sheetflex/common.py
core/sheetflex/realhit.py
core/sheetflex/spreadsheet.py
core/sheetflex/lp_vote.py
sheetflex_vote.py
sheetflex_lp_vote.py
self_consistency.py
tests/
```

优先复用：

- RealHiT 的 `process_decimal`、`normalize_answer` 和现有指标；
- SpreadsheetBench 的关键区域解析、值规范化和 `compare_cell_value`；
- 候选有效性、工作簿边界计算和区域相似度矩阵；
- run-map、ID 对齐、logprob 检查和固定顺序；
- LPVote 的稳定 softmax 权重计算；
- 现有最终评测和输出工作簿复制逻辑。

不得复制第二套略有差异的解析、相似度或评测实现。新增 CGVote 应放在小型独立模块，例如：

```text
core/sheetflex/cg_vote.py
sheetflex_cg_vote.py
```

对现有模块只做必要的参数化和纯函数提取。`SheetFlex-vote`、`SheetFlex-LPVote` 和 `Self-Consistency` 的既有行为必须保持不变。

## 3. 固定输入与候选有效性

六格式候选来自已有单次推理：

```text
temperature = 0
top_p = 1
```

每个候选保留：

```text
logprob_available
sequence_logprob_sum
sequence_logprob_mean
sequence_token_count
```

CGVote 只使用：

```text
sequence_logprob_mean
```

它是完整模型回复的 token 平均 logprob，只视为相对生成置信度，不声称是校准后的答案正确概率。

RealHiT 有效候选：

```text
format_valid == True
model_answer 非空
```

SpreadsheetBench 有效候选：

```text
execution_success == True
输出工作簿存在且可打开
answer_position 的全部关键区域可读取
候选与输入解析出的关键区域坐标一致
```

所有结果必须按字符串化 `sample_id` 和 `format` 对齐，不依赖数组位置。

## 4. 固定顺序

保留两套稳定顺序。

Recommend，默认：

```text
json_cells > latex > json_rows > markdown > excel_1_image > image
```

Legacy，仅用于复现实验：

```text
latex > markdown > json_cells > json_rows > image > excel_1_image
```

固定顺序只处理最终仍相同的分数和等价类代表选择，不作为软权重，也不随模型或数据集改变。

## 5. 统一的加权投票视角

设当前样例的有效候选为 `V`，候选 `i` 的输出为 `y_i`，平均 logprob 为 `l_i`。

### 5.1 候选权重

设置：

```text
lp_weight_strength = alpha >= 0
```

默认：

```text
alpha = 1.0
```

计算：

```text
raw_weight_i = exp(alpha * (l_i - max_l))
weight_i = raw_weight_i / sum_j raw_weight_j
```

记归一化权重为 `p_i`。无效候选权重为 0，有效候选权重之和为 1。

### 5.2 输出相似度核

两数据集使用同一个加权投票形式：

```text
support(y) = sum_j p_j * K(y, y_j)
```

区别只在相似度核 `K`。

RealHiT：

```text
K(y_i, y_j) = 1，当规范化答案相同
K(y_i, y_j) = 0，否则
```

SpreadsheetBench：

```text
K(y_i, y_j)
  = 关键区域中值相同的单元格数 / 关键区域总单元格数
```

因此 RealHiT 是离散答案投票，SpreadsheetBench 是连续相似度投票。

## 6. 输出等价类

CGVote 先将预测等价的候选折叠成类 `G`。

RealHiT 等价类：

```python
normalize_answer(process_decimal(model_answer))
```

相同规范化答案属于同一类。

SpreadsheetBench 等价类：

```text
region_hash 相同
```

`region_hash` 必须来自阶段 1 的规范化关键区域映射。相同 hash 表示在当前 benchmark 目标区域和值比较语义下等价，不代表完整工作簿完全相同。

折叠时不得丢弃多格式共识。类权重总量为：

```text
W_G = sum_{i in G} p_i
```

类成员数、成员格式、每个成员的 logprob 和权重都要保留在 trace 中。

## 7. SheetFlex-vote 与 LPVote

本节固定已有方法的语义，CGVote 不得改变它们。

### 7.1 SheetFlex-vote

RealHiT：按规范化答案计数，选择成员数最大的答案组。

SpreadsheetBench：

```text
score(i) = sum_j K(y_i, y_j)
```

选择分数最高的已有工作簿。

平票可使用 `sequence_logprob_mean` 或固定顺序，具体由现有 CLI 配置决定。

### 7.2 SheetFlex-LPVote

RealHiT 类分数：

```text
LPScore(G) = W_G
```

SpreadsheetBench 类分数：

```text
LPScore(G) = sum_H W_H * K(G, H)
```

候选级现有实现与类级公式等价：同一 `region_hash` 的候选具有相同的相似度向量，因而得到相同 LPScore。

## 8. SheetFlex-CGVote

### 8.1 核心动机

LPVote 只让高置信候选向相似结果提供更多支持。CGVote 进一步要求：候选类自身也应包含至少一个高置信成员。

类自身置信度定义为：

```text
Q_G = max_{i in G} p_i
```

使用最大值而不是求和，是为了避免成员数量同时在 `W_G` 和 `Q_G` 中重复计数。

设置非负门控强度：

```text
confidence_gate_strength = beta >= 0
```

默认主设置：

```text
beta = 1.0
```

### 8.2 类共识分数

统一定义：

```text
C_G = sum_H W_H * K(G, H)
```

RealHiT 的不同答案类之间 `K=0`，同类 `K=1`，因此：

```text
C_G = W_G
```

SpreadsheetBench 使用类代表的关键区域映射计算 `K(G,H)`。同一类内任一成员均可作为相似度代表。

### 8.3 门控分数

CGVote 最终分数：

```text
CGScore(G) = C_G * Q_G ** beta
```

实现时建议在 log 空间比较：

```text
log_CGScore(G)
  = log(C_G + epsilon) + beta * log(Q_G + epsilon)
```

`epsilon` 只能用于数值稳定，不得改变正常正值的排序。trace 同时保存原始 `C_G`、`Q_G` 和 `CGScore(G)`，便于审计。

### 8.4 退化性质

实现必须满足：

1. `beta=0`：

   ```text
   CGVote == 当前 SheetFlex-LPVote
   ```

2. `alpha=0`：所有有效候选 `p_i` 相同，`Q_G` 对所有类相同，因此：

   ```text
   CGVote == SheetFlex-vote + fixed order
   ```

3. 所有候选属于同一等价类：直接选择该类，类内按稳定规则选代表。

4. 只有一个有效候选：直接选择，不需要置信度比较。

这些性质必须逐样例回归验证，而不只是比较总分。

## 9. RealHiTBench CGVote

### 9.1 普通样例

1. 构造有效候选；
2. 计算 `p_i`；
3. 按规范化答案构造类 `G`；
4. 计算 `W_G`、`Q_G`、`C_G=W_G`；
5. 计算 `CGScore(G)`；
6. 选择分数最高的答案类。

获胜类内的原始回答来源：

1. 选择 `p_i` 最大的成员；
2. 若仍相同，使用指定固定顺序。

最终评测继续使用现有 `model_answer` 和 QAMetric。

### 9.2 Structure Comprehending

必须分别对：

```text
structure_reference_run
structure_swap_run
```

独立完成有效性检查、权重计算、分组和 CGVote。不得用顶层 `model_answer` 替代两个分支结果。

## 10. SpreadsheetBench CGVote

1. 使用现有公共函数校验候选并提取全部关键区域；
2. 使用输入文件和六个候选共同确定 `A:G` 等开放范围的有限边界；
3. 按 `region_hash` 折叠等价类；
4. 令 `W_G` 为类内候选权重之和；
5. 在不同类之间计算关键区域相似度 `K(G,H)`；
6. 计算 `C_G`、`Q_G` 和 `CGScore(G)`；
7. 选择最高分等价类；
8. 从获胜类按固定顺序选择一个代表工作簿并完整复制。

当前阶段不比较类内工作簿的公式、样式、非目标区域修改或完整性。类内代表选择只为产生确定的输出文件，不作为方法贡献，也不影响当前目标区域值评测。

禁止：

- 读取 golden workbook 或 `test_case_results` 参与聚合；
- 根据模型、数据集或测试集表现调整格式权重；
- 从不同候选拼接 sheet 或单元格；
- 使用 `sequence_logprob_sum` 替代 mean；
- 将相同 `region_hash` 的成员当作需要解决的预测冲突。

## 11. 缺失 logprob 与最终平票

沿用 LPVote 的策略：

```text
missing_logprob_policy = vote | error
```

默认 `vote`：若任一有效候选缺失合法 mean，整个样例回退到 `SheetFlex-vote + fixed-recommend`。不删除候选、不补 0、不跨样例填充。

`error`：发现缺失或非法 mean 立即报错，用于正式运行前检查。

若多个不同等价类的 `CGScore` 经严格 `math.isclose` 判断仍相同：

1. 按指定固定格式顺序比较各类中排名最高的成员；
2. trace 记录平票类、容差和 `tie_break_source=format_order`。

Spreadsheet 获胜类内有多个工作簿时同样按固定顺序选择代表，但该事件应记录为：

```text
equivalent_class_representative_selection
```

不得计入预测冲突平票率。

## 12. CLI 与输出

建议新增：

```text
sheetflex_cg_vote.py realhit
sheetflex_cg_vote.py spreadsheet
```

公共参数：

```text
--run_map
--output_dir
--ids
--limit
--lp_weight_strength             # alpha，默认 1.0
--confidence_gate_strength       # beta，默认 1.0
--missing_logprob_policy vote|error
--tie_break_order recommend|legacy
```

输出不得覆盖单格式、Self-Consistency、SheetFlex-vote 或 LPVote：

RealHiT：

```text
sheetflex_cg_vote.jsonl
sheetflex_cg_vote_eval.json
sheetflex_cg_vote_score.json
sheetflex_cg_vote_diagnostics.json
```

SpreadsheetBench：

```text
sheetflex_cg_vote.jsonl
spreadsheet_pot_eval.json
spreadsheet_pot_accuracy.json
sheetflex_cg_vote_diagnostics.json
spreadsheet/
```

不同 `alpha`、`beta` 必须写入不同目录或由输出 manifest 明确区分。非空目录默认拒绝覆盖。

## 13. Trace schema

每个样例至少保存：

```text
id
method = SheetFlex-CGVote
lp_weight_strength
confidence_gate_strength
missing_logprob_policy
valid_candidate_count
equivalence_class_count
selected_class_id
selected_format
fallback
fallback_reason
```

每个候选至少保存：

```text
format
valid
invalid_reason
sequence_logprob_mean
lp_raw_weight
lp_weight
class_id
selected
```

每个等价类至少保存：

```text
class_id                         # normalized_answer 或 region_hash
member_formats
member_count
class_weight_mass               # W_G
class_confidence                 # Q_G
consensus_score                 # C_G
confidence_gated_score          # CGScore(G)
selected
```

RealHiT 额外保存答案组和 Structure 两侧独立 trace。

SpreadsheetBench 额外保存：

```text
answer_position
answer_sheet
class_similarity_matrix
selected_source_file
equivalent_class_representative_selection
```

## 14. 诊断统计

至少输出：

```text
有效候选数分布
等价类数量和大小分布
mean logprob 完整率
fallback 数和比例
最终不同等价类平票数和比例
类内代表选择次数
最终选择格式分布
CGVote 相对 LPVote 的选择变化数
CGVote 相对 fixed-recommend 的选择变化数
每个样例最大候选权重和 top-2 权重差
W_G、Q_G、C_G、CGScore 的分布
```

SpreadsheetBench 额外按 Cell-Level / Sheet-Level 分开统计，并报告：

```text
等价平票数量（只诊断，不计为冲突）
valid_candidate_count = 2 且 hash 不同的强制平票
valid_candidate_count >= 3 的不同 hash 冲突平票
平均两两区域一致率
```

使用 gold 的 accuracy、rescue、harm 只能在聚合完成后计算，不能反馈到选择函数。

## 15. 测试要求

### 15.1 公共权重与类构造

- 权重和为 1，无效候选权重为 0；
- mean 越大，权重越大；
- `alpha=0` 时等权；
- 极端 logprob 数值稳定；
- 缺失 mean 的 `vote/error` 行为正确；
- 等价类按 ID/hash 正确折叠；
- 类权重 `W_G` 等于成员权重之和；
- 类置信度 `Q_G` 等于成员最大权重。

### 15.2 CGScore

- 手工小矩阵的 `C_G`、`Q_G`、`CGScore` 正确；
- 相同共识下，更高类置信度可以通过门控获胜；
- `beta=0` 与 LPVote 逐样例一致；
- `alpha=0` 与等权 fixed-recommend 逐样例一致；
- 最终类平票按指定格式顺序稳定处理。

### 15.3 RealHiT

- 普通答案组选择；
- 多成员答案组的 `W_G` 和 `Q_G`；
- Structure reference/swap 独立聚合；
- 获胜类内代表选择；
- 全部无效。

### 15.4 SpreadsheetBench

- 相同 `region_hash` 候选折叠且保留权重总量；
- 类级相似度与候选级 LPVote 在 `beta=0` 时等价；
- 不同 hash 类的门控选择；
- 相同 hash 类内按固定顺序复制一个已有工作簿；
- 损坏、缺失、非法区域候选不参与；
- 聚合选择阶段不访问 golden；
- 不生成拼接工作簿。

### 15.5 回归

必须保证现有全部测试继续通过，并在真实候选上验证：

```text
CGVote(beta=0) == LPVote
CGVote(alpha=0) == SheetFlex-vote + fixed-recommend
```

## 16. 实验协议

主设置预先固定：

```text
alpha = 1.0
beta = 1.0
missing_logprob_policy = error
tie_break_order = recommend
```

建议离线消融：

```text
alpha: 0, 0.5, 1, 2
beta: 0, 0.25, 0.5, 1, 2
```

不得在 verified_400 上选择最优超参数后将其表述为预设主结果。非默认值应由独立开发集确定，或明确标为测试集敏感性分析。

至少比较：

```text
六种单格式单次结果
六种单格式 Self-Consistency
Best Self-Consistency
SheetFlex-vote + mean
SheetFlex-vote + fixed-recommend
SheetFlex-LPVote
SheetFlex-CGVote
六格式 Oracle
```

事后分析至少报告：

```text
rescue
harm
keep-correct
keep-wrong
选择变化率
相对 LPVote 的净提升
等价类数与正确率关系
强制/冲突平票子集结果
```

## 17. 实现阶段

阶段 A：公共等价类与 CGScore 纯函数。

阶段 B：RealHiT 普通样例和 Structure Comprehending 适配。

阶段 C：SpreadsheetBench `region_hash` 类聚合与代表文件复制。

阶段 D：CLI、trace、diagnostics 和回归测试。

阶段 E：两个数据集各 5–10 个样例 smoke test，再由用户手动运行全量离线聚合。

任何阶段均不得重新请求模型。

## 18. 非目标与风险

当前不实现：

- Workbook Integrity Reranker；
- 公式完整性、样式或非目标区域质量排序；
- edit signature 聚合；
- 自验证置信度；
- 固定格式权重；
- 训练式路由；
- 逐单元格或逐 sheet 合成。

主要风险：

1. `sequence_logprob_mean` 反映生成流畅度，不是校准后的工作簿正确率。
2. `C_G` 已使用 logprob 权重，`Q_G ** beta` 会再次使用置信度；这是 CGVote 的有意门控，必须通过 `beta=0` 消融验证贡献。
3. 不同格式回复长度和生成风格可能造成 mean logprob 尺度偏差。
4. SpreadsheetBench 当前只按目标区域值定义等价类；类内完整工作簿质量不在本方法结论范围内。
5. 测试集上的小幅提升不能证明普适性，必须跨模型报告并给出 rescue/harm。

## 19. 验收输出

每阶段完成后报告：

```text
修改/新增文件
公式与关键函数对应关系
trace 和 diagnostics 示例
单元测试命令与结果
真实候选 smoke-test 命令与结果
全量离线运行命令
beta=0 / alpha=0 退化验证
证明选择阶段未读取 gold/golden
git diff --stat
尚未解决的风险
```
