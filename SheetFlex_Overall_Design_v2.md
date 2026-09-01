# SheetFlex 多表示推理结果聚合：整体设计说明

> 供 coding agent 在本地项目 `/mnt/data/zhw/sheet_router` 中实现。当前仓库名和包名暂不修改；不要把 `sheet_router` 批量重命名为 `SheetFlex`。

## 1. 目标与范围

SheetFlex 面向同一个电子表格样例在多种结构等价输入表示下得到的推理结果，进行测试时聚合。

固定使用六种表示，格式顺序同时作为最终确定性平票顺序：

```text
latex, markdown, json_cells, json_rows, image, excel_1_image
```

本设计实现三种多表示聚合方法：

- `SheetFlex-vote`：六种表示等权聚合，作为基础方法；候选生成时的累计序列 log-probability 只用于平票。
- `SheetFlex-VerbalConf`：对每个候选独立请求 LLM 给出 0–100 的显式自验证置信度，再进行置信度加权聚合。
- `SheetFlex-VerifyProb`：对每个候选独立执行二元正确性验证，并从验证标签 `A/B` 的 token log-probability 得到候选正确性分数，再进行加权聚合。

现有单格式 `Self-Consistency` 保留为对比基线。实现上述方法时不得改变其候选生成协议、聚合协议或历史输出。

暂不实现：

- `SheetFlex-route`；
- 训练式路由器或训练式聚合器；
- 使用测试集标签训练或校准 verifier；
- 逐单元格拼接新的 Excel 文件；
- 使用测试集准确率设定格式权重；
- 将完整候选生成序列概率直接作为主方法权重。

## 2. 与当前仓库代码的衔接

开始编码前必须阅读本地实际代码和至少一个真实输出目录，不得仅凭本文猜测接口。当前实现应优先复用以下职责：

```text
core/eval/spreadsheet_regions.py       # 关键区域解析、规范化和读取
core/sheetflex/common.py               # FORMAT_ORDER、run-map、平票、通用工具
core/sheetflex/realhit.py              # RealHiT 候选构造、答案规范化和 vote
core/sheetflex/spreadsheet.py          # Spreadsheet 候选校验、相似度和 medoid
sheetflex_vote.py                      # 已有等权聚合 CLI
core/solver/realhit_cot.py             # TableInputBuilder / RealHiT 输入构造
core/solver/spreadsheet_pot.py         # SpreadsheetTableInputBuilder / PoT 输入构造
core/utils.py                          # 模型请求与候选生成 logprob 摘要
```

实现时遵守以下兼容原则：

1. 不复制第二套答案规范化、关键区域解析、工作簿相似度或评测逻辑。
2. 如需抽取公共纯函数，只做最小重构，并用回归测试证明 `SheetFlex-vote` 与 `Self-Consistency` 的现有结果不变。
3. 候选、置信度 sidecar 和聚合结果始终按 `sample_id`、`format`、`branch` 对齐，不能依赖 JSONL 行号一致。
4. 六格式原始候选目录视为只读，不修改、不补写、不覆盖。
5. 验证与选择阶段不得读取 gold answer 或 golden workbook；标签只允许在最终评测和事后分析阶段使用。

## 3. 固定实验设定

### 3.1 候选生成

候选生成沿用已有运行结果和原参数：

```text
temperature = 0
top_p = 1
```

阶段 4 以后不得为了验证或聚合重新生成候选。候选生成时保存的：

```text
sequence_logprob_sum
sequence_logprob_mean
sequence_token_count
logprob_available
```

只属于候选生成轨迹，不属于 verifier 分数。

### 3.2 Verifier 解码

两种验证方法默认使用：

```text
temperature = 0
top_p = 1
n = 1
```

主实验默认 verifier 与候选生成模型相同，但 CLI 必须允许显式指定：

```text
verifier_url
verifier_model_name
```

并在 metadata 中记录。不得根据运行环境悄悄切换模型。

建议参数：

```text
SheetFlex-VerbalConf: max_tokens 取足以返回短 JSON 的值，例如 256
SheetFlex-VerifyProb: max_tokens 取 4 左右，只允许输出 A 或 B
```

是否传递 `seed`、`chat_template_kwargs`、`enable_thinking` 等参数，必须与实际服务能力一致并记录；不得在不同方法间无记录地改变。

### 3.3 融合温度

加权聚合中的 softmax 温度记为：

```text
fusion_temperature = 1.0
```

它只控制候选权重的尖锐程度，与模型生成 `temperature` 无关。

正式结果固定报告 `fusion_temperature=1.0`，并提供离线敏感性分析：

```text
0.5, 1.0, 2.0, 5.0, inf/equal
```

`inf/equal` 表示所有有效候选等权，应退化为 `SheetFlex-vote`。

## 4. 统一符号与聚合框架

对样例 `x`，六种表示产生候选集合。通过数据集对应的有效性检查后，有效候选集合记为：

```text
V_x
```

候选 `i` 的输出记为 `o_i`，候选之间的一致性函数记为：

```text
kappa(o_i, o_j)
```

候选最终融合权重记为 `w_i`，且只在当前样例的有效候选之间归一化：

```text
w_i >= 0
sum_{i in V_x} w_i = 1
```

无效候选权重严格为 0。

所有加权方法统一使用：

```text
S_i = sum_{j in V_x} w_j * kappa(o_i, o_j)
```

其中权重放在作为“参照意见”的候选 `j` 上。候选自身的一致性 `kappa(o_i, o_i)=1` 保留，与现有 medoid 计算保持一致。

### 4.1 SheetFlex-vote

所有有效候选等权：

```text
w_i = 1 / |V_x|
```

由于统一缩放不改变排序，这与现有实现中使用 `w_i=1`、累加一致性总分等价。

### 4.2 SheetFlex-VerbalConf

对每个有效候选独立获得显式置信度：

```text
confidence_i in {0, 1, ..., 100}
```

它的语义必须是：

> 在给定原始任务、可用电子表格证据和当前候选结果的条件下，该候选满足所有实质要求、因而“完全正确”的估计概率百分数。

它不是语言流畅度、代码可执行性、局部正确程度，也不是候选生成序列概率。

转换为候选可靠性：

```text
p_i = clamp(confidence_i / 100, confidence_clip, 1 - confidence_clip)
c_i = log(p_i / (1 - p_i))
```

默认：

```text
confidence_clip = 0.01
```

### 4.3 SheetFlex-VerifyProb

Verifier 面对同一候选，只能在以下两个互斥标签中选择：

```text
A = 候选完全正确，满足全部实质要求
B = 候选并非完全正确，包括错误、部分正确、遗漏要求、存在副作用、证据不足或无法确认
```

请求模型返回标签位置的 top token log-probability。设所有经过轻量规范化后表示 `A` 的 token 变体为集合 `T_A`，表示 `B` 的 token 变体为 `T_B`。先用 log-sum-exp 聚合同义 token 变体：

```text
ell_A = logsumexp({logprob(t) | t in T_A})
ell_B = logsumexp({logprob(t) | t in T_B})
```

再计算二元归一化验证概率：

```text
q_i = exp(ell_A) / (exp(ell_A) + exp(ell_B))
    = sigmoid(ell_A - ell_B)
```

为与 `SheetFlex-VerbalConf` 使用相同的融合接口：

```text
p_i = clamp(q_i, confidence_clip, 1 - confidence_clip)
c_i = log(p_i / (1 - p_i))
```

在没有数值截断时：

```text
c_i = ell_A - ell_B
```

因此 `VerifyProb` 使用的是“候选是否完全正确”这一二元语义事件的标签概率，而不是完整回答、推理文本或 Python 代码的序列概率。

### 4.4 统一权重

两种置信度方法均使用：

```text
w_i = softmax(c_i / fusion_temperature)
```

实现 softmax 时必须减去最大值，避免数值溢出。

若所有有效候选具有相同的 `c_i`，则所有权重相同，聚合选择必须与 `SheetFlex-vote` 完全一致。

### 4.5 缺失分数策略

正式运行默认：

```text
missing_score_policy = error
```

只要某个本应被验证的有效候选缺失分数，该样例或整个运行应按 CLI 约定显式失败，不允许静默使用 0、1 或其他伪造值。

另提供仅用于 smoke test 和诊断的：

```text
missing_score_policy = neutral
```

此时缺失分数按：

```text
p_i = 0.5
c_i = 0
```

处理，并必须在逐样例 trace 和汇总中统计 `missing_score_fallback_count`。正式主实验应做到该计数为 0。

### 4.6 固定格式先验消融

固定格式权重不是主方法。若需要消融，从外部 JSON 读取正值先验 `pi_format`：

```text
w_i ∝ exp(
    c_i / fusion_temperature
    + format_weight_strength * log(pi_format_i)
)
```

主实验固定：

```text
format_weight_strength = 0
```

不得在代码中硬编码由测试集准确率得到的格式权重。

## 5. 三类概率信号必须严格区分

### 5.1 候选生成累计序列 logprob

```text
sequence_logprob_sum
  = sum_t log P(generated_token_t | input, previous_generated_tokens)
```

它覆盖完整 assistant response：

- RealHiTBench 中通常包括 reasoning、JSON 结构、final answer 和标点；
- SpreadsheetBench 中通常包括完整 Python 代码、代码围栏和可能的解释。

它衡量模型生成“这一段具体序列”的偏好，混合了输出长度、措辞、代码风格和模板影响，不等价于候选正确概率。因此：

- 不作为 `SheetFlex-VerbalConf` 或 `SheetFlex-VerifyProb` 的主权重；
- 只沿用现有逻辑作为聚合分数完全平票后的次级判据；
- 缺失时退回固定格式顺序，不影响主加权计算。

### 5.2 Verbalized confidence

```text
confidence_i / 100
```

是模型显式报告的自验证分数。它可以用于排序，但未经标注集分析不能宣称已经校准成真实正确率。

### 5.3 Verification-label probability

```text
q_i = P(A) / (P(A) + P(B))
```

是验证标签 `A/B` 在相同标签位置上的归一化 token probability。它比完整序列概率更直接对应“候选是否正确”，但仍是模型分数，不能未经分析直接宣称具有完美概率校准。

## 6. 通用验证原则

两种验证方法必须共享同一套候选有效性判断、证据构造和任务判据。两种方法的主要差别只能是“如何从 verifier 输出中提取候选分数”。

### 6.1 候选独立验证

每个 `(sample_id, format, branch)` 独立请求 verifier：

- 不同时展示六个候选；
- 不告诉 verifier 当前多数答案；
- 不提供其他候选的分数；
- 不提供候选在评测中是否正确；
- 不提供 Best Single、Oracle 或格式总体准确率。

这样避免 verifier 仅复制多数意见或使用测试标签。

### 6.2 严禁使用 gold

验证证据和聚合选择均不得读取：

- RealHiTBench 的标准答案；
- SpreadsheetBench 的 golden/output workbook；
- 已计算的样例正确标签；
- 任何由 golden 派生的 diff、得分或错误类型。

代码层面应把“gold-free 选择”和“labeled evaluation”保持为不同函数/阶段。

### 6.3 重建与候选一致的表格表示

Verifier 应复用现有输入 builder 重建候选生成时使用的表示：

```text
RealHiTBench: TableInputBuilder
SpreadsheetBench: SpreadsheetTableInputBuilder
```

必须尽可能复用候选记录中的：

```text
table_format
include_coordinates
fill_merged
max_text_tokens
图像表示类型
```

不得另外实现一套格式序列化，也不得无记录地让 verifier 看到比候选生成阶段更完整的源表内容。

不要直接复用“请回答问题”或“请生成 Python 代码”的求解 prompt 作为 verifier 指令。应复用源表表示构造器，再使用独立 verifier prompt，防止任务角色混淆。

### 6.4 图像格式

对 `image` 和 `excel_1_image`：

- 以多模态 message 附上与该格式对应的图片；
- sidecar 不保存巨大的 base64 data URL；
- 保存图片数量、稳定路径或相对路径、内容 hash 和是否缺失；
- 图片构建失败时分数不可用，不得退化成只看候选文本。

### 6.5 证据完整性与截断

每次验证必须记录：

```text
source_representation_truncated
source_text_chars/source_text_tokens
prompt_truncated
region_summary_truncated
side_effect_summary_truncated
```

Verifier prompt 中也要明确告知证据是否被截断。证据不足时，模型应降低置信度或选择 `B`，而不是把缺失信息当作正确证据。

## 7. RealHiTBench 验证证据

对普通样例，验证输入至少包括：

1. 数据集任务类型和问题；
2. 与该候选对应的原始表格表示或图片；
3. 原任务的答案格式约束；
4. 候选最终有效 response；
5. 候选解析出的 reasoning；
6. 候选解析出的 final answer / model_answer；
7. 表格表示是否截断及相关元数据。

Verifier 必须检查：

- 是否使用了正确的 sheet、表头、行列和数据范围；
- 是否正确处理单位、百分比、日期、合并表头和坐标；
- 数值计算或事实判断是否由表格支持；
- reasoning 与 final answer 是否一致；
- 是否满足任务要求的最终答案格式；
- 是否存在关键证据缺失，导致无法确认。

不得只根据 reasoning 是否流畅判断。

### 7.1 Structure Comprehending

该类型必须独立验证两个 branch：

```text
reference = structure_reference_run
swap      = structure_swap_run
```

sidecar key 使用：

```text
(sample_id, format, branch)
```

其中 `branch` 只能是：

```text
main, reference, swap
```

`reference` 和 `swap` 分别构建证据、请求 verifier、保存分数，不共享一个 confidence。

## 8. SpreadsheetBench 验证证据

只对通过现有有效性检查的候选工作簿请求 verifier。验证输入至少包括：

1. 原始操作指令和 `instruction_type`；
2. 与该候选对应的源工作簿表示或图片；
3. `answer_sheet` 和 `answer_position`；
4. 最终成功执行的完整 response 和提取出的 Python 代码；
5. 执行成功、输出文件存在且可打开的信息；
6. 输入工作簿与候选输出工作簿之间的 gold-free 变化摘要；
7. 证据是否截断或扫描不完整。

“代码成功执行”只能证明程序运行，不能证明操作正确。Verifier 必须判断候选输出工作簿是否满足完整指令。

### 8.1 目标区域变化摘要

复用 `spreadsheet_regions.py` 的区域解析和 benchmark 值规范化，至少保存并展示：

```text
target_cell_count
changed_target_cell_count
unchanged_target_cell_count
before/after changed-cell preview
before/after formula changes
before/after style fingerprint changes
目标区域 hash
preview 是否截断
```

changed-cell preview 上限做成 CLI 参数，例如：

```text
--max_changed_cells 100
```

### 8.2 工作簿级与副作用摘要

仅检查目标区域会遗漏错误副作用。应在不读取 golden 的条件下，比较输入工作簿和候选输出工作簿，至少提供：

```text
added_sheets
deleted_sheets
sheet_order_changed
used_range_changes
merged_range_changes
conditional_formatting_summary
changed_cells_outside_target_count
changed_formulas_outside_target_count
changed_styles_outside_target_count
outside-target changed-cell preview
side-effect scan 是否达到上限
```

对超大工作簿必须设置安全扫描上限并记录 `summary_incomplete=True`，不得假装完整检查。

样式 fingerprint 至少考虑：

```text
fill, font, border, alignment, number_format, protection
```

是否实现条件格式的规则级 diff 应根据现有 openpyxl 能力和本地样例验证；若当前无法可靠实现，必须在 summary 中明确标记未检查，而不是默认为无变化。

### 8.3 变更摘要建议 schema

```json
{
  "target_region": {
    "cell_count": 0,
    "changed_cell_count": 0,
    "value_changes": [],
    "formula_changes": [],
    "style_changes": [],
    "before_hash": "...",
    "after_hash": "...",
    "truncated": false
  },
  "workbook_level": {
    "added_sheets": [],
    "deleted_sheets": [],
    "sheet_order_changed": false,
    "merged_range_changes": [],
    "conditional_formatting_checked": false,
    "conditional_formatting_changes": [],
    "changed_cells_outside_target_count": 0,
    "changed_formulas_outside_target_count": 0,
    "changed_styles_outside_target_count": 0,
    "outside_change_preview": [],
    "summary_incomplete": false
  }
}
```

## 9. SheetFlex-VerbalConf 协议

### 9.1 输出语义

Verifier 返回短 JSON。推荐固定 schema：

```json
{
  "evidence_check": "pass",
  "requirement_check": "pass",
  "result_check": "pass",
  "side_effect_check": "not_applicable",
  "issues": [],
  "confidence": 90
}
```

字段约束：

```text
evidence_check: pass | fail | uncertain
requirement_check: pass | fail | uncertain
result_check: pass | fail | uncertain
side_effect_check: pass | fail | uncertain | not_applicable
issues: 最多 3 个简短问题描述
confidence: 0–100 的整数
```

只有 `confidence` 进入权重计算，其他字段用于审计和错误分析。

### 9.2 置信度尺度

Prompt 中明确以下尺度，减少不同样例间含义漂移：

```text
95–100: 证据完整，所有要求均被明确验证，几乎没有剩余疑点
75–94 : 很可能完全正确，但仍有轻微未消除的不确定性
50–74 : 存在关键点未能确认，不能视为强正确证据
25–49 : 更可能错误、遗漏或仅部分正确
0–24  : 存在明确矛盾、实质错误或严重证据缺失
```

必须强调：

- `confidence` 是“完全正确”的概率判断；
- 任一实质性要求未满足，就不能因为其他部分正确而给高分；
- SpreadsheetBench 中仅执行成功不能获得高分；
- 证据被截断或无法验证时应体现不确定性。

### 9.3 解析与修复

- 首次回复必须严格解析为 JSON；
- `confidence` 必须是整数且在 `[0,100]`；
- 允许最多 1 次只修复格式的请求；
- 修复请求不得增加新证据，不得展示其他候选；
- 保存首次和修复 response、解析错误、最终采用的 attempt；
- 仍失败时 `score_available=False`，不得伪造 50 分；是否用中性 fallback 由后续聚合 CLI 决定。

## 10. SheetFlex-VerifyProb 协议

### 10.1 二元标签 prompt

Verifier 必须得到与 `SheetFlex-VerbalConf` 等价的任务证据和检查标准，但最终只输出一个标签：

```text
A = The candidate is fully correct and satisfies every substantive requirement.
B = The candidate is not fully correct. Choose B for any error, omission, partial completion,
    unintended side effect, insufficient evidence, or unresolved uncertainty.

Output exactly one letter: A or B.
Answer:
```

Prompt 应要求模型先在内部完成证据、要求、结果和副作用检查，但不得输出解释，以确保标签位置短且明确。

### 10.2 API 请求

至少请求：

```json
{
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 4,
  "logprobs": true,
  "top_logprobs": 20
}
```

`top_logprobs` 应为 CLI 参数。服务支持时可用 `-1` 返回完整词表，但不得假设所有 vLLM/OpenAI-compatible 版本都支持；全量运行前必须做 preflight probe。

### 10.3 严格 logprob 要求

候选生成阶段为了不中断求解，可以在 API 拒绝 logprob 时保留答案并记录不可用原因；`SheetFlex-VerifyProb` 不允许这种降级，因为标签概率就是主评分信号。

实现方式可选：

- 为 `model_resp` 增加默认关闭的 `require_logprobs=True` 严格模式；或
- 新增专用的严格 verifier 请求函数。

无论采用哪种方式，都必须保证：

- 现有 solver 的默认 fallback 行为不变；
- VerifyProb 请求被拒绝、`choice.logprobs` 缺失或 A/B 概率不全时，当前分数显式失败；
- 不能删除 `logprobs/top_logprobs` 后重新请求并把生成标签当作概率结果；
- 保存 HTTP/API 错误和不可用原因。

### 10.4 标签位置解析

从 `choice.logprobs.content` 中：

1. 允许标签前只有空白 token；
2. 找到第一个非空白的实质 token；
3. 其轻量规范化结果必须是 `A` 或 `B`；
4. 若标签前已经出现其他非空白内容，判为格式无效；
5. 从同一位置的 `top_logprobs` 中提取 A/B 候选。

Token 规范化只允许：

```text
Unicode NFKC
去除首尾空白
去除包围标签的单层引号或双引号
```

不得把 `yes/no`、`true/false` 或其他词自行映射到 A/B。

如果同一位置有多个 token 经规范化后都属于 A，例如 `"A"`、`" A"`、`"A\n"`，使用 log-sum-exp 合并它们的概率质量；B 同理。

### 10.5 可用性判据

只有同时满足以下条件时：

```text
score_available = True
```

- 文本输出可解析出唯一 A/B 标签；
- 标签位置存在 top-logprobs；
- 至少一个 A token 变体和一个 B token 变体存在；
- `ell_A`、`ell_B` 均为有限数；
- 概率计算结果为有限数。

任一条件不满足时，保存原因并停止该候选评分。不得把缺失 B 当作 `P(B)=0`，也不得把生成 A 直接记为 `q=1`。

### 10.6 logprobs 模式

主实验优先使用并记录服务端：

```text
logprobs_mode = raw_logprobs
```

它用于表示采样变换前的模型分布。若服务器只能提供 `processed_logprobs`，可以单独运行，但必须在 metadata 和实验名称中明确区分，不得与 raw 结果混合。

不要使用 guided decoding、logit bias 或强制选择 A/B 作为主实验协议，因为这些机制可能改变标签概率。若作为工程 fallback 或消融使用，必须单独命名和报告。

## 11. RealHiTBench 聚合

### 11.1 有效候选

候选必须满足：

```text
format_valid == True
model_answer 非空
```

两种置信度方法还要求对应 sidecar 分数可用；缺失行为由 `missing_score_policy` 控制。

### 11.2 答案等价关系

必须复用现有评测逻辑：

```python
normalize_answer(process_decimal(model_answer))
```

两个规范化答案完全相同时：

```text
kappa(o_i, o_j) = 1
```

否则：

```text
kappa(o_i, o_j) = 0
```

因此，对任一规范化答案组 `G`，加权聚合分数等价于：

```text
score(G) = sum_{i in G} w_i
```

- `SheetFlex-vote` 选择成员数最多的答案组；
- `SheetFlex-VerbalConf` 选择 verbal confidence 权重和最大的答案组；
- `SheetFlex-VerifyProb` 选择 verification probability 权重和最大的答案组。

获胜组内原始 answer 的代表选择继续复用现有稳定平票逻辑，避免另外引入影响指标的字符串选择规则。

### 11.3 Structure Comprehending

分别对：

```text
structure_reference_run
structure_swap_run
```

加载独立 sidecar 分数并执行聚合。两个 branch 可以有不同有效候选集、不同权重和不同获胜格式。得到两个聚合答案后，再沿用现有 QAMetric 逻辑。

## 12. SpreadsheetBench verified_400 聚合

### 12.1 有效候选

一个候选必须同时满足：

- `execution_success == True`；
- 对应输出文件存在；
- 文件可被 `openpyxl` 打开；
- 目标工作表存在；
- `answer_position` 全部区域均可解析和读取；
- 对应验证 sidecar 分数可用，或由显式 missing policy 处理。

无输出文件、损坏文件、缺失 sheet 或区域读取失败的候选均弃权。六个候选全部无效时，不生成聚合工作簿。

### 12.2 关键区域一致性

复用现有关键区域和值比较逻辑：

```text
kappa(o_i, o_j)
  = matched_target_cells(i, j) / total_target_cells
```

值相等必须调用 benchmark 对应比较函数，不能直接使用 Python `==`。

整列范围 `A:G` 的有限边界由输入工作簿与六个候选工作簿共同确定，选择阶段不得读取 golden。

### 12.3 置信度加权的一致性中心工作簿

Medoid 中文统一写作：

> 一致性中心候选（medoid）：从已有候选中选择与其他候选总体一致度最高的实际候选。

加权版本为：

> 置信度加权的一致性中心工作簿：选择与高权重候选具有最大加权总一致度的已有候选工作簿。

对候选 `i`：

```text
S_i = sum_{j in V_x} w_j * kappa(o_i, o_j)
```

选择 `S_i` 最大的已有工作簿并完整复制到新输出目录。

禁止：

- 逐单元格投票后合成新工作簿；
- 从多个候选拼接 sheet；
- 用 golden 修正候选；
- 因 verifier 分数高而跳过工作簿有效性检查。

## 13. 聚合平票规则

### 13.1 第一层：聚合分数

使用与现有代码一致的严格 `math.isclose` 容差判断聚合分数是否平票。

### 13.2 第二层：候选生成累计 logprob

只有所有待比较对象都有有效：

```text
sequence_logprob_sum
```

时才使用，数值越大越优先。

RealHiTBench 答案组平票时：

1. 每组取组内最大的 `sequence_logprob_sum`；
2. 选择该值最大的答案组；
3. 获胜组内再复用现有代表候选选择逻辑。

SpreadsheetBench 候选工作簿平票时，选择 `sequence_logprob_sum` 最大的候选。

`VerifyProb` 的 `ell_A/ell_B` 已经用于主权重，不能再次冒充候选生成 logprob 平票信号。

### 13.3 第三层：固定格式顺序

任一待比较对象缺失候选生成 logprob，或累计值仍相同，则按：

```text
latex > markdown > json_cells > json_rows > image > excel_1_image
```

稳定打破平局。

每个 trace 必须记录：

```text
tie
tied_count
tie_break_source
tie_break_reason
```

## 14. 验证 sidecar 与可追溯性

六格式原始候选结果不得被修改。每种验证方法独立保存 sidecar。

### 14.1 建议文件名

`SheetFlex-VerbalConf`：

```text
sheetflex_verbalconf_scores.jsonl
sheetflex_verbalconf_metadata.json
sheetflex_verbalconf_diagnostics.json
```

`SheetFlex-VerifyProb`：

```text
sheetflex_verifyprob_scores.jsonl
sheetflex_verifyprob_metadata.json
sheetflex_verifyprob_diagnostics.json
```

### 14.2 Sidecar 主键

每一行对应唯一：

```text
(sample_id, format, branch)
```

即使候选无效，也应写一行 `candidate_valid=False`，但不发送 verifier 请求。这样可以审计六格式覆盖率。

### 14.3 公共字段

至少保存：

```text
schema_version
method
sample_id
format
branch
source_run_dir
candidate_valid
candidate_invalid_reason
score_available
score_unavailable_reason
verifier_model
verifier_url（不得包含密钥）
verifier_temperature
verifier_top_p
verifier_max_tokens
prompt_version
evidence_hash
candidate_record_hash
source_representation_truncated
prompt_truncated
region_summary_truncated
side_effect_summary_truncated
attempts
request_count
raw_response_text
parse_valid
error
```

图像 data URL、完整大表和 API key 不得写入 sidecar。

### 14.4 VerbalConf 特有字段

```text
confidence
evidence_check
requirement_check
result_check
side_effect_check
issues
probability_after_clip
reliability_logit
```

### 14.5 VerifyProb 特有字段

```text
generated_label
label_token_index
top_logprobs_k
logprobs_mode
matched_A_token_variants
matched_B_token_variants
logprob_A
logprob_B
logit_margin
verify_probability
probability_after_clip
reliability_logit
```

只需保存与 A/B 匹配的 token 及 logprob，不必默认保存完整词表 top-logprobs 大对象。可提供 debug 开关另存原始 API response。

### 14.6 Metadata 与配置指纹

Metadata 至少包含：

```text
dataset
run_map 内容与 hash
候选格式顺序
候选模型信息（可从 run 中提取）
verifier 配置
prompt_version
builder 配置
CLI 参数
git commit SHA
创建时间
```

`--resume` 只允许复用与当前配置指纹完全一致的记录。若 run-map、prompt、verifier 模型、top-logprobs 或证据参数改变，应拒绝混合继续，除非用户显式指定新的输出目录。

写 JSONL 时应采用临时文件/原子替换或稳定 checkpoint，避免中断后最后一行损坏。

## 15. 推荐代码结构

保持职责分离，建议：

```text
core/eval/spreadsheet_regions.py        # 已有关键区域逻辑
core/sheetflex/common.py                # 已有 run-map、平票和通用结构
core/sheetflex/realhit.py               # 已有 RealHiT vote，扩展可复用加权入口
core/sheetflex/spreadsheet.py           # 已有 workbook 相似度/medoid，扩展加权入口
core/sheetflex/verification.py          # 公共候选证据、prompt 构造、fingerprint
core/sheetflex/verbal_conf.py           # VerbalConf JSON 解析与分数记录
core/sheetflex/verify_prob.py            # A/B token logprob 解析与概率计算
core/sheetflex/weights.py                # clip、logit、softmax、format prior
core/sheetflex/weighted.py               # 两种方法共享的 sidecar 加载和聚合适配
sheetflex_vote.py                        # 已有等权聚合 CLI
sheetflex_verbalconf.py                  # VerbalConf 采集 CLI
sheetflex_verifyprob.py                  # VerifyProb 采集 CLI
sheetflex_weighted.py                    # --method verbal_conf|verify_prob 的离线聚合 CLI
```

文件名可根据本地结构小幅调整，但必须满足：

- 两种方法共享证据构造；
- 两种方法共享融合和聚合实现；
- VerifyProb 的严格 logprob 请求与候选生成 fallback 明确隔离；
- 不把所有逻辑塞进一个脚本。

建议核心函数签名接近：

```python
def build_realhit_verification_evidence(...): ...
def build_spreadsheet_verification_evidence(...): ...
def parse_verbal_confidence(...): ...
def parse_binary_label_logprobs(...): ...
def reliability_to_weights(...): ...
def load_score_sidecar(...): ...
def aggregate_weighted_realhit_sample(...): ...
def select_weighted_spreadsheet_medoid(...): ...
```

## 16. 聚合输出

### 16.1 SheetFlex-VerbalConf

RealHiTBench：

```text
sheetflex_verbalconf.jsonl
sheetflex_verbalconf_eval.json
sheetflex_verbalconf_score.json
sheetflex_verbalconf_diagnostics.json
```

SpreadsheetBench：

```text
sheetflex_verbalconf.jsonl
spreadsheet_pot_eval.json
spreadsheet_pot_accuracy.json
sheetflex_verbalconf_diagnostics.json
spreadsheet/
```

### 16.2 SheetFlex-VerifyProb

RealHiTBench：

```text
sheetflex_verifyprob.jsonl
sheetflex_verifyprob_eval.json
sheetflex_verifyprob_score.json
sheetflex_verifyprob_diagnostics.json
```

SpreadsheetBench：

```text
sheetflex_verifyprob.jsonl
spreadsheet_pot_eval.json
spreadsheet_pot_accuracy.json
sheetflex_verifyprob_diagnostics.json
spreadsheet/
```

不同方法、模型、数据集和 `fusion_temperature` 使用不同输出目录，不覆盖单格式、vote、self-consistency 或其他 weighted 结果。

### 16.3 聚合 trace

每个样例至少保存：

```text
sample_id
method
valid_candidate_count
每个候选的有效性与原因
生成 sequence logprob 摘要
验证原始分数
clipped probability
reliability logit
normalized fusion weight
RealHiT normalized answer 或 Spreadsheet region hash
answer groups 或 similarity matrix
aggregation_score
selected_format
selected_source_file（Spreadsheet）
tie 与 tie-break 信息
missing-score fallback
provenance/fingerprint
```

Spreadsheet trace 不保存巨大完整区域数组，只保存 cell 数、hash、相似度和必要的 preview。

## 17. 测试要求

### 17.1 公共证据构造

至少测试：

- 普通文本格式和图像格式；
- 源表表示参数从候选 metadata 正确恢复；
- gold 字段不会进入 prompt；
- 其他候选不会进入 prompt；
- 截断标记和 evidence hash 稳定；
- Structure 的 reference/swap 分离；
- Spreadsheet 目标区域、公式、样式和 outside-target 摘要；
- 缺失文件、损坏文件、sheet 缺失的清晰错误。

### 17.2 VerbalConf

至少测试：

- `confidence=0/100/普通整数`；
- 越界值、浮点数、字符串值、缺字段；
- 非 JSON、额外文本和一次格式修复；
- 无效候选不请求模型；
- resume 不重复请求；
- 分数缺失不被伪造成 0 或 50。

### 17.3 VerifyProb

使用人工 API fixture 覆盖：

- 首 token 为 `A` 或 `B`；
- 标签前有空格或换行；
- `"A"`、`" A"` 等多个 token 变体需要 log-sum-exp；
- 标签前有其他实质文本；
- 缺失 `choice.logprobs`；
- 缺失 `top_logprobs`；
- 只有 A、没有 B，或反之；
- 非有限 logprob；
- 服务拒绝 logprobs，严格模式不得静默重试无 logprob；
- `sigmoid(ell_A-ell_B)` 数值正确且稳定。

### 17.4 加权聚合

至少验证：

- 六个分数相同时，两种 weighted 方法与 vote 选择完全一致；
- 高置信少数答案在权重足够大时可以获胜；
- 无效候选始终权重为 0；
- `missing_score_policy=error/neutral` 行为正确；
- Structure 两个 branch 权重独立；
- Spreadsheet 加权一致性中心计算正确；
- 只复制现有候选工作簿；
- `fusion_temperature=inf` 退化为 vote；
- `format_weight_strength=0` 时外部格式先验不产生影响；
- 固定输入下结果可复现；
- 现有 vote 和 self-consistency 回归测试不变。

## 18. 诊断与实验分析

### 18.1 不使用标签的运行前诊断

在正式评测前输出：

```text
candidate_valid 覆盖率
score_available 覆盖率
按格式/任务类型/branch 的分数分布
VerbalConf 的整数唯一值数、全相同分数比例
VerifyProb 的 A/B 标签分布与 margin 分布
A/B 同时出现在 top-logprobs 的覆盖率
解析重试率和 API 失败率
证据截断率
missing-score fallback 数
最终权重熵和最大权重分布
```

若 VerifyProb 的 A/B 概率覆盖不足，不得直接开始全量聚合；应先调整标签、prompt 或 `top_logprobs` 并重新做 preflight。

### 18.2 使用标签的事后分析

标签只在验证 sidecar 固定后离线加入，至少报告：

```text
候选级 AUROC / AUPRC
Brier Score
ECE 与 reliability bins
正确候选和错误候选的分数分布
同一样例内正确候选高于错误候选的 pairwise ranking rate
高置信错误案例
```

聚合层至少比较：

```text
Best Single Format
Self-Consistency
SheetFlex-vote
SheetFlex-VerbalConf
SheetFlex-VerifyProb
Oracle-6
```

相对 `SheetFlex-vote` 报告：

```text
rescue: vote 错，weighted 对
harm: vote 对，weighted 错
keep-correct
keep-wrong
```

同时报告：

```text
verifier 请求次数
输入/输出 token（服务提供时）
平均有效候选数
运行时间
API/解析失败率
```

未经 held-out 数据校准的分数只称为“verification score”或“estimated confidence”，不得仅凭数值把 `0.8` 宣称为真实 80% 正确率。

### 18.3 校准约束

主方法保持 training-free，不在测试集上拟合温度、Platt scaling 或 isotonic regression。

如以后做校准消融：

- 只能使用独立开发集或交叉验证；
- 校准器参数、数据划分和模型必须单独保存；
- 结果必须与未校准主方法分开报告。

## 19. 实现顺序

采用以下阶段，并设置独立验收关口：

1. 统一 SpreadsheetBench 关键区域解析与评测。
2. 增加候选生成累计 token log-probability 的可选记录。
3. 实现并运行 `SheetFlex-vote`。
4. 建立公共 verifier 证据模块并采集 `SheetFlex-VerbalConf` sidecar。
5. 复用公共证据模块采集 `SheetFlex-VerifyProb` sidecar。
6. 实现两种方法共享的离线加权聚合与格式先验消融接口。
7. 完成全量诊断、置信度分析和统一方法比较。

在阶段 3 与阶段 4 之间先验收 vote；阶段 4 与阶段 5 之间验收 VerbalConf 覆盖和分布；阶段 5 与阶段 6 之间验收 VerifyProb 的 A/B logprob 可用率。任何 sidecar 未通过验收，不进入加权聚合。

## 20. 工程约束

- 每次只实现当前阶段，不提前加入后续功能。
- 先阅读本地代码、测试和真实输出，再修改。
- 不删除现有 solver、router、vote、self-consistency 或运行脚本。
- 所有选择按样例 ID 对齐，禁止依赖列表下标。
- 新功能默认不覆盖历史结果，使用新目录或后缀。
- 候选生成设置保持不变。
- 选择函数不得访问测试标签或 golden 文件。
- Verifier sidecar 和聚合结果必须有 schema version、prompt version 和配置指纹。
- API key、Authorization header 和图片 base64 不得写入结果。
- 对核心纯函数编写小型单元测试，并提供最小 smoke-test 命令。
- coding agent 不得自行启动全量昂贵实验；只提供命令，由用户手动确认执行。
- 每阶段结束时报告：修改文件、复用关系、公式到代码的对应、测试结果、smoke-test、运行命令和未解决问题。

## 附录 A：Verifier prompt 模板约束

以下为语义模板。实现时可按当前模型的 chat template 调整外层 message，但不得改变任务判据或向不同格式提供不同正确性标准。

### A.1 公共 system instruction

```text
You are a strict verifier for spreadsheet reasoning results.
Judge whether the candidate is fully correct with respect to every substantive task requirement.
Independently inspect the supplied spreadsheet evidence. Do not trust the candidate merely because its
reasoning is fluent, its JSON is well formed, or its code executed successfully.

You must check:
1. whether the relevant worksheet, headers, rows, columns, coordinates, units, and ranges were interpreted correctly;
2. whether every requested calculation or workbook operation was completed;
3. whether the final answer or output-workbook changes are correct;
4. whether any required part was omitted;
5. for workbook manipulation, whether unintended changes or side effects are present;
6. whether the supplied evidence is sufficient to verify the result.

A candidate is fully correct only when all substantive requirements are satisfied.
If there is an error, an omission, partial completion, an unintended side effect, insufficient evidence,
or unresolved uncertainty, do not treat it as fully correct.
Do not use or assume access to a gold answer or golden workbook.
```

### A.2 RealHiT evidence sections

```text
[Dataset and question type]
...

[Question]
...

[Answer constraints]
...

[Spreadsheet representation]
...

[Candidate final response]
...

[Parsed candidate reasoning]
...

[Parsed candidate final answer]
...

[Evidence completeness]
source_representation_truncated = true|false
...
```

### A.3 SpreadsheetBench evidence sections

```text
[Instruction]
...

[Instruction type]
...

[Answer sheet and answer position]
...

[Input-workbook representation]
...

[Candidate final response and extracted code]
...

[Execution status]
...

[Target-region before/after summary]
...

[Workbook-level and outside-target change summary]
...

[Evidence completeness]
summary_incomplete = true|false
...
```

### A.4 VerbalConf output instruction

```text
Return exactly one JSON object with this schema and no surrounding text:
{
  "evidence_check": "pass|fail|uncertain",
  "requirement_check": "pass|fail|uncertain",
  "result_check": "pass|fail|uncertain",
  "side_effect_check": "pass|fail|uncertain|not_applicable",
  "issues": ["at most three short issues"],
  "confidence": 0
}

The confidence field must be an integer from 0 to 100 and means the estimated percentage chance that
the candidate is fully correct on all substantive requirements. It is not a fluency score and not the
probability that the code merely runs.
```

### A.5 VerifyProb output instruction

```text
Choose exactly one label:
A = The candidate is fully correct and satisfies every substantive requirement.
B = The candidate is not fully correct. Choose B for any error, omission, partial completion,
    unintended side effect, insufficient evidence, or unresolved uncertainty.

Perform the checks internally. Output exactly one letter and no explanation.
Answer:
```
