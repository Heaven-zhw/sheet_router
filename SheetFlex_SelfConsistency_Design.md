# SheetFlex：Self-Consistency Baseline 实现方案

## 1. 目标

在现有 `sheetflex-aggr` 分支上，为以下两个数据集实现与 `SheetFlex-vote` 等预算的 Self-Consistency baseline：

- RealHiTBench：同一种表格格式重复生成 6 条候选答案，再投票得到最终答案。
- SpreadsheetBench `verified_400`：同一种表格格式重复生成 6 个候选工作簿，再按关键区域一致性选择最终工作簿。

Self-Consistency 与 `SheetFlex-vote` 使用相同的候选数：

```text
num_samples = 6
temperature = 0.1
top_p = 1.0
base_seed = 42
```

候选生成、有效性判断、投票方式、关键区域比较和平票处理应尽量直接复用现有 `SheetFlex-vote` 代码，不能重新实现一套语义不同的聚合逻辑。

---

## 2. 核心实验定义

### 2.1 固定一种输入格式

每次 Self-Consistency 实验只使用一种 `table_format`。6 条候选必须来自：

- 同一个模型；
- 同一个数据集样例；
- 同一种表格格式；
- 同一套 prompt 和求解流程；
- 不同的随机采样轨迹。

`table_format` 由运行配置显式指定，不在代码中根据测试集结果自动选择。

### 2.2 随机种子

Self-Consistency 的公开配置使用：

```text
base_seed = 42
```

为避免对相同 prompt 连续发送完全相同的随机种子而得到重复轨迹，6 条候选使用确定性派生种子：

```text
trajectory_seed(k) = base_seed + k
k = 0, 1, 2, 3, 4, 5
```

即默认种子为：

```text
42, 43, 44, 45, 46, 47
```

输出 metadata 中同时保存 `base_seed`、`sample_index` 和实际 `seed`。代码对外仍以 `--base_seed 42` 作为默认配置。

### 2.3 不使用单次请求的 `n=6`

当前项目的求解器按“一次请求—一次格式校验/代码执行—必要时修复重试—一个最终候选”的方式组织。SpreadsheetBench 的每条轨迹还必须独立执行代码并保存独立工作簿。

因此不把 `n=6` 直接塞入一次 API 请求，而是运行 6 个独立、可恢复、可审计的完整求解轨迹。每个轨迹只贡献一个最终候选；轨迹内部的格式修复或代码执行重试不作为额外投票样本。

---

## 3. 推荐的数据流

采用“候选生成”和“离线聚合”两步结构：

```text
同一格式 × 6 个 seed
        │
        ├── candidate_0 / seed=42
        ├── candidate_1 / seed=43
        ├── candidate_2 / seed=44
        ├── candidate_3 / seed=45
        ├── candidate_4 / seed=46
        └── candidate_5 / seed=47
        │
        ▼
Self-Consistency 离线聚合
        │
        ├── RealHiTBench：答案投票
        └── SpreadsheetBench：关键区域一致性中心候选
```

这样可以：

- 复用现有两个 solver；
- 单独检查每条采样轨迹；
- 中断后按 run 恢复；
- 不把生成、执行、聚合和评测堆进一个大脚本；
- 在不重新请求模型的情况下修改或复查聚合逻辑。

---

## 4. 候选生成改动

### 4.1 为现有运行入口增加可选 seed

在以下入口及对应 solver 中传递 `seed`：

- `realhit_cot.py`
- `spreadsheet_pot.py`
- `core/solver/realhit_cot.py`
- `core/solver/spreadsheet_pot.py`

兼容要求：

- 普通旧实验的 `--seed` 默认仍可为 `None`，避免改变已有行为；
- Self-Consistency 生成器的 `--base_seed` 默认是 `42`；
- seed 非空时写入模型请求的 `model_params["seed"]`；
- seed 非空时写入 run metadata 和逐样例结果；
- 输出目录后缀应包含 seed，防止 6 个 run 相互覆盖。

### 4.2 生成参数

Self-Consistency 的候选生成固定为：

```text
temperature = 0.1
top_p = 1.0
save_logprobs = True
num_samples = 6
base_seed = 42
```

必须保存最终有效 response 的：

```text
logprob_available
sequence_logprob_sum
sequence_logprob_mean
sequence_token_count
```

平票处理只使用 `sequence_logprob_sum`。

### 4.3 运行清单

建议为每次 Self-Consistency 实验生成一个 manifest，例如：

```json
{
  "method": "self_consistency",
  "dataset": "realhitbench",
  "table_format": "latex",
  "num_samples": 6,
  "temperature": 0.1,
  "top_p": 1.0,
  "base_seed": 42,
  "runs": [
    {
      "candidate_id": "sample_0",
      "sample_index": 0,
      "seed": 42,
      "run_dir": "/path/to/run_0"
    },
    {
      "candidate_id": "sample_1",
      "sample_index": 1,
      "seed": 43,
      "run_dir": "/path/to/run_1"
    }
  ]
}
```

完整 manifest 必须恰好有 6 个 run，且六个 run 的模型、数据集、格式、温度和 `top_p` 一致。

---

## 5. RealHiTBench 聚合

### 5.1 有效候选

沿用 `SheetFlex-vote`：

```text
format_valid == True
且 model_answer 非空
```

无效轨迹弃权，不作为一个答案类别。

### 5.2 投票

沿用现有答案规范化：

```python
normalize_answer(process_decimal(model_answer))
```

将规范化答案相同的候选分为同一组，选择成员数最多的答案组。

### 5.3 平票

完全复用 `SheetFlex-vote` 的语义：

1. 若多个答案组票数相同，对每组取组内最大的 `sequence_logprob_sum`；
2. 若所有待比较组都具有有效 log-probability，选择组内最大值最高的答案组；
3. 获胜组内部选择 `sequence_logprob_sum` 最大的原始候选作为最终输出；
4. 若相关 log-probability 缺失或仍完全相同，按 `sample_index=0,1,...,5` 的固定顺序选择。

这里的固定顺序是 Self-Consistency 对应的稳定候选顺序，作用等同于 `SheetFlex-vote` 中的固定格式顺序。

### 5.4 Structure Comprehending

必须分别聚合：

- `structure_reference_run`
- `structure_swap_run`

六条 reference 轨迹投票得到 reference 答案，六条 swap 轨迹投票得到 swap 答案，然后调用现有指标逻辑比较二者。不能只聚合顶层 `model_answer`。

---

## 6. SpreadsheetBench verified_400 聚合

### 6.1 有效候选

沿用 `SheetFlex-vote`：

- `execution_success == True`；
- 输出文件存在且可打开；
- 全部关键区域可解析；
- 目标 sheet 和目标单元格均可读取。

无效轨迹弃权。六条轨迹都无效时，不生成聚合输出文件。

### 6.2 一致性分数

继续使用现有关键区域提取和单元格比较逻辑。对候选工作簿 `i`、`j`：

```text
similarity(i, j)
  = 关键区域内值相同的单元格数 / 关键区域总单元格数
```

对每个有效候选计算：

```text
score(i) = sum_j similarity(i, j)
```

选择分数最高的已有候选工作簿。禁止逐单元格拼接新的工作簿。

### 6.3 平票

沿用 `SheetFlex-vote`：

1. medoid 分数相同的候选中，若均有有效 `sequence_logprob_sum`，选择累计 log-probability 最大者；
2. 否则按 `sample_index=0,1,...,5` 的固定顺序选择；
3. 复制被选中的完整工作簿到 Self-Consistency 输出目录。

聚合阶段可以读取 `answer_position`、`answer_sheet` 和输入工作簿，但不能读取 golden 工作簿。golden 只能用于最终评测。

---

## 7. 代码复用要求

当前 `SheetFlex-vote` 的核心代码位于：

```text
core/sheetflex/common.py
core/sheetflex/realhit.py
core/sheetflex/spreadsheet.py
sheetflex_vote.py
```

实现 Self-Consistency 时应做小范围泛化，而不是复制：

- 将平票 fallback 从“固定格式 rank”泛化为可传入的候选顺序；
- 将候选唯一标识从仅支持 `format` 泛化为支持 `candidate_id`；
- `SheetFlex-vote` 继续默认使用现有六格式顺序；
- Self-Consistency 使用 `sample_index` 顺序；
- 保证所有现有 `SheetFlex-vote` 测试继续通过。

建议新增：

```text
self_consistency.py
configs/self_consistency/
sc_scripts/                 # 或与当前 vote_scripts 并列的清晰目录
```

不需要建立一套重复的 `core/self_consistency` 投票实现；优先复用或轻量泛化 `core/sheetflex` 中已有纯函数。

---

## 8. 输出

### 8.1 RealHiTBench

建议输出：

```text
self_consistency.jsonl
self_consistency_eval.json
self_consistency_score.json
self_consistency_diagnostics.json
```

### 8.2 SpreadsheetBench

建议输出：

```text
self_consistency.jsonl
spreadsheet/                         # 选中的完整候选文件
spreadsheet_pot_eval.json
spreadsheet_pot_accuracy.json
self_consistency_diagnostics.json
```

逐样例 trace 至少包含：

```text
sample_id
table_format
num_samples
base_seed
candidate_id
sample_index
seed
source_run_dir
candidate_valid
invalid_reason
sequence_logprob_sum
normalized_answer 或 region_hash
aggregation_score
selected
tie_occurred
tie_break_source
```

---

## 9. 诊断统计

至少统计：

- 有效候选数分布；
- 全部候选无效的样例数；
- 平票率；
- 使用 log-probability 解决平票的比例；
- 固定 `sample_index` fallback 次数；
- 最终选择的 sample index 分布；
- 实际生成 attempt 数；
- RealHiTBench 每题唯一规范化答案数量；
- SpreadsheetBench 每题唯一关键区域 hash 数量；
- 六条候选完全一致的样例比例。

低温度下六条候选完全一致是允许的实验结果，不能为了制造多样性而改变参数或重采样。

---

## 10. 验收标准

1. Self-Consistency 默认配置严格为 `n=6`、`temperature=0.1`、`top_p=1.0`、`base_seed=42`。
2. 六条候选来自同一种表格格式。
3. 每条候选是一个独立完整 solver 轨迹，内部重试不增加票数。
4. RealHiTBench 的普通样例和 Structure Comprehending 均正确聚合。
5. SpreadsheetBench 只选择已有工作簿，不合成新文件。
6. 两个数据集的投票和平票语义与 `SheetFlex-vote` 一致。
7. 聚合阶段不访问 gold/golden。
8. 原有 `SheetFlex-vote` 行为和测试不受影响。
9. 支持小样例 smoke test、断点恢复和离线重新聚合。
10. coding agent 最终提供候选生成、聚合、评测和诊断的完整运行命令。

---

## 11. 暂不实现

- `SheetFlex-agg` 的自验证置信度；
- 固定格式权重；
- 自动选择最佳单格式；
- 单请求 `n=6` 的多 choice 执行；
- 多种格式混合的 Self-Consistency；
- 逐单元格合成工作簿。
