# 五格式 SheetFlex 与五 seed Self-Consistency

只读取已有候选进行离线聚合、复制选中工作簿和评测。默认四个模型：
`gemma-3-12b-it gemma-4-12B-it Qwen3.5-9B Qwen3-VL-30B-A3B-Instruct`。

| 方法 | RealHiTBench | SpreadsheetBench |
|---|---|---|
| SC | 六格式各自聚合五个 seed | 六格式各自聚合五个 seed |
| vote mean/fixed、LPVote、CGVote | 四文本格式 + image | 四文本格式 + excel_1_image |

SC 六格式为 `latex markdown json_rows json_cells image excel_1_image`。
两个图像格式都已经融入常规 SC；不再有 extra_formats 独立脚本。
SC 每次是同格式五条轨迹投票，不是六格式混合投票。

## 本次补跑命令

```bash
conda activate spreadsheetagent
cd /mnt/data/zhw/sheet_router

# 四模型、两数据集：LPVote alpha=1，CGVote alpha=1 beta=1
bash format_ablation_scripits_5formats/aggregate_sheetflex_lpvote_5formats.sh
bash format_ablation_scripits_5formats/aggregate_sheetflex_cgvote_5formats.sh

# gemma3、Qwen3-VL：两数据集、六格式，默认 seeds 42–46
MODEL_NAMES='gemma-3-12b-it Qwen3-VL-30B-A3B-Instruct' \
bash format_ablation_scripits_5formats/aggregate_self_consistency_5formats.sh

# 同两模型：其余五种组合，两个数据集都运行六格式 SC
MODEL_NAMES='gemma-3-12b-it Qwen3-VL-30B-A3B-Instruct' \
bash format_ablation_scripits_5formats/aggregate_self_consistency_5formats_seed_combinations.sh
```

## 输出及重复运行

输出位置沿用原实验目录，脚本搬迁不移动结果：

```text
format_ablation_outs_5formats/
  self_consistency/{realhitbench,spreadsheetbench_verified_400}/<model>/<format>/
  {vote_mean,vote_fixed,lpvote,cgvote}/{realhit,spreadsheet}/<model>/
format_ablation_outs_5formats_seed_combinations/
  seed_<组合>/self_consistency/{realhitbench,spreadsheetbench_verified_400}/<model>/<format>/
```

默认 `SKIP_COMPLETED=1 RESUME=1`：配置、来源和 ID 覆盖一致，且评测、诊断及成功样例
工作簿均存在时跳过。缺失产物的任务重新计算该模型/格式/组合，完成后生成评测。
不会仅凭目录或 score 文件存在就认定完成。配置/种子/来源/样例范围不同则报错，
需换输出目录。当前 resume 是整组重算，不是逐题断点续跑。

## 选择任务及预览

```bash
# 只预览，不写结果；输出 run/skip/resume
DRY_RUN=1 MODEL_NAMES='gemma-3-12b-it Qwen3-VL-30B-A3B-Instruct' \
bash format_ablation_scripits_5formats/aggregate_self_consistency_5formats.sh

# 只跑某数据集和某种子组合；组合内 seed 按升序
BENCHMARKS=realhit MODEL_NAMES=Qwen3-VL-30B-A3B-Instruct \
SEED_COMBINATIONS='43,44,45,46,47' \
bash format_ablation_scripits_5formats/aggregate_self_consistency_5formats_seed_combinations.sh

# 同样支持原来的 vote 方法
bash format_ablation_scripits_5formats/aggregate_sheetflex_vote_5formats.sh
bash format_ablation_scripits_5formats/aggregate_sheetflex_vote_fixed_5formats.sh
```

`MODEL_NAMES`、`BENCHMARKS`、`SC_FORMATS` 都是空格分隔；默认包含四模型、两数据集、
六个 SC 格式。`SC_FORMATS` 不改变 SheetFlex 的跨格式候选集。
`SEED_COMBINATIONS` 是分号分隔的组合，默认其余五种；`SELF_CONSISTENCY_SEEDS` 是
普通 SC 的种子列表，默认 `42,43,44,45,46`。其他组合脚本输出在 `OUTPUT_ROOT_BASE`。

平票默认 mean/recommend；支持 `TIE_BREAK_LOGPROB=mean|sum`、
`TIE_BREAK_ORDER=recommend|legacy`、`LP_WEIGHT_STRENGTH`、`CONFIDENCE_GATE_STRENGTH`、
`MISSING_LOGPROB_POLICY=error|vote`。变更策略或超参时请设置新的 `OUTPUT_ROOT`。
`CANDIDATE_ROOT`、`SOURCE_ROOT`、`DATASET_ROOT`、`IDS`、`LIMIT` 保持可用。
小样本测试也需独立输出目录，防止和全量范围混用。不启用 no-op。
