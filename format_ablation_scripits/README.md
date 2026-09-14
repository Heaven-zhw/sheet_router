# Four-Format Ablation

These scripts use existing offline candidates only. They remove `image` and
`excel_1_image` from cross-format aggregation, and use only seeds `42, 43, 44,
45` for one-format Self-Consistency.

The retained formats are:

```text
json_cells > latex > json_rows > markdown
```

Outputs go under `format_ablation_outs/` by default. Existing `outs/`,
`sc_outs/`, and `lp_outs/` are never overwritten.

## Self-Consistency

```bash
bash format_ablation_scripits/aggregate_self_consistency_4formats.sh
```

This runs all three currently configured models, both benchmarks, and all four
retained formats. Each output manifest records `seeds=[42,43,44,45]` and
`num_candidates=4`.

## SheetFlex-vote

```bash
bash format_ablation_scripits/aggregate_sheetflex_vote_4formats.sh
```

The default is mean-logprob tie breaking:

```bash
bash format_ablation_scripits/aggregate_sheetflex_vote_4formats.sh
```

The fixed-recommend variant is:

```bash
bash format_ablation_scripits/aggregate_sheetflex_vote_fixed_4formats.sh
```

## SheetFlex-LPVote

```bash
bash format_ablation_scripits/aggregate_sheetflex_lpvote_4formats.sh
```

Override `LP_WEIGHT_STRENGTH`, `MISSING_LOGPROB_POLICY`, or `TIE_BREAK_ORDER`
with environment variables.

## SheetFlex-CGVote

```bash
bash format_ablation_scripits/aggregate_sheetflex_cgvote_4formats.sh
```

Override `LP_WEIGHT_STRENGTH` and `CONFIDENCE_GATE_STRENGTH` for alpha/beta
ablations. All scripts accept `OUTPUT_ROOT`, `IDS`, `LIMIT`, and `RESUME`.

The Python adapter at `format_ablation.py` performs the subset selection and
reuses the existing aggregation/evaluation code. It does not call a model or
execute candidate code.
