# SheetFlex-CGVote Implementation

This note records implementation facts that supplement, but do not replace,
`SheetFlex_Overall_Design_v3.md`.

## Components

```text
core/sheetflex/cg_vote.py
    Pure weighting, equivalence-class scoring, task adapters, and diagnostics.

sheetflex_cg_vote.py
    Offline RealHiTBench/SpreadsheetBench CLI, manifest, output, copy, and eval.

analysis/check_sheetflex_cg_vote_regression.py
    Per-sample alpha=0 and beta=0 compatibility checks.

tests/test_sheetflex_cg_vote.py
    Synthetic formula, compatibility, safety, and workbook tests.
```

Spreadsheet aggregation enters through the existing
`aggregate_spreadsheet_sample(selection_fn=..., selection_kwargs=...)` hook.
Candidate validation, shared open-range boundaries, region extraction,
`region_hash`, candidate similarity, output paths, copy, and evaluation are not
duplicated.

## Formula Mapping

`compute_lp_weights` supplies candidate weights:

```text
p_i = softmax(alpha * sequence_logprob_mean_i)
```

`build_equivalence_classes` computes:

```text
W_G = sum_{i in G} p_i
Q_G = max_{i in G} p_i
```

`score_equivalence_classes` computes:

```text
C_G = sum_H W_H * K(G, H)
CGScore(G) = C_G * Q_G ** beta
```

RealHiT classes use the existing normalized answer. Spreadsheet classes use the
existing target-region hash and retain self-support in `K`.

Normal CG scores are compared in the
`max_shifted_exp_log_confidence_gated_score` space. The best score is 1 and all
other scores are `exp(log_score - best_log_score)`. The comparison uses the
repository's existing `math.isclose` tolerances:

```text
rel_tol = 1e-12
abs_tol = 1e-12
```

This avoids raw-score underflow and avoids applying absolute tolerance directly
to large negative log scores. `alpha=0` and `beta=0` compare consensus scores
directly to preserve baseline selections exactly.

## Missing Logprob Compatibility

CGVote checks every valid candidate before calling `compute_lp_weights`.

- `vote`: any missing/invalid mean causes the whole sample to use equal weights
  and fixed-recommend selection. A lone valid candidate is still selected and
  the fallback is recorded.
- `error`: any missing/invalid mean raises an error containing sample ID,
  branch, and format, including the lone-valid-candidate case.

For RealHiT Structure Comprehending, reference and swap use separate valid sets
and separate softmaxes. If either side needs the `vote` fallback, both sides use
fixed-recommend. This preserves the existing LPVote whole-question fallback.

The existing `compute_lp_weights` helper intentionally remains unchanged: it
accepts a lone valid candidate without a mean before applying `error`. CGVote's
stricter precheck is a documented compatibility difference on invalid input.

## Selection Boundaries

The selector consumes only dataset metadata, input workbooks, candidate output
workbooks, and candidate generation logprob summaries. It does not consume
golden workbooks, answers, test results, accuracy files, or candidate correctness.

Spreadsheet candidates with the same `region_hash` form one prediction class.
After class selection, the representative workbook is chosen only by the
requested stable format order. No formula, style, non-target-area, or workbook
integrity reranking is implemented.

The CLI saves `sheetflex_cg_vote.jsonl` before final evaluation. Spreadsheet
files are copied byte-for-byte through the existing copy helper and are never
reopened and saved by CGVote.

## Historical Candidate Metadata

The available `lp_outs` run directories contain `realhit_cot.jsonl` or
`spreadsheet_pot.jsonl` and valid logprob summaries. They do not contain
`run_metadata.json`, so `temperature=0` and `top_p=1` cannot be independently
verified from those run directories. CGVote reports this limitation rather than
inferring metadata from directory names. The output manifest records
`candidate_generation_config.status=unverified_missing_run_metadata`. When all
candidate metadata files are present, the CLI validates temperature, top-p,
format, and cross-format model identity before aggregation. Independently, the
CLI validates the per-result `table_metadata.table_format` and `token_model`
fields when present and records the observed identity in the manifest.

## Commands

```bash
python sheetflex_cg_vote.py realhit \
  --run_map configs/sheetflex/logprobs_generated_maps/Qwen3.5-9B_realhit.json \
  --output_dir cg_outs/realhitbench/Qwen3.5-9B/a1_b1_recommend \
  --lp_weight_strength 1 \
  --confidence_gate_strength 1 \
  --missing_logprob_policy error \
  --tie_break_order recommend

python sheetflex_cg_vote.py spreadsheet \
  --run_map configs/sheetflex/logprobs_generated_maps/Qwen3.5-9B_spreadsheet.json \
  --output_dir cg_outs/spreadsheetbench_verified_400/Qwen3.5-9B/a1_b1_recommend \
  --lp_weight_strength 1 \
  --confidence_gate_strength 1 \
  --missing_logprob_policy error \
  --tie_break_order recommend

python analysis/check_sheetflex_cg_vote_regression.py realhit \
  --run_map configs/sheetflex/logprobs_generated_maps/Qwen3.5-9B_realhit.json

python analysis/check_sheetflex_cg_vote_regression.py spreadsheet \
  --run_map configs/sheetflex/logprobs_generated_maps/Qwen3.5-9B_spreadsheet.json
```

Use distinct, empty output directories for different alpha/beta settings.
