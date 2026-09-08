# SheetFlex Format Complementarity Analysis

This report mirrors FLEXTAF Figure 3 and Table 4 for the six SheetFlex candidate representations.

## Definitions

- RealHiTBench correctness: exact match (`EM = 100`).
- SpreadsheetBench correctness: every test restriction passes (`total_hard_restriction = 1`).
- Conditional overlap at row `r`, column `c`: `|S_r intersect S_c| / |S_c|`. This is asymmetric and matches FLEXTAF Figure 3.
- Exclusive rate for format `f`: instances solved only by `f`, divided by all instances solved by `f`. This matches FLEXTAF Table 4.
- Oracle union: an instance is correct if at least one format is correct. It is an upper bound on selecting one existing candidate with gold knowledge.

## Coverage And Aggregation

| Dataset | Model | N | None | Exactly 1 | 2 to 5 | All 6 | Best fixed | SheetFlex | Oracle | Gain vs best | Rescue / regress | McNemar p |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| RealHiTBench | Qwen3.5-9B | 2384 | 387 (16.2%) | 183 (7.7%) | 988 (41.4%) | 826 (34.6%) | Excel-Image: 66.1% | 72.3% | 83.8% | 6.2% | 251 / 103 | 2.18e-15 |
| RealHiTBench | Qwen3-VL-30B-A3B-Instruct | 2384 | 455 (19.1%) | 205 (8.6%) | 890 (37.3%) | 834 (35.0%) | LaTeX: 63.7% | 67.8% | 80.9% | 4.1% | 181 / 84 | 2.48e-09 |
| RealHiTBench | gemma-3-12b-it | 2384 | 759 (31.8%) | 342 (14.3%) | 1103 (46.3%) | 180 (7.6%) | LaTeX: 49.4% | 48.3% | 68.2% | -1.1% | 172 / 198 | 0.194 |
| RealHiTBench | gemma-4-12B-it | 2384 | 422 (17.7%) | 121 (5.1%) | 1239 (52.0%) | 602 (25.3%) | LaTeX: 72.3% | 74.5% | 82.3% | 2.2% | 108 / 56 | 5.98e-05 |
| RealHiTBench | gemma-4-26B-A4B-it | 2384 | 419 (17.6%) | 107 (4.5%) | 1126 (47.2%) | 732 (30.7%) | LaTeX: 73.5% | 76.1% | 82.4% | 2.6% | 99 / 36 | 5.5e-08 |
| SpreadsheetBench | Qwen3.5-9B | 400 | 168 (42.0%) | 83 (20.8%) | 140 (35.0%) | 9 (2.2%) | JSON-Cell: 28.7% | 34.5% | 58.0% | 5.8% | 43 / 20 | 0.00515 |
| SpreadsheetBench | Qwen3-VL-30B-A3B-Instruct | 400 | 205 (51.2%) | 53 (13.2%) | 129 (32.2%) | 13 (3.2%) | JSON-Cell: 29.5% | 34.8% | 48.8% | 5.2% | 40 / 19 | 0.00864 |
| SpreadsheetBench | gemma-3-12b-it | 400 | 277 (69.2%) | 44 (11.0%) | 68 (17.0%) | 11 (2.8%) | JSON-Cell: 17.2% | 19.8% | 30.8% | 2.5% | 24 / 14 | 0.143 |
| SpreadsheetBench | gemma-4-12B-it | 400 | 85 (21.2%) | 36 (9.0%) | 178 (44.5%) | 101 (25.2%) | JSON-Cell: 62.0% | 64.5% | 78.8% | 2.5% | 27 / 17 | 0.174 |
| SpreadsheetBench | gemma-4-26B-A4B-it | 400 | 65 (16.2%) | 33 (8.2%) | 167 (41.8%) | 135 (33.8%) | JSON-Cell: 67.2% | 69.5% | 83.8% | 2.2% | 26 / 17 | 0.222 |

`McNemar p` is the two-sided exact paired test using SheetFlex rescues and regressions relative to the observed best fixed format. Treat it as exploratory because that baseline is selected on the same evaluation set; a confirmatory test should preselect the baseline on a development split.

## FLEXTAF-Style Exclusive Rates

Each cell is `exclusive count / all correct count for that format (rate)`.

| Dataset | Model | LaTeX | Markdown | JSON-Cell | JSON-Row | Image | Excel-Image |
|---|---|---:|---:|---:|---:|---:|---:|
| RealHiTBench | Qwen3.5-9B | 23 / 1486 (1.5%) | 15 / 1448 (1.0%) | 27 / 1532 (1.8%) | 19 / 1489 (1.3%) | 56 / 1529 (3.7%) | 43 / 1576 (2.7%) |
| RealHiTBench | Qwen3-VL-30B-A3B-Instruct | 29 / 1519 (1.9%) | 23 / 1470 (1.6%) | 28 / 1503 (1.9%) | 35 / 1499 (2.3%) | 58 / 1392 (4.2%) | 32 / 1298 (2.5%) |
| RealHiTBench | gemma-3-12b-it | 103 / 1177 (8.8%) | 41 / 969 (4.2%) | 46 / 912 (5.0%) | 46 / 969 (4.7%) | 69 / 731 (9.4%) | 37 / 492 (7.5%) |
| RealHiTBench | gemma-4-12B-it | 20 / 1723 (1.2%) | 17 / 1664 (1.0%) | 23 / 1693 (1.4%) | 16 / 1652 (1.0%) | 31 / 1011 (3.1%) | 14 / 881 (1.6%) |
| RealHiTBench | gemma-4-26B-A4B-it | 18 / 1752 (1.0%) | 13 / 1736 (0.7%) | 19 / 1750 (1.1%) | 5 / 1716 (0.3%) | 39 / 1178 (3.3%) | 13 / 1025 (1.3%) |
| SpreadsheetBench | Qwen3.5-9B | 18 / 95 (18.9%) | 11 / 90 (12.2%) | 22 / 115 (19.1%) | 21 / 102 (20.6%) | 4 / 61 (6.6%) | 7 / 80 (8.8%) |
| SpreadsheetBench | Qwen3-VL-30B-A3B-Instruct | 11 / 100 (11.0%) | 6 / 84 (7.1%) | 21 / 118 (17.8%) | 7 / 110 (6.4%) | 2 / 61 (3.3%) | 6 / 63 (9.5%) |
| SpreadsheetBench | gemma-3-12b-it | 7 / 55 (12.7%) | 4 / 49 (8.2%) | 18 / 69 (26.1%) | 5 / 47 (10.6%) | 5 / 46 (10.9%) | 5 / 51 (9.8%) |
| SpreadsheetBench | gemma-4-12B-it | 4 / 224 (1.8%) | 2 / 226 (0.9%) | 10 / 248 (4.0%) | 11 / 235 (4.7%) | 5 / 167 (3.0%) | 4 / 196 (2.0%) |
| SpreadsheetBench | gemma-4-26B-A4B-it | 5 / 252 (2.0%) | 9 / 267 (3.4%) | 11 / 269 (4.1%) | 4 / 256 (1.6%) | 1 / 200 (0.5%) | 3 / 228 (1.3%) |

## Conditional Overlap Matrices: Qwen3.5-9B

Rows are the formats that also solve an instance; columns are the conditioning correct sets. Values are asymmetric `P(row correct | column correct)`.

### RealHiTBench

| Also solved by / conditioned on | LaTeX | Markdown | JSON-Cell | JSON-Row | Image | Excel-Image |
|---|---:|---:|---:|---:|---:|---:|
| LaTeX | 1.00 | 0.86 | 0.82 | 0.83 | 0.80 | 0.80 |
| Markdown | 0.84 | 1.00 | 0.81 | 0.83 | 0.78 | 0.79 |
| JSON-Cell | 0.84 | 0.86 | 1.00 | 0.87 | 0.82 | 0.82 |
| JSON-Row | 0.84 | 0.85 | 0.85 | 1.00 | 0.80 | 0.81 |
| Image | 0.82 | 0.82 | 0.82 | 0.82 | 1.00 | 0.83 |
| Excel-Image | 0.85 | 0.86 | 0.85 | 0.86 | 0.85 | 1.00 |

### SpreadsheetBench

| Also solved by / conditioned on | LaTeX | Markdown | JSON-Cell | JSON-Row | Image | Excel-Image |
|---|---:|---:|---:|---:|---:|---:|
| LaTeX | 1.00 | 0.48 | 0.42 | 0.36 | 0.49 | 0.51 |
| Markdown | 0.45 | 1.00 | 0.45 | 0.45 | 0.46 | 0.42 |
| JSON-Cell | 0.51 | 0.58 | 1.00 | 0.53 | 0.61 | 0.53 |
| JSON-Row | 0.39 | 0.51 | 0.47 | 1.00 | 0.43 | 0.42 |
| Image | 0.32 | 0.31 | 0.32 | 0.25 | 1.00 | 0.45 |
| Excel-Image | 0.43 | 0.38 | 0.37 | 0.33 | 0.59 | 1.00 |

## Key Findings

1. **RealHiTBench:** across 5 models (11920 model-instance pairs), 958 (8.0%) pairs are solved by exactly one format, 5346 (44.8%) by two to five formats, and 3174 (26.6%) by all six.
2. **RealHiTBench aggregation:** macro accuracy is 67.8% for SheetFlex, versus 65.0% for the per-model best fixed format and 79.5% for the oracle union.
3. **SpreadsheetBench:** across 5 models (2000 model-instance pairs), 249 (12.4%) pairs are solved by exactly one format, 682 (34.1%) by two to five formats, and 269 (13.5%) by all six.
4. **SpreadsheetBench aggregation:** macro accuracy is 44.6% for SheetFlex, versus 40.9% for the per-model best fixed format and 60.0% for the oracle union.
5. **Unique contributions:** pooled across all dataset-model evaluations, Image contributes 270 exclusive cases, LaTeX contributes 238 exclusive cases, JSON-Cell contributes 225 exclusive cases, JSON-Row contributes 169 exclusive cases, Excel-Image contributes 164 exclusive cases, Markdown contributes 141 exclusive cases.
6. **Interpretation:** unique and partial coverage establishes representation complementarity and oracle headroom. The SheetFlex-vs-best-fixed comparison measures whether the current aggregation rule converts that headroom into realized accuracy; complementarity alone is not proof that every aggregation rule helps.

## Output Files

- `summary.csv`: best fixed, SheetFlex, oracle, and coverage headline numbers.
- `correct_format_count_distribution.csv`: exact `k = 0..6` counts.
- `format_exclusive_rates.csv`: FLEXTAF Table 4 analogue.
- `pairwise_overlap.csv`: conditional overlap and symmetric Jaccard values.
- `format_combination_distribution.csv`: counts for every exact subset of correct formats.
- `per_sample_correct_formats.csv`: auditable sample-level correctness matrix.
- `coverage_by_task_type.csv`: task-type breakdown.
- `analysis.json`: full matrices, tables, source paths, and sample-level records.
- `figures/`: paper-ready PNG and PDF plots.

Output directory: `/mnt/data/zhw/sheet_router/analysis/format_overlap`
