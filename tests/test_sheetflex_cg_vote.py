import copy
import math
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import openpyxl

import sheetflex_cg_vote as cg_cli
from core.sheetflex.cg_vote import (
    aggregate_answer_cg_vote,
    aggregate_realhit_cg_sample,
    aggregate_spreadsheet_cg_sample,
    build_equivalence_classes,
    check_cg_logprobs,
    score_equivalence_classes,
    select_spreadsheet_cg_candidate,
    spreadsheet_cg_diagnostics,
    validate_confidence_gate_strength,
)
from core.sheetflex.common import (
    LEGACY_FORMAT_ORDER,
    RECOMMEND_FORMAT_ORDER,
    SheetFlexError,
    index_rows_by_id,
)
from core.client import ClientJupyterKernel
from core.sheetflex.lp_vote import (
    aggregate_answer_lp_vote,
    select_spreadsheet_lp_candidate,
)
from core.sheetflex.realhit import aggregate_answer_vote
from core.sheetflex.spreadsheet import (
    copy_selected_workbooks,
    select_spreadsheet_medoid,
)


def answer_candidate(format_name, probability, answer, *, valid=True):
    return {
        "format": format_name,
        "valid": valid,
        "invalid_reason": None if valid else "invalid",
        "model_answer": answer,
        "normalized_answer": answer.lower() if valid else None,
        "aggregation_score": None,
        "selected": False,
        "logprob_available": probability is not None,
        "sequence_logprob_mean": (
            math.log(probability) if probability is not None else None
        ),
        "sequence_logprob_sum": -999.0 if probability is not None else None,
    }


def cell_map(left, right):
    return OrderedDict(
        [(('Target', 'A1'), left), (('Target', 'B1'), right)]
    )


def spreadsheet_candidate(format_name, probability, values, region_hash=None):
    candidate = answer_candidate(format_name, probability, "unused")
    candidate.update(
        {
            "output_file": f"{format_name}.xlsx",
            "region_hash": region_hash or format_name,
            "_cells": cell_map(*values),
        }
    )
    return candidate


class CGVotePureFunctionTest(unittest.TestCase):
    def test_manual_realhit_gate_changes_lpvote_winner(self):
        candidates = [
            answer_candidate("latex", 0.275, "A"),
            answer_candidate("markdown", 0.275, "A"),
            answer_candidate("json_cells", 0.45, "B"),
        ]
        lp = aggregate_answer_cg_vote(
            candidates, strength=1.0, confidence_gate_strength=0.0
        )
        cg = aggregate_answer_cg_vote(
            candidates, strength=1.0, confidence_gate_strength=1.0
        )
        self.assertEqual(lp["selected_answer"], "A")
        self.assertEqual(cg["selected_answer"], "B")
        classes = {item["class_id"]: item for item in cg["equivalence_classes"]}
        self.assertAlmostEqual(classes["a"]["class_weight_mass"], 0.55)
        self.assertAlmostEqual(classes["a"]["class_confidence"], 0.275)
        self.assertAlmostEqual(classes["a"]["confidence_gated_score"], 0.15125)
        self.assertAlmostEqual(classes["b"]["confidence_gated_score"], 0.2025)

    def test_manual_spreadsheet_matrix(self):
        candidates = [
            spreadsheet_candidate("latex", 0.2, (0, 0), "A"),
            spreadsheet_candidate("markdown", 0.3, (0, 1), "B"),
            spreadsheet_candidate("json_cells", 0.5, (1, 1), "C"),
        ]
        result = select_spreadsheet_cg_candidate(candidates)
        classes = {
            item["class_id"]: item for item in result["equivalence_classes"]
        }
        self.assertAlmostEqual(classes["A"]["consensus_score"], 0.35)
        self.assertAlmostEqual(classes["B"]["consensus_score"], 0.65)
        self.assertAlmostEqual(classes["C"]["consensus_score"], 0.65)
        self.assertAlmostEqual(classes["A"]["confidence_gated_score"], 0.07)
        self.assertAlmostEqual(classes["B"]["confidence_gated_score"], 0.195)
        self.assertAlmostEqual(classes["C"]["confidence_gated_score"], 0.325)
        self.assertEqual(result["selected_class_id"], "C")

    def test_class_weight_mass_and_confidence_are_not_averages(self):
        values = [
            {**answer_candidate("latex", 0.2, "A"), "lp_weight": 0.2},
            {**answer_candidate("markdown", 0.3, "A"), "lp_weight": 0.3},
            {**answer_candidate("json_cells", 0.5, "B"), "lp_weight": 0.5},
        ]
        _, classes = build_equivalence_classes(
            values, class_key="normalized_answer"
        )
        by_id = {item["class_id"]: item for item in classes}
        self.assertEqual(by_id["a"]["class_weight_mass"], 0.5)
        self.assertEqual(by_id["a"]["class_confidence"], 0.3)

    def test_invalid_alpha_beta_and_boolean_beta(self):
        for value in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(SheetFlexError, ">= 0"):
                    validate_confidence_gate_strength(value)
        values = [answer_candidate("latex", 1.0, "A")]
        for alpha, beta in ((-1.0, 1.0), (1.0, -1.0)):
            with self.subTest(alpha=alpha, beta=beta):
                with self.assertRaises(SheetFlexError):
                    aggregate_answer_cg_vote(
                        values,
                        strength=alpha,
                        confidence_gate_strength=beta,
                    )

    def test_extreme_means_do_not_create_nan(self):
        values = [
            answer_candidate("latex", 1.0, "A"),
            answer_candidate("json_cells", 1.0, "B"),
        ]
        values[0]["sequence_logprob_mean"] = -1e300
        values[1]["sequence_logprob_mean"] = 1e300
        result = aggregate_answer_cg_vote(
            values, strength=10.0, confidence_gate_strength=2.0
        )
        for output_class in result["equivalence_classes"]:
            self.assertFalse(math.isnan(output_class["confidence_gated_score"]))
        self.assertEqual(result["selected_answer"], "B")

    def test_single_valid_missing_mean_is_strict_in_error_mode(self):
        values = [answer_candidate("latex", None, "A")]
        with self.assertRaisesRegex(
            SheetFlexError, "sample_id=s branch=ordinary.*latex"
        ):
            check_cg_logprobs(
                values,
                "error",
                sample_id="s",
                branch="ordinary",
            )
        selected = aggregate_answer_cg_vote(
            values,
            missing_logprob_policy="vote",
            sample_id="s",
        )
        self.assertEqual(selected["selected_answer"], "A")
        self.assertTrue(selected["fallback"])
        self.assertEqual(selected["candidates"][0]["lp_weight"], 1.0)

    def test_all_invalid_is_not_an_error(self):
        values = [
            answer_candidate(name, None, "", valid=False)
            for name in RECOMMEND_FORMAT_ORDER
        ]
        result = aggregate_answer_cg_vote(
            values, missing_logprob_policy="error", sample_id="s"
        )
        self.assertFalse(result["format_valid"])
        self.assertIsNone(result["selected_class_id"])

    def test_missing_format_candidate_is_retained_as_invalid(self):
        result = aggregate_realhit_cg_sample(
            "s",
            "Fact Checking",
            {
                "latex": {
                    "format_valid": True,
                    "model_answer": "A",
                    "logprob_available": True,
                    "sequence_logprob_mean": -1.0,
                }
            },
        )
        traces = {
            item["format"]: item for item in result["trace"]["answer_vote"]["candidates"]
        }
        self.assertEqual(len(traces), 6)
        self.assertFalse(traces["json_cells"]["valid"])
        self.assertEqual(
            traces["json_cells"]["invalid_reason"],
            "sample_id_missing_from_run_results",
        )

    def test_stringified_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(SheetFlexError, "Duplicate id"):
            index_rows_by_id([{"id": 1}, {"id": "1"}], source="cg-test")

    def test_beta_zero_matches_lpvote_and_alpha_zero_matches_fixed_vote(self):
        values = [
            answer_candidate("latex", 0.2, "A"),
            answer_candidate("markdown", 0.3, "A"),
            answer_candidate("json_cells", 0.5, "B"),
        ]
        rank = lambda item: RECOMMEND_FORMAT_ORDER.index(item["format"])
        lp = aggregate_answer_lp_vote(values, rank_getter=rank)
        beta_zero = aggregate_answer_cg_vote(
            values, confidence_gate_strength=0.0, rank_getter=rank
        )
        fixed = aggregate_answer_vote(
            values, rank_getter=rank, logprob_field=None
        )
        alpha_zero = aggregate_answer_cg_vote(
            values,
            strength=0.0,
            confidence_gate_strength=2.0,
            rank_getter=rank,
        )
        self.assertEqual(beta_zero["selected_answer"], lp["selected_answer"])
        self.assertEqual(beta_zero["selected_format"], lp["selected_format"])
        self.assertEqual(alpha_zero["selected_answer"], fixed["selected_answer"])
        self.assertEqual(alpha_zero["selected_format"], fixed["selected_format"])

    def test_tiny_but_distinct_scores_are_not_false_ties(self):
        classes = [
            {
                "class_id": "A",
                "class_weight_mass": 0.5,
                "class_confidence": 1e-100,
                "rank": 1,
                "selected": False,
            },
            {
                "class_id": "B",
                "class_weight_mass": 0.5,
                "class_confidence": 2e-100,
                "rank": 0,
                "selected": False,
            },
        ]
        matrix = {"A": {"A": 1.0, "B": 0.0}, "B": {"A": 0.0, "B": 1.0}}
        scored = score_equivalence_classes(classes, matrix, 2.0)
        from core.sheetflex.cg_vote import select_confidence_gated_class

        selected, tied, space = select_confidence_gated_class(
            scored, alpha=1.0, beta=2.0
        )
        self.assertEqual(selected["class_id"], "B")
        self.assertEqual(sum(item["selected"] for item in tied), 1)
        self.assertIn("log_confidence", space)


class RealHiTCGVoteTest(unittest.TestCase):
    @staticmethod
    def branch(answer, probability, valid=True):
        return {
            "format_valid": valid,
            "model_answer": answer,
            "logprob_available": probability is not None,
            "sequence_logprob_mean": (
                math.log(probability) if probability is not None else None
            ),
        }

    def test_structure_uses_nested_values_and_independent_valid_sets(self):
        records = {
            "latex": {
                "model_answer": "wrong-top-level",
                "sequence_logprob_mean": 999.0,
                "structure_reference_run": self.branch("R1", 0.8),
                "structure_swap_run": self.branch("S1", 0.2),
            },
            "markdown": {
                "structure_reference_run": self.branch("R2", 0.2),
                "structure_swap_run": self.branch("S2", 0.8),
            },
            "json_cells": {
                "structure_reference_run": self.branch("", 0.5, False),
                "structure_swap_run": self.branch("S2", 0.5),
            },
            **{name: None for name in ("json_rows", "image", "excel_1_image")},
        }
        result = aggregate_realhit_cg_sample(
            "s", "Structure Comprehending", records
        )
        self.assertEqual(result["structure_reference_answer"], "R1")
        self.assertEqual(result["structure_swap_answer"], "S2")
        self.assertEqual(
            result["branch_valid_candidate_count"], {"reference": 2, "swap": 3}
        )
        self.assertNotEqual(result["model_answer"], "wrong-top-level")

    def test_structure_missing_on_one_side_falls_back_both_sides(self):
        records = {
            "latex": {
                "structure_reference_run": self.branch("R1", None),
                "structure_swap_run": self.branch("S1", 0.8),
            },
            "json_cells": {
                "structure_reference_run": self.branch("R2", 0.2),
                "structure_swap_run": self.branch("S2", 0.2),
            },
            **{name: None for name in ("markdown", "json_rows", "image", "excel_1_image")},
        }
        result = aggregate_realhit_cg_sample(
            "s", "Structure Comprehending", records
        )
        self.assertTrue(result["trace"]["fallback"])
        for key in ("structure_reference_vote", "structure_swap_vote"):
            branch = result["trace"][key]
            self.assertTrue(branch["fallback"])
            self.assertEqual(branch["tie_break_order"], "recommend")


def save_book(path, values):
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Target"
    worksheet["A1"] = values[0]
    worksheet["B1"] = values[1]
    workbook.save(path)
    workbook.close()


class SpreadsheetCGVoteTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.xlsx"
        save_book(self.input_path, (0, 0))
        self.run_dirs = {}
        for format_name in RECOMMEND_FORMAT_ORDER:
            run_dir = self.root / format_name
            (run_dir / "spreadsheet").mkdir(parents=True)
            self.run_dirs[format_name] = run_dir
        self.item = {
            "id": "sample",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1:B1",
            "answer_sheet": "Target",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, format_name, probability, values=None, success=True):
        if values is not None:
            save_book(
                self.run_dirs[format_name]
                / "spreadsheet"
                / "1_sample_output.xlsx",
                values,
            )
        return {
            "id": "sample",
            "output_file": "1_sample_output.xlsx",
            "execution_success": success,
            "logprob_available": probability is not None,
            "sequence_logprob_mean": (
                math.log(probability) if probability is not None else None
            ),
            "test_case_results": [1],
            "total_hard_restriction": 1,
        }

    def records(self):
        return {
            name: self.record(name, None, success=False)
            for name in RECOMMEND_FORMAT_ORDER
        }

    def test_same_hash_members_preserve_mass_and_use_order_for_file(self):
        candidates = [
            spreadsheet_candidate("latex", 0.8, (1, 1), "same"),
            spreadsheet_candidate("json_cells", 0.2, (1, 1), "same"),
        ]
        result = select_spreadsheet_cg_candidate(candidates)
        self.assertEqual(result["equivalence_class_count"], 1)
        self.assertAlmostEqual(
            result["equivalence_classes"][0]["class_weight_mass"], 1.0
        )
        self.assertAlmostEqual(
            result["equivalence_classes"][0]["class_confidence"], 0.8
        )
        self.assertEqual(result["selected_format"], "json_cells")
        self.assertTrue(result["equivalent_class_representative_selection"])
        self.assertFalse(result["tie"])

    def test_beta_zero_matches_lpvote_and_alpha_zero_matches_fixed(self):
        values = [
            spreadsheet_candidate("latex", 0.2, (0, 0)),
            spreadsheet_candidate("markdown", 0.3, (0, 1)),
            spreadsheet_candidate("json_cells", 0.5, (1, 1)),
        ]
        rank = lambda item: RECOMMEND_FORMAT_ORDER.index(item["format"])
        lp = select_spreadsheet_lp_candidate(values, rank_getter=rank)
        beta_zero = select_spreadsheet_cg_candidate(
            values, rank_getter=rank, confidence_gate_strength=0.0
        )
        fixed = select_spreadsheet_medoid(
            values, rank_getter=rank, logprob_field=None
        )
        alpha_zero = select_spreadsheet_cg_candidate(
            values,
            rank_getter=rank,
            strength=0.0,
            confidence_gate_strength=2.0,
        )
        self.assertEqual(beta_zero["selected_format"], lp["selected_format"])
        self.assertEqual(alpha_zero["selected_format"], fixed["selected_format"])

    def test_selector_does_not_mutate_input_cells(self):
        values = [
            spreadsheet_candidate("latex", 0.4, (0, 0)),
            spreadsheet_candidate("json_cells", 0.6, (1, 1)),
        ]
        before = copy.deepcopy(values)
        select_spreadsheet_cg_candidate(values)
        self.assertEqual(values, before)
        self.assertTrue(all("_cells" in item for item in values))

    def test_aggregate_ignores_gold_fields_and_never_looks_up_golden(self):
        records = self.records()
        records["latex"] = self.record("latex", 0.4, (0, 0))
        records["json_cells"] = self.record("json_cells", 0.6, (1, 1))
        changed = copy.deepcopy(records)
        for record in changed.values():
            record["test_case_results"] = [0]
            record["total_hard_restriction"] = 0
            record["gold_answer"] = "poison"
        with (
            patch(
                "core.sheetflex.spreadsheet._find_dataset_workbook",
                side_effect=AssertionError("golden lookup during selection"),
            ),
            patch("core.utils.model_resp") as model_request,
            patch.object(ClientJupyterKernel, "execute") as execute,
        ):
            first = aggregate_spreadsheet_cg_sample(
                self.item, records, self.run_dirs, self.input_path
            )
            second = aggregate_spreadsheet_cg_sample(
                self.item, changed, self.run_dirs, self.input_path
            )
        model_request.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(first["selected_class_id"], second["selected_class_id"])
        self.assertEqual(first["selected_format"], second["selected_format"])

    def test_copy_is_byte_identical_and_no_composition_occurs(self):
        records = self.records()
        records["latex"] = self.record("latex", 0.4, (0, 0))
        records["json_cells"] = self.record("json_cells", 0.6, (1, 1))
        aggregate = aggregate_spreadsheet_cg_sample(
            self.item, records, self.run_dirs, self.input_path
        )
        output_dir = self.root / "output"
        self.assertEqual(copy_selected_workbooks([aggregate], output_dir), 1)
        copied = output_dir / "spreadsheet" / "1_sample_output.xlsx"
        self.assertEqual(
            copied.read_bytes(), Path(aggregate["selected_source_file"]).read_bytes()
        )

    def test_missing_and_all_invalid(self):
        records = self.records()
        records["latex"] = self.record("latex", None, (1, 1))
        with self.assertRaisesRegex(SheetFlexError, "sample_id=sample.*latex"):
            aggregate_spreadsheet_cg_sample(
                self.item,
                records,
                self.run_dirs,
                self.input_path,
                missing_logprob_policy="error",
            )
        fallback = aggregate_spreadsheet_cg_sample(
            self.item, records, self.run_dirs, self.input_path
        )
        self.assertTrue(fallback["trace"]["fallback"])
        invalid = aggregate_spreadsheet_cg_sample(
            self.item,
            self.records(),
            self.run_dirs,
            self.input_path,
            missing_logprob_policy="error",
        )
        self.assertFalse(invalid["format_valid"])
        self.assertIsNone(invalid["selected_source_file"])
        diagnostics = spreadsheet_cg_diagnostics([invalid])
        self.assertEqual(diagnostics["final_class_tie_events"], 0)
        self.assertEqual(diagnostics["equal_weight_equivalent_tie_samples"], 0)


class CGVoteCLITest(unittest.TestCase):
    def test_defaults_and_invalid_strengths(self):
        args = cg_cli.parse_args(
            ["realhit", "--run_map", "map.json", "--output_dir", "out"]
        )
        self.assertEqual(args.lp_weight_strength, 1.0)
        self.assertEqual(args.confidence_gate_strength, 1.0)
        self.assertEqual(args.missing_logprob_policy, "vote")
        self.assertEqual(args.tie_break_order, "recommend")
        for flag, value in (
            ("--lp_weight_strength", "nan"),
            ("--confidence_gate_strength", "inf"),
            ("--confidence_gate_strength", "-1"),
        ):
            with self.subTest(flag=flag, value=value):
                with self.assertRaises(SystemExit):
                    cg_cli.parse_args(
                        [
                            "realhit",
                            "--run_map",
                            "map.json",
                            "--output_dir",
                            "out",
                            flag,
                            value,
                        ]
                    )

    def test_output_cannot_be_inside_candidate_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(SheetFlexError, "candidate directory"):
                cg_cli._validate_output_isolation(
                    root / "candidate" / "output",
                    {"latex": root / "candidate"},
                )

    def test_candidate_generation_metadata_is_honest_and_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_map = {}
            for format_name in RECOMMEND_FORMAT_ORDER:
                run_dir = root / format_name
                run_dir.mkdir()
                run_map[format_name] = run_dir
            missing = cg_cli._candidate_generation_config(run_map)
            self.assertEqual(
                missing["status"], "unverified_missing_run_metadata"
            )
            (run_map["latex"] / "run_metadata.json").write_text(
                '{"temperature": 0.1, "top_p": 1, "table_format": "latex"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SheetFlexError, "settings mismatch"):
                cg_cli._candidate_generation_config(run_map)

    def test_candidate_result_identity_rejects_mixed_models_and_formats(self):
        indexed = {
            "latex": {
                "1": {
                    "table_metadata": {
                        "table_format": "latex",
                        "token_model": "model-a",
                    }
                }
            },
            "markdown": {
                "1": {
                    "table_metadata": {
                        "table_format": "markdown",
                        "token_model": "model-b",
                    }
                }
            },
        }
        with self.assertRaisesRegex(SheetFlexError, "mix models"):
            cg_cli._validate_candidate_result_identity(indexed)
        indexed["markdown"]["1"]["table_metadata"]["token_model"] = "model-a"
        indexed["markdown"]["1"]["table_metadata"]["table_format"] = "latex"
        with self.assertRaisesRegex(SheetFlexError, "format mismatch"):
            cg_cli._validate_candidate_result_identity(indexed)

    def test_legacy_order_is_used_only_for_final_class_tie(self):
        candidates = [
            answer_candidate("latex", 0.5, "A"),
            answer_candidate("json_cells", 0.5, "B"),
        ]
        result = aggregate_answer_cg_vote(
            candidates,
            rank_getter=lambda item: LEGACY_FORMAT_ORDER.index(item["format"]),
            tie_break_order="legacy",
        )
        self.assertEqual(result["selected_format"], "latex")
        self.assertEqual(result["tie_break_source"], "format_order")


if __name__ == "__main__":
    unittest.main()
