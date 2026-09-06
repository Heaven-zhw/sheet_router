import math
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import openpyxl

import sheetflex_lp_vote as lp_cli
from core.sheetflex.common import RECOMMEND_FORMAT_ORDER, SheetFlexError
from core.sheetflex.lp_vote import (
    aggregate_answer_lp_vote,
    aggregate_realhit_lp_sample,
    aggregate_spreadsheet_lp_sample,
    compute_lp_weights,
    select_spreadsheet_lp_candidate,
)
from core.sheetflex.realhit import aggregate_answer_vote
from core.sheetflex.spreadsheet import (
    copy_selected_workbooks,
    select_spreadsheet_medoid,
)


def candidate(format_name, mean=None, *, valid=True, answer=None):
    return {
        "format": format_name,
        "valid": valid,
        "invalid_reason": None if valid else "invalid",
        "model_answer": answer,
        "normalized_answer": answer.lower() if valid and answer else None,
        "aggregation_score": None,
        "selected": False,
        "logprob_available": mean is not None,
        "sequence_logprob_mean": mean,
        "sequence_logprob_sum": -999.0 if mean is not None else None,
    }


class LPWeightTest(unittest.TestCase):
    def test_higher_mean_has_higher_weight_and_sum_is_one(self):
        values = [candidate("latex", -2.0), candidate("markdown", -1.0)]
        result = compute_lp_weights(values, 1.0)
        self.assertGreater(result.weights[1], result.weights[0])
        self.assertAlmostEqual(sum(result.weights), 1.0)

    def test_invalid_is_zero_and_single_valid_is_one_without_mean(self):
        invalid = candidate("latex", 100.0, valid=False)
        lone = candidate("markdown", None)
        result = compute_lp_weights([invalid, lone])
        self.assertEqual(result.weights, (0.0, 1.0))
        self.assertFalse(result.fallback)

    def test_equal_means_and_alpha_zero_are_equal_weight(self):
        values = [candidate("latex", -2.0), candidate("markdown", -2.0)]
        self.assertEqual(compute_lp_weights(values).weights, (0.5, 0.5))
        values[1]["sequence_logprob_mean"] = 100.0
        self.assertEqual(compute_lp_weights(values, 0.0).weights, (0.5, 0.5))

    def test_extreme_values_are_stable(self):
        values = [candidate("latex", -1e300), candidate("markdown", 1e300)]
        result = compute_lp_weights(values, 10.0)
        self.assertTrue(all(math.isfinite(value) for value in result.weights))
        self.assertEqual(result.weights, (0.0, 1.0))
        self.assertEqual(compute_lp_weights(values, 0.0).weights, (0.5, 0.5))

    def test_missing_vote_falls_back_without_deleting_candidate(self):
        values = [candidate("latex", None), candidate("markdown", -1.0)]
        result = compute_lp_weights(values, missing_logprob_policy="vote")
        self.assertTrue(result.fallback)
        self.assertEqual(result.weights, (0.5, 0.5))
        self.assertIn("latex", result.fallback_reason)

    def test_missing_error_and_negative_strength_are_rejected(self):
        values = [candidate("latex", None), candidate("markdown", -1.0)]
        with self.assertRaisesRegex(SheetFlexError, "latex"):
            compute_lp_weights(values, missing_logprob_policy="error")
        with self.assertRaisesRegex(SheetFlexError, ">= 0"):
            compute_lp_weights(values, strength=-0.1)

        illegal = [candidate("latex", float("nan")), candidate("markdown", -1.0)]
        with self.assertRaisesRegex(SheetFlexError, "latex"):
            compute_lp_weights(illegal, missing_logprob_policy="error")

    def test_cli_rejects_negative_strength(self):
        with self.assertRaises(SystemExit):
            lp_cli.parse_args(
                [
                    "realhit",
                    "--run_map",
                    "map.json",
                    "--output_dir",
                    "out",
                    "--lp_weight_strength",
                    "-1",
                ]
            )


class RealHiTLPVoteTest(unittest.TestCase):
    def test_minority_high_weight_answer_beats_low_weight_majority(self):
        result = aggregate_answer_lp_vote(
            [
                candidate("latex", -10.0, answer="A"),
                candidate("markdown", -10.0, answer="A"),
                candidate("json_cells", 0.0, answer="B"),
            ]
        )
        self.assertEqual(result["selected_answer"], "B")
        self.assertEqual(result["selected_format"], "json_cells")

    def test_alpha_zero_matches_fixed_recommend_vote(self):
        values = [
            candidate("latex", -100.0, answer="A"),
            candidate("json_cells", 0.0, answer="B"),
        ]
        expected = aggregate_answer_vote(
            values,
            rank_getter=lambda item: RECOMMEND_FORMAT_ORDER.index(item["format"]),
            logprob_field=None,
        )
        result = aggregate_answer_lp_vote(values, strength=0.0)
        self.assertEqual(result["selected_format"], expected["selected_format"])
        self.assertEqual(result["selected_answer"], expected["selected_answer"])

    def test_winning_group_representative_uses_weight_then_order(self):
        result = aggregate_answer_lp_vote(
            [
                candidate("latex", -3.0, answer="Same"),
                candidate("markdown", -1.0, answer="Same"),
            ]
        )
        self.assertEqual(result["selected_format"], "markdown")
        self.assertEqual(result["representative_selection_source"], "lp_weight")

    def test_missing_fallback_is_fixed_recommend_even_if_legacy_requested(self):
        values = [
            candidate("latex", None, answer="A"),
            candidate("json_cells", -1.0, answer="B"),
        ]
        result = aggregate_answer_lp_vote(
            values,
            missing_logprob_policy="vote",
            rank_getter=lambda item: ("latex", "json_cells").index(item["format"]),
            tie_break_order="legacy",
        )
        self.assertTrue(result["lpvote_fallback"])
        self.assertEqual(result["selected_format"], "json_cells")
        self.assertEqual(result["tie_break_order"], "recommend")

    def test_structure_branches_read_independent_nested_means(self):
        def record(ref_answer, ref_mean, swap_answer, swap_mean):
            def branch(answer, mean):
                return {
                    "format_valid": True,
                    "model_answer": answer,
                    "logprob_available": True,
                    "sequence_logprob_mean": mean,
                    "sequence_logprob_sum": -999.0,
                }

            return {
                "model_answer": "wrong-top-level",
                "sequence_logprob_mean": 999.0,
                "structure_reference_run": branch(ref_answer, ref_mean),
                "structure_swap_run": branch(swap_answer, swap_mean),
            }

        records = {
            "latex": record("R1", 0.0, "S1", -10.0),
            "markdown": record("R2", -10.0, "S2", 0.0),
            "json_cells": None,
            "json_rows": None,
            "image": None,
            "excel_1_image": None,
        }
        result = aggregate_realhit_lp_sample(
            "sample", "Structure Comprehending", records
        )
        self.assertEqual(result["structure_reference_answer"], "R1")
        self.assertEqual(result["structure_swap_answer"], "S2")
        self.assertNotEqual(result["model_answer"], "wrong-top-level")

    def test_structure_missing_mean_falls_back_both_branches(self):
        def branch(answer, mean):
            return {
                "format_valid": True,
                "model_answer": answer,
                "logprob_available": mean is not None,
                "sequence_logprob_mean": mean,
            }

        records = {
            "latex": {
                "structure_reference_run": branch("R1", None),
                "structure_swap_run": branch("S1", 0.0),
            },
            "markdown": {
                "structure_reference_run": branch("R2", -1.0),
                "structure_swap_run": branch("S2", -1.0),
            },
            "json_cells": None,
            "json_rows": None,
            "image": None,
            "excel_1_image": None,
        }
        result = aggregate_realhit_lp_sample(
            "sample", "Structure Comprehending", records
        )
        self.assertTrue(result["trace"]["lpvote_fallback"])
        for key in ("structure_reference_vote", "structure_swap_vote"):
            self.assertTrue(result["trace"][key]["lpvote_fallback"])
            weights = [
                item["lp_weight"]
                for item in result["trace"][key]["candidates"]
                if item["valid"]
            ]
            self.assertEqual(weights, [0.5, 0.5])

    def test_all_invalid(self):
        result = aggregate_answer_lp_vote(
            [candidate(name, None, valid=False) for name in RECOMMEND_FORMAT_ORDER]
        )
        self.assertFalse(result["format_valid"])
        self.assertIsNone(result["selected_format"])


def cells(left, right):
    return OrderedDict([(("Target", "A1"), left), (("Target", "B1"), right)])


def spreadsheet_candidate(format_name, values, mean):
    item = candidate(format_name, mean)
    item.update({"output_file": f"{format_name}.xlsx", "_cells": cells(*values)})
    return item


def save_book(path, values):
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Target"
    worksheet["A1"] = values[0]
    worksheet["B1"] = values[1]
    workbook.save(path)
    workbook.close()


class SpreadsheetLPVoteTest(unittest.TestCase):
    def test_weighted_similarity_scores_supporter_dimension(self):
        values = [
            spreadsheet_candidate("latex", (0, 0), -2.0),
            spreadsheet_candidate("markdown", (0, 1), -1.0),
            spreadsheet_candidate("json_cells", (1, 1), 0.0),
        ]
        weights = compute_lp_weights(values).weights
        result = select_spreadsheet_lp_candidate(values)
        by_format = {item["format"]: item for item in result["candidates"]}
        self.assertAlmostEqual(
            by_format["latex"]["aggregation_score"], weights[0] + 0.5 * weights[1]
        )
        self.assertAlmostEqual(
            by_format["markdown"]["aggregation_score"],
            0.5 * weights[0] + weights[1] + 0.5 * weights[2],
        )
        self.assertAlmostEqual(
            by_format["json_cells"]["aggregation_score"],
            0.5 * weights[1] + weights[2],
        )

    def test_different_means_break_two_candidate_equal_weight_tie(self):
        result = select_spreadsheet_lp_candidate(
            [
                spreadsheet_candidate("latex", (0, 0), -10.0),
                spreadsheet_candidate("markdown", (1, 1), 0.0),
            ]
        )
        self.assertTrue(result["equal_weight_tie"])
        self.assertFalse(result["tie"])
        self.assertEqual(result["selected_format"], "markdown")

    def test_equal_weights_match_existing_equal_selection(self):
        values = [
            spreadsheet_candidate("latex", (0, 0), -1.0),
            spreadsheet_candidate("markdown", (1, 1), -1.0),
            spreadsheet_candidate("json_cells", (0, 1), -1.0),
        ]
        rank = lambda item: RECOMMEND_FORMAT_ORDER.index(item["format"])
        expected = select_spreadsheet_medoid(
            values, rank_getter=rank, logprob_field=None
        )
        result = select_spreadsheet_lp_candidate(values, rank_getter=rank)
        self.assertEqual(result["selected_format"], expected["selected_format"])

    def test_final_tie_uses_order_not_mean_again(self):
        result = select_spreadsheet_lp_candidate(
            [
                spreadsheet_candidate("latex", (1, 1), 100.0),
                spreadsheet_candidate("json_cells", (1, 1), -100.0),
            ]
        )
        self.assertTrue(result["tie"])
        self.assertEqual(result["selected_format"], "json_cells")
        self.assertEqual(result["tie_break_source"], "format_order")

    def test_only_existing_candidate_is_copied_and_selection_never_reads_golden(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.xlsx"
            save_book(input_path, (0, 0))
            run_dirs = {}
            records = {}
            for index, format_name in enumerate(RECOMMEND_FORMAT_ORDER):
                run_dir = root / format_name
                (run_dir / "spreadsheet").mkdir(parents=True)
                run_dirs[format_name] = run_dir
                output_name = "1_sample_output.xlsx"
                save_book(run_dir / "spreadsheet" / output_name, (index, index))
                records[format_name] = {
                    "execution_success": True,
                    "output_file": output_name,
                    "logprob_available": True,
                    "sequence_logprob_mean": float(index),
                }
            item = {
                "id": "sample",
                "instruction_type": "Cell-Level Manipulation",
                "answer_position": "A1:B1",
                "answer_sheet": "Target",
            }
            with patch(
                "core.sheetflex.spreadsheet._find_dataset_workbook",
                side_effect=AssertionError("golden lookup during selection"),
            ):
                aggregate = aggregate_spreadsheet_lp_sample(
                    item, records, run_dirs, input_path
                )
            output_dir = root / "lpvote"
            self.assertEqual(copy_selected_workbooks([aggregate], output_dir), 1)
            copied = output_dir / "spreadsheet" / "1_sample_output.xlsx"
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), Path(aggregate["selected_source_file"]).read_bytes())

    def test_alpha_zero_matches_fixed_recommend_medoid(self):
        values = [
            spreadsheet_candidate("latex", (0, 0), -100.0),
            spreadsheet_candidate("json_cells", (1, 1), 0.0),
        ]
        rank = lambda item: RECOMMEND_FORMAT_ORDER.index(item["format"])
        expected = select_spreadsheet_medoid(
            values, rank_getter=rank, logprob_field=None
        )
        result = select_spreadsheet_lp_candidate(
            values, rank_getter=rank, strength=0.0
        )
        self.assertEqual(result["selected_format"], expected["selected_format"])


if __name__ == "__main__":
    unittest.main()
