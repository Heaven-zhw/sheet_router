import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import openpyxl

from core.sheetflex.common import (
    FORMAT_ORDER,
    index_rows_by_id,
    load_indexed_runs,
    max_score_items,
)
from core.sheetflex.realhit import aggregate_answer_vote, aggregate_realhit_sample
from core.sheetflex.spreadsheet import (
    aggregate_spreadsheet_sample,
    copy_selected_workbooks,
    select_spreadsheet_medoid,
)


def realhit_candidate(format_name, answer=None, *, valid=True, logprob=None):
    return {
        "format": format_name,
        "source_run_dir": None,
        "valid": valid,
        "invalid_reason": None if valid else "invalid",
        "model_answer": answer,
        "normalized_answer": answer.lower() if valid else None,
        "aggregation_score": None,
        "selected": False,
        "logprob_available": logprob is not None,
        "sequence_logprob_sum": logprob,
        "sequence_logprob_mean": logprob,
        "sequence_token_count": 1 if logprob is not None else 0,
        "logprob_unavailable_reason": None if logprob is not None else "missing",
    }


def structure_record(reference, swap, *, valid=True, logprob=None):
    def run(answer):
        return {
            "format_valid": valid,
            "model_answer": answer,
            "logprob_available": logprob is not None,
            "sequence_logprob_sum": logprob,
            "sequence_logprob_mean": logprob,
            "sequence_token_count": 1 if logprob is not None else 0,
        }

    return {
        "structure_reference_run": run(reference),
        "structure_swap_run": run(swap),
    }


class RealHiTVoteTest(unittest.TestCase):
    def test_normal_majority_vote(self):
        result = aggregate_answer_vote(
            [
                realhit_candidate("latex", "A"),
                realhit_candidate("markdown", "A"),
                realhit_candidate("json_cells", "B"),
            ]
        )
        self.assertEqual(result["selected_answer"], "A")
        self.assertEqual(result["winning_group_size"], 2)
        self.assertFalse(result["tie"])

    def test_existing_decimal_normalization_defines_vote_key(self):
        records = {
            "latex": {
                "format_valid": True,
                "model_answer": "The value is 1.24",
            },
            "markdown": {
                "format_valid": True,
                "model_answer": "value is 1.2",
            },
            **{name: None for name in FORMAT_ORDER[2:]},
        }
        result = aggregate_realhit_sample("sample", "Fact Checking", records)
        groups = result["trace"]["answer_vote"]["answer_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["size"], 2)

    def test_three_way_tie_uses_logprob(self):
        result = aggregate_answer_vote(
            [
                realhit_candidate("latex", "A", logprob=-3.0),
                realhit_candidate("markdown", "B", logprob=-1.0),
                realhit_candidate("json_cells", "C", logprob=-2.0),
            ]
        )
        self.assertTrue(result["tie"])
        self.assertEqual(result["tie_break_source"], "logprob")
        self.assertEqual(result["selected_format"], "markdown")

    def test_same_answer_chooses_best_original_candidate(self):
        result = aggregate_answer_vote(
            [
                realhit_candidate("latex", "Same", logprob=-3.0),
                realhit_candidate("markdown", "Same", logprob=-1.0),
                realhit_candidate("json_cells", "Same", logprob=-2.0),
            ]
        )
        self.assertFalse(result["tie"])
        self.assertEqual(result["selected_format"], "markdown")
        self.assertEqual(result["representative_selection_source"], "logprob")

    def test_missing_logprob_and_input_order_use_fixed_format_order(self):
        candidates = [
            realhit_candidate("json_cells", "C"),
            realhit_candidate("markdown", "B", logprob=-1.0),
            realhit_candidate("latex", "A", logprob=-2.0),
        ]
        result = aggregate_answer_vote(candidates)
        self.assertEqual(result["tie_break_source"], "format_order")
        self.assertEqual(result["selected_format"], "latex")

    def test_all_invalid(self):
        result = aggregate_answer_vote(
            [realhit_candidate(name, valid=False) for name in FORMAT_ORDER]
        )
        self.assertFalse(result["format_valid"])
        self.assertIsNone(result["selected_format"])
        self.assertEqual(result["valid_candidate_count"], 0)

    def test_strict_isclose_score_tolerance(self):
        _, tied = max_score_items(
            [
                {"aggregation_score": 1.0, "format": "latex"},
                {"aggregation_score": 1.0 + 5e-13, "format": "markdown"},
            ]
        )
        self.assertEqual(len(tied), 2)
        _, not_tied = max_score_items(
            [
                {"aggregation_score": 1.0, "format": "latex"},
                {"aggregation_score": 1.0 + 1e-8, "format": "markdown"},
            ]
        )
        self.assertEqual(len(not_tied), 1)

    def test_structure_sides_are_voted_independently(self):
        records = {
            "latex": structure_record("R", "X"),
            "markdown": structure_record("R", "Y"),
            "json_cells": structure_record("S", "Y"),
            "json_rows": structure_record("", "", valid=False),
            "image": None,
            "excel_1_image": None,
        }
        result = aggregate_realhit_sample(
            "sample", "Structure Comprehending", records
        )
        self.assertEqual(result["structure_reference_answer"], "R")
        self.assertEqual(result["structure_swap_answer"], "Y")
        self.assertEqual(result["selected_format"]["reference"], "latex")
        self.assertEqual(result["selected_format"]["swap"], "markdown")

    def test_result_index_is_id_based_and_rejects_duplicates(self):
        indexed = index_rows_by_id([{"id": 2}, {"id": 1}])
        self.assertEqual(indexed["1"]["id"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            index_rows_by_id([{"id": 1}, {"id": "1"}])

    def test_synthetic_jsonl_runs_align_by_id_not_array_position(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_map = {}
            for index, format_name in enumerate(FORMAT_ORDER):
                run_dir = Path(temp_dir) / format_name
                run_dir.mkdir()
                rows = [{"id": "a", "value": format_name}, {"id": "b"}]
                if index % 2:
                    rows.reverse()
                (run_dir / "results.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                run_map[format_name] = run_dir
            indexed = load_indexed_runs(run_map, "results.jsonl")
            self.assertEqual(indexed["markdown"]["a"]["value"], "markdown")
            self.assertEqual(indexed["excel_1_image"]["b"]["id"], "b")


def save_book(path, values, sheet_name="Target"):
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    for coordinate, value in values.items():
        worksheet[coordinate] = value
    workbook.save(path)
    workbook.close()


class SpreadsheetVoteTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.xlsx"
        save_book(self.input_path, {"A1": 0, "B1": 0})
        self.run_dirs = {}
        for format_name in FORMAT_ORDER:
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

    def record(self, format_name, values=None, *, success=True, logprob=None):
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
            "logprob_available": logprob is not None,
            "sequence_logprob_sum": logprob,
        }

    def empty_records(self):
        return {
            format_name: self.record(format_name, success=False)
            for format_name in FORMAT_ORDER
        }

    def test_xlsx_similarity_matrix_and_unique_medoid(self):
        records = self.empty_records()
        records["latex"] = self.record("latex", {"A1": 0, "B1": 0})
        records["markdown"] = self.record("markdown", {"A1": 0, "B1": 1})
        records["json_cells"] = self.record(
            "json_cells", {"A1": 1, "B1": 1}
        )
        result = aggregate_spreadsheet_sample(
            self.item, records, self.run_dirs, self.input_path
        )
        self.assertEqual(result["selected_format"], "markdown")
        matrix = result["trace"]["similarity_matrix"]
        self.assertEqual(matrix["latex"]["markdown"], 0.5)
        self.assertEqual(matrix["latex"]["json_cells"], 0.0)
        self.assertEqual(matrix["markdown"]["json_cells"], 0.5)
        self.assertFalse(result["trace"]["tie"])

    def test_medoid_tie_uses_logprob(self):
        records = self.empty_records()
        records["latex"] = self.record("latex", {"A1": 0, "B1": 0}, logprob=-2)
        records["markdown"] = self.record(
            "markdown", {"A1": 1, "B1": 1}, logprob=-1
        )
        result = aggregate_spreadsheet_sample(
            self.item, records, self.run_dirs, self.input_path
        )
        self.assertTrue(result["trace"]["tie"])
        self.assertEqual(result["trace"]["tie_break_source"], "logprob")
        self.assertEqual(result["selected_format"], "markdown")

    def test_medoid_tie_missing_logprob_uses_fixed_order(self):
        records = self.empty_records()
        records["latex"] = self.record("latex", {"A1": 0, "B1": 0})
        records["markdown"] = self.record(
            "markdown", {"A1": 1, "B1": 1}, logprob=-1
        )
        result = aggregate_spreadsheet_sample(
            self.item, records, self.run_dirs, self.input_path
        )
        self.assertEqual(result["trace"]["tie_break_source"], "format_order")
        self.assertEqual(result["selected_format"], "latex")

    def test_missing_and_damaged_files_are_invalid(self):
        records = self.empty_records()
        records["latex"] = self.record("latex", {"A1": 0, "B1": 0})
        records["markdown"] = self.record("markdown", values=None, success=True)
        damaged = (
            self.run_dirs["json_cells"]
            / "spreadsheet"
            / "1_sample_output.xlsx"
        )
        damaged.write_text("not an xlsx", encoding="utf-8")
        records["json_cells"] = self.record(
            "json_cells", values=None, success=True
        )
        result = aggregate_spreadsheet_sample(
            self.item, records, self.run_dirs, self.input_path
        )
        traces = {c["format"]: c for c in result["trace"]["candidates"]}
        self.assertTrue(traces["latex"]["valid"])
        self.assertIn("not_found", traces["markdown"]["invalid_reason"])
        self.assertIn("failed_to_open", traces["json_cells"]["invalid_reason"])
        self.assertEqual(result["selected_format"], "latex")

    def test_all_invalid_does_not_copy_output(self):
        result = aggregate_spreadsheet_sample(
            self.item, self.empty_records(), self.run_dirs, self.input_path
        )
        output_dir = self.root / "vote"
        copied = copy_selected_workbooks([result], output_dir)
        self.assertEqual(copied, 0)
        self.assertFalse((output_dir / "spreadsheet" / "1_sample_output.xlsx").exists())

    def test_low_level_medoid_strips_internal_cell_maps(self):
        candidates = []
        for format_name, values in (
            ("latex", [0, 0]),
            ("markdown", [0, 1]),
            ("json_cells", [1, 1]),
        ):
            cells = OrderedDict(
                [(('Target', 'A1'), values[0]), (('Target', 'B1'), values[1])]
            )
            candidates.append(
                {
                    "format": format_name,
                    "valid": True,
                    "selected": False,
                    "output_file": f"{format_name}.xlsx",
                    "aggregation_score": None,
                    "logprob_available": False,
                    "sequence_logprob_sum": None,
                    "_cells": cells,
                }
            )
        result = select_spreadsheet_medoid(candidates)
        self.assertEqual(result["selected_format"], "markdown")
        self.assertTrue(all("_cells" not in c for c in result["candidates"]))


if __name__ == "__main__":
    unittest.main()
