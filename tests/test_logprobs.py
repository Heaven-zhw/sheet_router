import unittest
from unittest.mock import patch

import realhit_cot as realhit_cli
import spreadsheet_pot as spreadsheet_cli

from core.solver.realhit_cot import RealHiTCoTSolver
from core.solver.spreadsheet_pot import SpreadSheetPoTSolver
from core.utils import model_resp, summarize_choice_logprobs


def choice(content, logprob_values=None):
    result = {"message": {"role": "assistant", "content": content}}
    if logprob_values is not None:
        result["logprobs"] = {
            "content": [
                {"token": f"t{index}", "logprob": value, "top_logprobs": []}
                for index, value in enumerate(logprob_values)
            ]
        }
    return result


class LogprobSummaryTest(unittest.TestCase):
    def test_normal_content_sum_mean_and_count(self):
        summary = summarize_choice_logprobs(choice("answer", [-1.0, -2.5, -0.5]))
        self.assertTrue(summary["logprob_available"])
        self.assertEqual(summary["sequence_logprob_sum"], -4.0)
        self.assertEqual(summary["sequence_logprob_mean"], -4.0 / 3)
        self.assertEqual(summary["sequence_token_count"], 3)
        self.assertIsNone(summary["logprob_unavailable_reason"])

    def test_empty_content_is_unavailable(self):
        summary = summarize_choice_logprobs(
            {"message": {"content": ""}, "logprobs": {"content": []}}
        )
        self.assertFalse(summary["logprob_available"])
        self.assertIsNone(summary["sequence_logprob_sum"])
        self.assertIsNone(summary["sequence_logprob_mean"])
        self.assertEqual(summary["sequence_token_count"], 0)
        self.assertIn("empty", summary["logprob_unavailable_reason"])

    def test_missing_logprobs_is_unavailable(self):
        summary = summarize_choice_logprobs(choice("answer"))
        self.assertFalse(summary["logprob_available"])
        self.assertIsNone(summary["sequence_logprob_sum"])
        self.assertIn("omitted", summary["logprob_unavailable_reason"])

    def test_partial_missing_token_logprob_does_not_create_partial_score(self):
        fixture = choice("answer", [-1.0, None, -2.0])
        summary = summarize_choice_logprobs(fixture)
        self.assertFalse(summary["logprob_available"])
        self.assertIsNone(summary["sequence_logprob_sum"])
        self.assertIsNone(summary["sequence_logprob_mean"])
        self.assertEqual(summary["sequence_token_count"], 3)
        self.assertIn("index(es): 1", summary["logprob_unavailable_reason"])


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class ModelResponseFallbackTest(unittest.TestCase):
    @patch("core.utils.requests.post")
    def test_logprob_rejection_retries_generation_without_logprobs(self, post):
        post.side_effect = [
            FakeHTTPResponse({"error": {"message": "logprobs unsupported"}}),
            FakeHTTPResponse({"choices": [choice("same answer")]}),
        ]
        result = model_resp(
            "localhost:8000",
            [{"role": "user", "content": "question"}],
            model_params={"temperature": 0, "top_p": 1, "logprobs": True},
            model_name="unsupported-model",
        )
        self.assertEqual(result["message"]["content"], "same answer")
        self.assertTrue(post.call_args_list[0].kwargs["json"]["logprobs"])
        self.assertNotIn("logprobs", post.call_args_list[1].kwargs["json"])
        summary = summarize_choice_logprobs(result)
        self.assertFalse(summary["logprob_available"])
        self.assertIn("logprobs unsupported", summary["logprob_unavailable_reason"])


class LogprobCLITest(unittest.TestCase):
    def test_switch_defaults_off_and_can_be_enabled(self):
        for cli, program in (
            (realhit_cli, "realhit_cot.py"),
            (spreadsheet_cli, "spreadsheet_pot.py"),
        ):
            with self.subTest(cli=program):
                with patch("sys.argv", [program]):
                    disabled = cli.parse_args()
                with patch("sys.argv", [program, "--save_logprobs"]):
                    enabled = cli.parse_args()
                with patch("sys.argv", [program, "--no-save_logprobs"]):
                    explicitly_disabled = cli.parse_args()
                self.assertFalse(disabled.save_logprobs)
                self.assertTrue(enabled.save_logprobs)
                self.assertFalse(explicitly_disabled.save_logprobs)
                self.assertNotIn("logprobs", cli.build_suffix(disabled))
                self.assertIn("logprobs", cli.build_suffix(enabled))


class RealHiTLogprobTest(unittest.TestCase):
    valid_response = (
        '{"reasoning":["checked"],'
        '"final_answer":"[Final Answer]: final value"}'
    )

    def solver(self, save_logprobs):
        solver = RealHiTCoTSolver(
            save_logprobs=save_logprobs,
            max_retries=2,
            temperature=0,
            top_p=1,
        )
        solver.build_prompt = lambda data: ("prompt", {"table_format": "markdown"})
        return solver

    @patch("core.solver.realhit_cot.model_resp")
    def test_top_level_uses_only_final_format_valid_attempt(self, mocked_model):
        mocked_model.side_effect = [
            choice("not json", [-0.5]),
            choice(self.valid_response, [-2.0, -3.0]),
        ]
        result = self.solver(True).get_solution({"id": 1})
        self.assertFalse(result["attempts"][0]["valid"])
        self.assertEqual(result["attempts"][0]["sequence_logprob_sum"], -0.5)
        self.assertTrue(result["attempts"][1]["valid"])
        self.assertEqual(result["attempts"][1]["sequence_logprob_sum"], -5.0)
        self.assertEqual(result["sequence_logprob_sum"], -5.0)
        self.assertEqual(result["model_answer"], "final value")

    @patch("core.solver.realhit_cot.model_resp")
    def test_switch_does_not_change_answer_or_retry_logic(self, mocked_model):
        mocked_model.return_value = choice(self.valid_response, [-1.25])
        disabled = self.solver(False).get_solution({"id": 1})
        mocked_model.return_value = choice(self.valid_response, [-1.25])
        enabled = self.solver(True).get_solution({"id": 1})
        self.assertEqual(disabled["model_answer"], enabled["model_answer"])
        self.assertEqual(disabled["solution"], enabled["solution"])
        self.assertNotIn("logprob_available", disabled)
        self.assertNotIn("logprob_available", disabled["attempts"][0])
        self.assertTrue(enabled["logprob_available"])
        self.assertNotIn("logprobs", self.solver(False).model_params)
        self.assertTrue(self.solver(True).model_params["logprobs"])

    def test_structure_runs_keep_independent_final_summaries(self):
        solver = self.solver(True)
        reference = {
            "model_answer": "reference",
            "format_valid": True,
            "error": None,
            "attempts": [],
            "parsed_response": {},
            "table_metadata": {},
            "solution": "reference response",
            "logprob_available": True,
            "sequence_logprob_sum": -10.0,
            "sequence_logprob_mean": -1.0,
            "sequence_token_count": 10,
            "logprob_unavailable_reason": None,
        }
        swap = dict(reference)
        swap.update(
            {
                "model_answer": "swap",
                "solution": "swap response",
                "sequence_logprob_sum": -20.0,
                "sequence_logprob_mean": -2.0,
            }
        )
        with patch.object(solver, "get_solution", side_effect=[reference, swap]):
            with patch.object(solver, "_score_qa", return_value={}):
                result = solver(
                    {
                        "FileName": "table",
                        "QuestionType": "Structure Comprehending",
                        "ProcessedAnswer": "",
                    }
                )
        self.assertEqual(
            result["structure_reference_run"]["sequence_logprob_sum"], -10.0
        )
        self.assertEqual(result["structure_swap_run"]["sequence_logprob_sum"], -20.0)
        self.assertEqual(result["sequence_logprob_sum"], -20.0)

    def test_structure_final_summary_survives_metric_error(self):
        solver = self.solver(True)
        base = {
            "model_answer": "reference",
            "format_valid": True,
            "error": None,
            "attempts": [],
            "parsed_response": {},
            "table_metadata": {},
            "solution": "response",
            "logprob_available": True,
            "sequence_logprob_sum": -1.0,
            "sequence_logprob_mean": -1.0,
            "sequence_token_count": 1,
            "logprob_unavailable_reason": None,
        }
        swap = dict(base)
        swap.update({"model_answer": "swap", "sequence_logprob_sum": -2.0})
        with patch.object(solver, "get_solution", side_effect=[base, swap]):
            with patch.object(solver, "_score_qa", side_effect=RuntimeError("metric")):
                result = solver(
                    {
                        "FileName": "table",
                        "QuestionType": "Structure Comprehending",
                        "ProcessedAnswer": "",
                    }
                )
        self.assertEqual(result["model_answer"], "swap")
        self.assertEqual(result["sequence_logprob_sum"], -2.0)
        self.assertIn("RuntimeError: metric", result["eval_error"])


class FakeCodeClient:
    def __init__(self):
        self.last_code = None

    def execute(self, code):
        if "assert os.path.exists" in code:
            return "OUTPUT_FILE_CREATED" if self.last_code == "good" else ""
        if "if os.path.exists" in code:
            return ""
        self.last_code = code.strip()
        if self.last_code == "bad":
            return "Traceback (most recent call last): failed"
        return ""


class SpreadsheetLogprobTest(unittest.TestCase):
    def solver(self, save_logprobs, retries=2):
        solver = SpreadSheetPoTSolver(
            save_logprobs=save_logprobs,
            max_retries=retries,
            temperature=0,
            top_p=1,
        )
        solver.build_prompt = lambda data: ("prompt", {"table_format": "markdown"})
        return solver

    @patch("core.solver.spreadsheet_pot.model_resp")
    def test_top_level_uses_only_successful_output_attempt(self, mocked_model):
        mocked_model.side_effect = [
            choice("bad", [-0.5]),
            choice("```python\ngood\n```", [-2.0, -3.0]),
        ]
        result = self.solver(True).get_solution(
            {
                "id": "sample",
                "input_path": "/mnt/data/input/source.xlsx",
                "output_path": "/mnt/data/output/result.xlsx",
            },
            FakeCodeClient(),
        )
        self.assertFalse(result["attempts"][0]["success"])
        self.assertEqual(result["attempts"][0]["sequence_logprob_sum"], -0.5)
        self.assertTrue(result["attempts"][1]["success"])
        self.assertTrue(result["attempts"][1]["output_created"])
        self.assertEqual(result["sequence_logprob_sum"], -5.0)
        self.assertEqual(result["solution"], "good")

    @patch("core.solver.spreadsheet_pot.model_resp")
    def test_switch_does_not_change_final_code_or_execution(self, mocked_model):
        mocked_model.return_value = choice("```python\ngood\n```", [-1.25])
        disabled = self.solver(False, retries=1).get_solution(
            {"id": "sample", "output_path": "/mnt/data/output/result.xlsx"},
            FakeCodeClient(),
        )
        mocked_model.return_value = choice("```python\ngood\n```", [-1.25])
        enabled = self.solver(True, retries=1).get_solution(
            {"id": "sample", "output_path": "/mnt/data/output/result.xlsx"},
            FakeCodeClient(),
        )
        self.assertEqual(disabled["solution"], enabled["solution"])
        self.assertEqual(disabled["execution_success"], enabled["execution_success"])
        self.assertNotIn("logprob_available", disabled)
        self.assertNotIn("logprob_available", disabled["attempts"][0])
        self.assertTrue(enabled["logprob_available"])


if __name__ == "__main__":
    unittest.main()
