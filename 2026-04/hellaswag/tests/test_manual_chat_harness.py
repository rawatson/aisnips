import json
import unittest

from manual_chat_harness import (
    coerce_example,
    parse_answer_text,
    render_prompt,
    score_answers,
)


class ManualChatHarnessTests(unittest.TestCase):
    def test_render_prompt_includes_strict_json_contract(self):
        example = coerce_example(
            {
                "ind": 7,
                "ctx": "Someone starts a sentence and",
                "endings": ["finishes it.", "eats a helmet.", "opens lava.", "becomes a note."],
                "label": 0,
            },
            1,
        )

        prompt = render_prompt([example])

        self.assertIn("Return only JSON", prompt)
        self.assertIn('"answers":[{"id":"EXAMPLE_ID","choice":0}]', prompt)
        self.assertIn("id: 7", prompt)
        self.assertIn("0: finishes it.", prompt)

    def test_parse_answer_text_accepts_markdown_json_block(self):
        answers = parse_answer_text(
            """```json
            {"answers":[{"id":"7","choice":3}]}
            ```"""
        )

        self.assertEqual(answers, {"7": 3})

    def test_score_answers_reports_mistakes_and_accuracy(self):
        examples = [
            coerce_example(
                {"ind": "a", "ctx": "x", "endings": ["a", "b", "c", "d"], "label": 1},
                1,
            ),
            coerce_example(
                {"ind": "b", "ctx": "y", "endings": ["a", "b", "c", "d"], "label": 2},
                2,
            ),
        ]

        result = score_answers(examples, {"a": 1, "b": 0})

        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["incorrect"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["mistakes"], [{"id": "b", "expected": 2, "actual": 0}])

    def test_score_result_is_json_serializable(self):
        example = coerce_example(
            {"ind": 1, "ctx": "x", "endings": ["a", "b", "c", "d"], "label": 3},
            1,
        )

        json.dumps(score_answers([example], {"1": 3}))


if __name__ == "__main__":
    unittest.main()
