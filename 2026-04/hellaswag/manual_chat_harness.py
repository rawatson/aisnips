#!/usr/bin/env python3
"""Manual, chat-only HellaSwag harness.

This script is intentionally API-free. It prepares benchmark batches as plain
text prompts that can be pasted into a chat UI, then scores the JSON answers
you paste back into local files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import urlopen


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
HELLASWAG_VAL_URL = "https://github.com/rowanz/hellaswag/raw/master/data/hellaswag_val.jsonl"
HELLASWAG_VAL_SHA256 = "0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588"


@dataclass(frozen=True)
class Example:
    example_id: str
    context: str
    endings: list[str]
    label: int | None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return records


def coerce_example(record: dict[str, Any], fallback_id: int) -> Example:
    endings = record.get("endings")
    if not isinstance(endings, list) or len(endings) != 4:
        raise ValueError(f"example {fallback_id} must contain exactly four endings")

    context = record.get("ctx") or " ".join(
        part.strip()
        for part in [str(record.get("ctx_a", "")), str(record.get("ctx_b", ""))]
        if part is not None and str(part).strip()
    )
    if not context:
        raise ValueError(f"example {fallback_id} is missing ctx or ctx_a/ctx_b")

    raw_label = record.get("label")
    label = int(raw_label) if raw_label not in (None, "") else None
    if label is not None and label not in range(4):
        raise ValueError(f"example {fallback_id} label must be 0, 1, 2, or 3")

    raw_id = record.get("ind", record.get("id", fallback_id))
    return Example(
        example_id=str(raw_id),
        context=str(context).strip(),
        endings=[str(ending).strip() for ending in endings],
        label=label,
    )


def load_examples(path: Path) -> list[Example]:
    examples = []
    for index, record in enumerate(load_jsonl(path), start=1):
        try:
            examples.append(coerce_example(record, index))
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
    return examples


def select_examples(
    examples: list[Example], limit: int | None, offset: int, seed: int | None
) -> list[Example]:
    if offset < 0:
        raise SystemExit("--offset must be >= 0")
    selected = examples[:]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(selected)
    selected = selected[offset:]
    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be >= 1")
        selected = selected[:limit]
    return selected


def render_prompt(examples: Iterable[Example]) -> str:
    rows = list(examples)
    if not rows:
        raise SystemExit("No examples selected")

    parts = [
        "You are completing a multiple-choice HellaSwag benchmark batch.",
        "",
        "For each item, pick the ending that most plausibly continues the context.",
        "Return only JSON in this exact shape:",
        '{"answers":[{"id":"EXAMPLE_ID","choice":0}]}',
        "",
        "Rules:",
        "- choice must be one integer: 0, 1, 2, or 3.",
        "- answer every id exactly once.",
        "- do not include explanations, markdown, confidence, or extra keys.",
        "",
        "Items:",
    ]

    for number, example in enumerate(rows, start=1):
        parts.append("")
        parts.append(f"{number}. id: {example.example_id}")
        parts.append(f"context: {example.context}")
        parts.append("endings:")
        for choice, ending in enumerate(example.endings):
            parts.append(f"{choice}: {ending}")

    return "\n".join(parts) + "\n"


def parse_answer_text(text: str) -> dict[str, int]:
    stripped = text.strip()
    matches = JSON_BLOCK_RE.findall(stripped)
    candidates = [match.strip() for match in matches] if matches else [stripped]

    parsed: Any | None = None
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    if parsed is None:
        raise SystemExit("Could not parse answer JSON: " + "; ".join(errors))

    if isinstance(parsed, list):
        answer_rows = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("answers"), list):
        answer_rows = parsed["answers"]
    else:
        raise SystemExit('Answer JSON must be a list or an object with an "answers" list')

    answers: dict[str, int] = {}
    for index, row in enumerate(answer_rows, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"answer row {index} must be an object")
        if "id" not in row or "choice" not in row:
            raise SystemExit(f"answer row {index} must contain id and choice")
        example_id = str(row["id"])
        try:
            choice = int(row["choice"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"answer for {example_id} has non-integer choice") from exc
        if choice not in range(4):
            raise SystemExit(f"answer for {example_id} choice must be 0, 1, 2, or 3")
        if example_id in answers:
            raise SystemExit(f"duplicate answer for id {example_id}")
        answers[example_id] = choice
    return answers


def score_answers(examples: list[Example], answers: dict[str, int]) -> dict[str, Any]:
    labeled = [example for example in examples if example.label is not None]
    if not labeled:
        raise SystemExit("Input examples do not include labels; cannot score")

    expected_ids = {example.example_id for example in labeled}
    answer_ids = set(answers)
    missing = sorted(expected_ids - answer_ids)
    extra = sorted(answer_ids - expected_ids)

    correct = []
    incorrect = []
    for example in labeled:
        if example.example_id not in answers:
            continue
        result = {
            "id": example.example_id,
            "expected": example.label,
            "actual": answers[example.example_id],
        }
        if answers[example.example_id] == example.label:
            correct.append(result)
        else:
            incorrect.append(result)

    attempted = len(correct) + len(incorrect)
    accuracy = (len(correct) / attempted) if attempted else 0.0
    return {
        "total_labeled": len(labeled),
        "attempted": attempted,
        "correct": len(correct),
        "incorrect": len(incorrect),
        "accuracy": accuracy,
        "missing": missing,
        "extra": extra,
        "mistakes": incorrect,
    }


def write_text(path: Path | None, text: str) -> None:
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def download_jsonl(url: str, output: Path, limit: int | None) -> tuple[int, str | None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    rows = 0

    try:
        with urlopen(url, timeout=60) as response, output.open("wb") as handle:
            for line in response:
                if limit is not None and rows >= limit:
                    break
                if not line.strip():
                    continue
                handle.write(line)
                hasher.update(line)
                rows += 1
    except URLError as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    digest = hasher.hexdigest() if limit is None else None
    return rows, digest


def command_fetch(args: argparse.Namespace) -> None:
    rows, digest = download_jsonl(args.url, args.output, args.limit)
    if args.limit is None and digest != HELLASWAG_VAL_SHA256:
        raise SystemExit(
            "Downloaded file checksum did not match expected HellaSwag validation checksum: "
            f"{digest}"
        )
    print(f"Wrote {rows} rows to {args.output}")
    if digest is not None:
        print(f"sha256: {digest}")


def command_make_prompt(args: argparse.Namespace) -> None:
    examples = load_examples(args.input)
    selected = select_examples(examples, args.limit, args.offset, args.seed)
    write_text(args.output, render_prompt(selected))


def command_score(args: argparse.Namespace) -> None:
    examples = load_examples(args.input)
    selected = select_examples(examples, args.limit, args.offset, args.seed)
    answers = parse_answer_text(args.answers.read_text(encoding="utf-8"))
    result = score_answers(selected, answers)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    write_text(args.output, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and score copy/paste chat prompts for HellaSwag."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="download real HellaSwag validation examples"
    )
    fetch_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/hellaswag_val.jsonl"),
        help="where to write downloaded JSONL",
    )
    fetch_parser.add_argument(
        "--limit",
        type=int,
        help="only keep the first N rows; omit to download and verify the full file",
    )
    fetch_parser.add_argument(
        "--url",
        default=HELLASWAG_VAL_URL,
        help="source JSONL URL",
    )
    fetch_parser.set_defaults(func=command_fetch)

    prompt_parser = subparsers.add_parser("make-prompt", help="write a chat prompt")
    prompt_parser.add_argument("input", type=Path, help="HellaSwag JSONL file")
    prompt_parser.add_argument("-o", "--output", type=Path, help="prompt text file")
    prompt_parser.add_argument("--limit", type=int, help="number of examples")
    prompt_parser.add_argument("--offset", type=int, default=0, help="skip examples first")
    prompt_parser.add_argument("--seed", type=int, help="shuffle with a fixed seed first")
    prompt_parser.set_defaults(func=command_make_prompt)

    score_parser = subparsers.add_parser("score", help="score pasted chat answers")
    score_parser.add_argument("input", type=Path, help="same HellaSwag JSONL file")
    score_parser.add_argument("answers", type=Path, help="file containing pasted JSON answer")
    score_parser.add_argument("-o", "--output", type=Path, help="score JSON file")
    score_parser.add_argument("--limit", type=int, help="same limit used for prompt")
    score_parser.add_argument("--offset", type=int, default=0, help="same offset used for prompt")
    score_parser.add_argument("--seed", type=int, help="same seed used for prompt")
    score_parser.set_defaults(func=command_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
