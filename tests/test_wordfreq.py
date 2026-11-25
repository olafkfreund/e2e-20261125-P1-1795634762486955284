"""Unit and CLI tests for the wordfreq package."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from wordfreq.cli import count_words, format_table, main

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parent.parent


def _run_python_module(args: Sequence[str], stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wordfreq", *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


# --- count_words -----------------------------------------------------------


def test_count_words_basic_ranking() -> None:
    text = "Apple banana apple, cherry! Banana apple."
    assert count_words(text) == [("apple", 3), ("banana", 2), ("cherry", 1)]


def test_count_words_breaks_ties_alphabetically() -> None:
    text = "pear apple banana apple pear"
    assert count_words(text) == [("apple", 2), ("pear", 2), ("banana", 1)]


def test_count_words_tie_order_does_not_depend_on_input_order() -> None:
    forward = count_words("x y z x y z")
    reversed_input = count_words("z y x z y x")
    assert forward == reversed_input == [("x", 2), ("y", 2), ("z", 2)]


def test_count_words_top_applies_after_tie_break() -> None:
    assert count_words("b a c a b c d", top=2) == [("a", 2), ("b", 2)]


def test_count_words_top_zero_returns_empty() -> None:
    assert count_words("one two", top=0) == []


def test_count_words_lowercases_by_default() -> None:
    assert count_words("Apple APPLE apple") == [("apple", 3)]


def test_count_words_case_sensitive_keeps_variants_separate() -> None:
    rows = count_words("Apple apple APPLE", ignore_case=False)
    # ASCII ordering: uppercase letters sort before lowercase ones.
    assert rows == [("APPLE", 1), ("Apple", 1), ("apple", 1)]


def test_count_words_min_len_filters_short_tokens() -> None:
    assert count_words("a bb ccc bb", min_len=2) == [("bb", 2), ("ccc", 1)]


def test_count_words_tokens_are_ascii_alphanumeric() -> None:
    # Non-ASCII letters and punctuation split tokens; underscores do too.
    rows = count_words("café naïve 日本 x_y")
    assert rows == [("caf", 1), ("na", 1), ("ve", 1), ("x", 1), ("y", 1)]


def test_count_words_digits_are_tokens_and_sort_as_strings() -> None:
    assert count_words("v2 v2 10 9") == [("v2", 2), ("10", 1), ("9", 1)]


def test_count_words_empty_input_returns_empty_list() -> None:
    assert count_words("") == []


@pytest.mark.parametrize("min_len", [0, -1])
def test_count_words_rejects_min_len_below_one(min_len: int) -> None:
    with pytest.raises(ValueError, match=re.escape("min_len must be >= 1")):
        count_words("hello", min_len=min_len)


def test_count_words_rejects_negative_top() -> None:
    with pytest.raises(ValueError, match=re.escape("top must be >= 0")):
        count_words("hello", top=-1)


# --- format_table ----------------------------------------------------------


def test_format_table_tsv() -> None:
    assert format_table([("apple", 3), ("banana", 2)]) == "apple\t3\nbanana\t2"


def test_format_table_empty_returns_empty_string() -> None:
    assert format_table([]) == ""


# --- main (in-process) -----------------------------------------------------


def test_cli_top_and_min_len(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("to be or not to be that is the question", encoding="utf-8")
    rc = main([str(sample), "--min-len", "3", "--top", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["not\t1", "question\t1"]


def test_cli_tsv_tie_breaking(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample = tmp_path / "ties.txt"
    sample.write_text("pear apple banana apple pear", encoding="utf-8")
    assert main([str(sample)]) == 0
    assert capsys.readouterr().out == "apple\t2\npear\t2\nbanana\t1\n"


def test_cli_case_sensitive_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample = tmp_path / "case.txt"
    sample.write_text("Apple apple APPLE", encoding="utf-8")

    assert main([str(sample)]) == 0
    assert capsys.readouterr().out == "apple\t3\n"

    assert main([str(sample), "--case-sensitive"]) == 0
    assert capsys.readouterr().out == "APPLE\t1\nApple\t1\napple\t1\n"


def test_cli_reads_stdin_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Dog cat dog\n"))
    assert main([]) == 0
    assert capsys.readouterr().out == "dog\t2\ncat\t1\n"


def test_cli_reads_stdin_for_dash_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Dog cat dog\n"))
    assert main(["-"]) == 0
    assert capsys.readouterr().out == "dog\t2\ncat\t1\n"


def test_cli_empty_stdin_tsv_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main([]) == 0
    assert capsys.readouterr().out == ""


def test_cli_tsv_ends_with_single_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "nl.txt"
    sample.write_text("apple banana apple", encoding="utf-8")
    assert main([str(sample)]) == 0
    out = capsys.readouterr().out
    assert out == "apple\t2\nbanana\t1\n"
    assert not out.endswith("\n\n")


def test_cli_json_output_ends_with_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "json.txt"
    sample.write_text("apple banana apple", encoding="utf-8")
    assert main([str(sample), "--json"]) == 0
    out = capsys.readouterr().out
    assert out == json.dumps([{"word": "apple", "count": 2}, {"word": "banana", "count": 1}]) + "\n"
    assert not out.endswith("\n\n")


def test_cli_json_empty_input_emits_empty_array(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["--json"]) == 0
    assert capsys.readouterr().out == "[]\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["--min-len", "0"],
        ["--min-len", "-2"],
        ["--top=-1"],
        ["--top", "abc"],
        ["--no-such-option"],
    ],
)
def test_cli_invalid_arguments_exit_2_with_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    sample = tmp_path / "args.txt"
    sample.write_text("hello world", encoding="utf-8")
    assert main([str(sample), *argv]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert captured.out == ""


def test_cli_missing_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "does-not-exist.txt"
    assert main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert "cannot read" in captured.err
    assert captured.out == ""


def test_cli_invalid_utf8_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    binary = tmp_path / "bad.txt"
    binary.write_bytes(b"\xff\xfe\xfd")
    assert main([str(binary)]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert captured.out == ""


# --- python -m wordfreq (subprocess) ---------------------------------------


def test_python_m_wordfreq_tsv_from_file(tmp_path: Path) -> None:
    sample = tmp_path / "module.txt"
    sample.write_text("pear apple banana apple pear", encoding="utf-8")
    result = _run_python_module([str(sample)])
    assert result.returncode == 0
    assert result.stdout == "apple\t2\npear\t2\nbanana\t1\n"


def test_python_m_wordfreq_reads_stdin_by_default() -> None:
    result = _run_python_module(["--json"], stdin="Dog cat dog\n")
    assert result.returncode == 0
    assert result.stdout == json.dumps([{"word": "dog", "count": 2}, {"word": "cat", "count": 1}]) + "\n"


def test_python_m_wordfreq_missing_file_exits_2(tmp_path: Path) -> None:
    result = _run_python_module([str(tmp_path / "missing.txt")])
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() != ""
