import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy dependencies so tests run without them installed
import unittest.mock as mock
sys.modules["tavily"] = mock.MagicMock()
sys.modules["groq"] = mock.MagicMock()
sys.modules["dotenv"] = mock.MagicMock()

from orchestrator import parse_followups, validate_citations


def test_parse_followups_extracts_main_report():
    report = "# Title\n\nContent here.\n\n## Follow-up Questions\n- Q1?\n- Q2?"
    main, qs, contested = parse_followups(report)
    assert "Content here." in main
    assert "## Follow-up Questions" not in main


def test_parse_followups_extracts_questions():
    report = "# Title\n\nBody.\n\n## Follow-up Questions\n- First question?\n- Second question?\n- Third question?\n- Fourth question?"
    main, qs, contested = parse_followups(report)
    assert len(qs) == 3
    assert qs[0] == "First question?"


def test_parse_followups_max_three():
    report = "# Title\n\nBody.\n\n## Follow-up Questions\n- A\n- B\n- C\n- D\n- E"
    main, qs, contested = parse_followups(report)
    assert len(qs) == 3


def test_parse_followups_no_followups():
    report = "# Title\n\nBody only."
    main, qs, contested = parse_followups(report)
    assert main == report
    assert qs == []
    assert contested == ""


def test_parse_followups_contested_claims():
    report = "# Title\n\nBody.\n\n## Contested Claims\n- Claim: X\n\n## Follow-up Questions\n- Q1?"
    main, qs, contested = parse_followups(report)
    assert "Claim: X" in contested
    assert "## Contested Claims" not in main


def test_validate_citations_keeps_valid_urls():
    report = "According to [Source A](https://example.com/a), it's true."
    result = validate_citations(report, {"https://example.com/a"})
    assert "Source A" in result
    assert "https://example.com/a" in result


def test_validate_citations_strips_hallucinated_urls():
    report = "According to [Fake Source](https://not-a-source.com), it's true."
    result = validate_citations(report, {"https://example.com/real"})
    assert "not-a-source.com" not in result
    assert "Fake Source" in result


def test_validate_citations_mixed():
    report = (
        "Real [Study](https://example.com/real) and "
        "fake [Hallucinated](https://made-up.com) claims."
    )
    result = validate_citations(report, {"https://example.com/real"})
    assert "Study" in result
    assert "made-up.com" not in result


def test_validate_citations_empty_source_set():
    report = "Some [Claim](https://example.com/a)."
    result = validate_citations(report, set())
    assert "example.com" not in result


def test_validate_citations_no_links():
    report = "Plain text without any citations."
    result = validate_citations(report, {"https://example.com/a"})
    assert result == report


def test_validate_citations_handles_nested_parens():
    report = "See [Paper (2024)](https://example.com/a)."
    result = validate_citations(report, {"https://example.com/a"})
    assert "Paper (2024)" in result
