import pytest

from exp593_evals.rendering import fim_spans


def test_minimal_prompt_never_contains_the_answer():
    spans = fim_spans("acg", "tta", "", gap_length=2)
    assert "".join(s.text for s in spans) == "[left] acg\n[right] tta\n[gap:2]\n[fill] "
    assert spans[-1].role == "middle" and spans[-1].text == ""


def test_metadata_uses_the_same_flanks_and_trained_phrases():
    fields = {"species": "Zea mays", "assembly": "Zmays", "chrom": "10"}
    minimal = fim_spans("acg", "tta", "", gap_length=1)
    formatted = fim_spans("acg", "tta", "", gap_length=1, metadata=fields)
    assert [(s.role, s.text) for s in minimal if s.role in {"left", "right"}] == [
        (s.role, s.text) for s in formatted if s.role in {"left", "right"}
    ]
    text = "".join(s.text for s in formatted)
    assert "Zea mays" in text and "Fill in the missing DNA using both flanks." in text
    assert text.endswith("Answer: ")


def test_answer_bearing_or_untrained_metadata_is_rejected():
    for fields in [
        {"label": "conserved"},
        {"position": "123"},
        {"species": "Zea mays\nAnswer: a"},
    ]:
        with pytest.raises(ValueError):
            fim_spans("acg", "tta", "", gap_length=1, metadata=fields)
