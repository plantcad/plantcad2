import pytest

from exp593_anneal_evals.publication import replace_tables

BASE = """<!-- exp593-anneal-final-results-v1 -->
Introduction.
<!-- primary-results:start -->
Old scores.
<!-- primary-results:end -->
Methods.
<!-- detailed-results:start -->
Old details.
<!-- detailed-results:end -->
"""


def test_manual_introduction_and_methods_edits_survive_score_updates():
    live = BASE.replace("Introduction.", "Human-written introduction.").replace(
        "Methods.", "Human-written methods."
    )
    result = replace_tables(live, BASE, {"primary-results": "New verified scores."})
    assert "Human-written introduction." in result
    assert "Human-written methods." in result
    assert "New verified scores." in result and "Old details." in result


def test_manual_edits_inside_an_owned_table_require_review():
    live = BASE.replace("Old scores.", "Manually revised scores.")
    with pytest.raises(ValueError, match="human edited"):
        replace_tables(live, BASE, {"primary-results": "New verified scores."})
