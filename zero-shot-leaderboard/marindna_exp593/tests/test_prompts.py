from exp593_dna.pipeline import load_encoder
from exp593_evals.prompts import matched_fim


def test_matched_context_budget_and_answer_exclusion():
    tokenizer = load_encoder().tokenizer
    values = matched_fim(
        tokenizer,
        "a" * 4096,
        "t" * 4096,
        "cgg",
        {"species": "Zea mays", "assembly": "Zmays"},
    )
    a, b = values["minimal"], values["metadata"]
    assert (a["left"], a["right"]) == (b["left"], b["right"])
    for value in values.values():
        assert len(value["ids"]) <= 8192
        assert value["target_positions"] == list(
            range(value["prompt_length"], len(value["ids"]))
        )
        assert value["ids"][value["prompt_length"] :] == [
            tokenizer.encode(base, add_special_tokens=False).ids[0] for base in "cgg"
        ]
