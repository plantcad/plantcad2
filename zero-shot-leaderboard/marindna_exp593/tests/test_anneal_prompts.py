from types import SimpleNamespace

import pytest

from exp593_anneal_evals.prompts import fim_spans, matched_flanks, prompt_ids
from exp593_anneal_evals.scoring import af_contexts, contexts, rc_dna


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) for c in text])


class Adapter:
    tokenizer = Tokenizer()

    def ids(self, text):
        return [ord(c) for c in text]


def test_actual_template_and_complete_gap_exclusion():
    tokenizer = Tokenizer()
    left, right = matched_flanks(tokenizer, "A" * 4094, "T" * 4095, 3)
    spans = fim_spans(left, right, 3)
    assert [s.text for s in spans if s.role == "text"] == [
        "[left] ",
        "\n[right] ",
        "\n[gap:3]\n[fill] ",
    ]
    ids = prompt_ids(tokenizer, left, right, 3)
    assert len(ids) + 3 == 8192 and set(left + right) == {"a", "t"}
    assert all(not s.text for s in spans if s.role == "middle")
    assert not any(s.score and s.text for s in spans)


def test_rc_uses_exactly_the_same_retained_flanks_and_never_gold_prefix():
    a = Adapter()
    seq = "a" * 4094 + "gca" + "t" * 4095
    f, r, answers = contexts(a, [seq], [4094, 4095, 4096], "fim")
    assert answers == ["gca"]
    assert all(
        i not in [ord("A"), ord("C"), ord("G"), ord("T")]
        for i in f[0]
        if i in map(ord, "ACGTacgt")
    )
    sf = "".join(map(chr, f[0]))
    sr = "".join(map(chr, r[0]))
    assert "gca" not in sf and "tgc" not in sr
    assert sf.startswith("[left] a") and sr.startswith("[left] a")
    seq2 = seq[:4094] + "ttt" + seq[4097:]
    f2, r2, _ = contexts(a, [seq2], [4094, 4095, 4096], "fim")
    assert f2 == f and r2 == r


def test_ar_is_the_original_target_preceding_prefix_in_each_orientation():
    a = Adapter()
    seq = "a" * 4094 + "gca" + "t" * 4095
    f, r, answers = contexts(a, [seq], [4094, 4095, 4096], "ar")
    assert f == [[ord("a")] * 4094] and r == [[ord("a")] * 4095]
    assert answers == ["gca"]
    with pytest.raises(ValueError):
        contexts(a, [seq], [4094, 4096], "fim")


def test_af_reverses_the_same_trimmed_flanks_and_never_includes_either_allele():
    adapter = Adapter()
    protocol = SimpleNamespace(
        af=SimpleNamespace(
            variant_position=lambda length, strand: 4096 if strand == "fwd" else 4095
        )
    )
    sequence = "a" * 4096 + "g" + "c" * 4095
    a = af_contexts(adapter, [{"sequence": sequence, "ref": "G", "alt": "T"}], protocol)
    b = af_contexts(
        adapter,
        [{"sequence": sequence[:4096] + "t" + sequence[4097:], "ref": "T", "alt": "G"}],
        protocol,
    )
    assert a["fwd"][0] == b["fwd"][0] and a["rc"][0] == b["rc"][0]
    forward, reverse = ("".join(map(chr, a[s][0][0])) for s in ("fwd", "rc"))
    left, rest = forward.removeprefix("[left] ").split("\n[right] ")
    right = rest.split("\n[gap:1]")[0]
    assert (
        reverse
        == "[left] "
        + rc_dna(right)
        + "\n[right] "
        + rc_dna(left)
        + "\n[gap:1]\n[fill] "
    )
    assert a["fwd"][1:] == ([2], [3]) and a["rc"][1:] == ([1], [0])
