import copy

import numpy as np
import pytest
from test_scoring import backend, character_tokenizer

from exp586_evals.runtime_validation import synthetic_requests, validate_outputs
from exp586_evals.scoring import paired_probe_feature


def test_paired_probe_uses_fp32_strand_averages_and_alt_minus_ref():
    feature = paired_probe_feature(
        np.asarray([1, 2]), np.asarray([5, 6]), np.asarray([3, 4]), np.asarray([7, 8])
    )
    np.testing.assert_array_equal(feature, [2, 3, 4, 4])
    assert feature.dtype == np.float32
    with pytest.raises(ValueError, match="matching"):
        paired_probe_feature(np.ones(2), np.ones(3), np.ones(2), np.ones(2))


def test_synthetic_validation_consumes_saved_outputs_and_rejects_mismatches():
    tokenizer = character_tokenizer()
    requests = synthetic_requests(tokenizer, bases=64)
    _, score = backend(6)
    logs = score([item["ids"] for item in requests["requests"]])
    features = np.random.default_rng(582).normal(size=(len(logs), 8))
    metrics = validate_outputs(tokenizer, requests, logs, features, features * 2)
    assert metrics["requests"] == len(logs)
    assert np.isfinite(list(metrics.values())).all()
    with pytest.raises(ValueError, match="counts differ"):
        validate_outputs(tokenizer, requests, logs[:-1], features, features)
    bad = copy.deepcopy(requests)
    bad["requests"][0]["ids"][0] = 99
    with pytest.raises(ValueError, match="canonical tokenization"):
        validate_outputs(tokenizer, bad, logs, features, features)
    bad = copy.deepcopy(requests)
    bad["tokenizer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Tokenizer differs"):
        validate_outputs(tokenizer, bad, logs, features, features)
    bad = copy.deepcopy(requests)
    case = bad["llr_cases"][0]
    from exp586_evals.scoring import reverse_complement

    rc = reverse_complement(case["reference"])
    row = next(x for x in bad["requests"] if x["sequence"] == rc)
    assert row["variant_position"] == 31
    row["variant_position"] = 32
    with pytest.raises(ValueError, match="strand transform"):
        validate_outputs(tokenizer, bad, logs, features, features)
