from exp593_evals.profiles import profile


def test_comparison_profiles_preserve_published_compiler_and_provider_choices():
    fast = profile("fast-fa3")
    ordinary = profile("vllm-fa3")
    torch = profile("torch-fa3")
    assert fast["compile"] == {
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }
    assert fast["freezing"]
    assert ordinary["compile"] == torch["compile"] == {"fullgraph": True}
    assert not ordinary["freezing"] and not torch["freezing"]
    assert ordinary["moe_backend"] == ordinary["attention_provider"] == "vllm"
    assert (
        torch["moe_backend"] == "torch" and torch["attention_provider"] == "flash_attn"
    )
    assert all(
        value["flash_attention_version"] == 3 for value in [fast, ordinary, torch]
    )
