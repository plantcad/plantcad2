"""Finite native-JAX weight export and real-input reference generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def export_and_reference(record: dict, output: Path, fixtures: Path) -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jmp
    from exp586_moe.config import d768_config
    from exp586_moe.data import pretrained_dna_tokenizer
    from experiments.grug import checkpointing
    from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
    from experiments.grug.moe_hero_ep.weights import restore_weights
    from levanter.grug.sharding import compact_grug_mesh
    from rigging.filesystem.s3_compat import configure_coreweave_s3
    from rigging.filesystem.storage_path import StoragePath
    from transformers import PreTrainedTokenizerFast

    from exp593_dna.pipeline import load_encoder
    from exp593_evals.readouts import canonical_ids

    configure_coreweave_s3()
    if jax.device_count() != 1 or "H100" not in jax.devices()[0].device_kind:
        raise RuntimeError("Native validation requires exactly one visible H100")
    raw = (StoragePath(record["checkpoint"]) / "metadata.json").read_bytes()
    metadata = json.loads(raw)
    canonical = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        hashlib.sha256(raw).hexdigest() != record["metadata_sha256"]
        or canonical != record["metadata_canonical_sha256"]
    ):
        raise ValueError("Native checkpoint metadata identity changed")
    if (
        metadata["exp593_bindings"] != record["bindings"]
        or metadata["exp593_clock"] != record["clock"]
    ):
        raise ValueError("Native checkpoint bindings or data clock changed")
    config = replace(
        d768_config("pretrained"),
        moe_implementation="sonic",
        expert_chunks=1,
        compact_input_lookup_ids=(),
        compact_input_lookup_one_hot=False,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    output.mkdir(parents=True, exist_ok=True)
    original = load_encoder().tokenizer
    dna_ids = canonical_ids(original)
    original.save(str(output / "original-tokenizer.json"))
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(record["checkpoint"], canonical, config, mesh)
        )
        jax.block_until_ready(model)
        # Independently read the saved compute weights, rather than comparing
        # Torch to a reference derived only from the exported FP32 masters.
        template = eqx.filter_eval_shape(
            Transformer.init, config, key=jax.random.PRNGKey(0)
        )
        saved = checkpointing.load_grug_checkpoint(
            state={
                "params": template,
                "pending_qb_betas": jax.ShapeDtypeStruct((8, 384), jnp.float32),
            },
            candidate=record["checkpoint"],
            mesh=mesh,
            allow_partial=False,
        )
        compute = eqx.filter_jit(policy.cast_to_compute)(
            apply_qb_betas(saved["params"], saved["pending_qb_betas"])
        )
        del saved
        differences = []
        for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(compute), strict=True):
            if eqx.is_array(a):
                differences.append(
                    jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
                )
        weight_difference = float(jnp.max(jnp.stack(differences)))
        if weight_difference != 0:
            raise ValueError(
                f"FP32-master export differs from saved BF16 compute weights: {weight_difference}"
            )
        del compute
        converter = (
            config.hf_checkpoint_converter()
            .replaced(
                tokenizer=PreTrainedTokenizerFast(
                    tokenizer_object=pretrained_dna_tokenizer(),
                    clean_up_tokenization_spaces=False,
                )
            )
            .with_config_overrides(
                {
                    "dtype": "bfloat16",
                    "exp593_checkpoint": record["checkpoint"],
                    "exp593_metadata_sha256": record["metadata_sha256"],
                }
            )
        )
        converter.save_pretrained(
            model,
            str(output),
            dtype=jnp.bfloat16,
            save_reference_code=False,
            max_shard_size=1_000_000_000,
            export_host_budget_bytes=8 << 30,
            max_concurrent_shards=2,
        )
        print(
            json.dumps(
                {
                    "phase": "export_complete",
                    "checkpoint": record["checkpoint"],
                    "original_case_sensitive_ids": dna_ids,
                }
            ),
            flush=True,
        )

        @eqx.filter_jit
        def reference(model, tokens):
            hidden, routing = model(tokens)
            head = model.output_proj[:, jnp.asarray(dna_ids[:8])]
            logits = jnp.einsum("bsh,hv->bsv", hidden, head)
            return logits, hidden, routing

        arrays = {}
        cases = json.loads(fixtures.read_text())["cases"]
        for index, case in enumerate(cases):
            ids = np.asarray(case["ids"], dtype=np.int32)
            tokens = np.zeros((1, 8192), dtype=np.int32)
            tokens[0, : len(ids)] = ids
            logits, hidden, routing = reference(model, jnp.asarray(tokens))
            jax.block_until_ready((logits, hidden))
            drops = np.asarray(routing["capacity_overflow_per_layer"])
            if not np.all(np.isfinite(drops)) or np.any(drops != 0):
                raise ValueError("Native validation dropped expert assignments")
            arrays[f"{index}__logits"] = np.asarray(
                logits[0, : len(ids)], dtype=np.float32
            )
            arrays[f"{index}__last_hidden"] = np.asarray(
                hidden[0, len(ids) - 1], dtype=np.float32
            )
            arrays[f"{index}__mean_hidden"] = np.asarray(
                jnp.mean(hidden[0, : len(ids)].astype(jnp.float32), axis=0)
            )
            arrays[f"{index}__ids"] = ids
            query = np.asarray(case["target_positions"][:4]) - 1
            projected = jnp.einsum(
                "qh,hv->qv",
                hidden[0, query].astype(jnp.float32),
                model.output_proj.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            full_lse = jax.scipy.special.logsumexp(projected, axis=-1)
            arrays[f"{index}__full_log_probs"] = np.asarray(
                projected[:, jnp.asarray(dna_ids[:8])] - full_lse[:, None],
                dtype=np.float32,
            )
            arrays[f"{index}__full_queries"] = query
            print(
                json.dumps(
                    {
                        "phase": "native_reference",
                        "completed": index + 1,
                        "total": len(cases),
                        "case": case["name"],
                    }
                ),
                flush=True,
            )
        np.savez_compressed(output / "native-reference.npz", **arrays)
    receipt = {
        "checkpoint": record,
        "fixtures_sha256": hashlib.sha256(fixtures.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            (output / "native-reference.npz").read_bytes()
        ).hexdigest(),
        "original_tokenizer_sha256": hashlib.sha256(
            (output / "original-tokenizer.json").read_bytes()
        ).hexdigest(),
        "dtype": "bfloat16",
        "master_vs_saved_compute_max_abs": weight_difference,
        "native_reference": "same authoritative effective weights used by the export; pending router bias applied by restore_weights",
        "inference_moe": "sonic",
        "native_device": str(jax.devices()[0]),
        "cases": len(cases),
    }
    (output / "native-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    export_and_reference(
        json.loads(args.checkpoint_record.read_text()), args.output, args.fixtures
    )


if __name__ == "__main__":
    main()
