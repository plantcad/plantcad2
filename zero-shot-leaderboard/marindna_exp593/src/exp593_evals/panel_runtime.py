"""Finite native-JAX or compiled-Torch predictions for the same real task panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from itertools import batched
from pathlib import Path

import numpy as np


def native_predictions(record: dict, cases, dna_ids: tuple, callback, total: int):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jmp
    from experiments.grug.moe_hero_ep.weights import restore_weights
    from jax.sharding import PartitionSpec as P
    from levanter.grug.sharding import compact_grug_mesh
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    from exp593_evals.native import model_config

    configure_coreweave_s3()
    if jax.device_count() != 1 or "H100" not in jax.devices()[0].device_kind:
        raise RuntimeError("Native comparison requires one H100 per process")
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                record["checkpoint"],
                record["metadata_canonical_sha256"],
                model_config(),
                mesh,
            )
        )
        jax.block_until_ready(model)
        head = jax.sharding.reshard(model.output_proj, P())

        @eqx.filter_jit
        def forward(model, tokens):
            hidden, routing = model(tokens)
            hidden = jax.sharding.reshard(hidden, P())
            readout_head = jax.sharding.reshard(model.output_proj, P())
            logits = jnp.einsum(
                "bsh,hv->bsv", hidden, readout_head[:, jnp.asarray(dna_ids[:8])]
            )
            return logits, hidden, routing["capacity_overflow_per_layer"]

        @eqx.filter_jit
        def full_projection(hidden, head):
            projected = jnp.einsum(
                "qh,hv->qv",
                hidden.astype(jnp.float32),
                head.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            normalizer = jax.scipy.special.logsumexp(projected, axis=-1)
            return projected[:, jnp.asarray(dna_ids)] - normalizer[:, None]

        for offset, case in enumerate(cases):
            ids = np.asarray(case["ids"], dtype=np.int32)
            tokens = np.zeros((1, 8192), dtype=np.int32)
            tokens[0, : len(ids)] = ids
            logits, hidden, drops = forward(model, jnp.asarray(tokens))
            jax.block_until_ready(logits)
            if not np.isfinite(np.asarray(drops)).all() or np.any(
                np.asarray(drops) != 0
            ):
                raise ValueError("Native model dropped expert assignments")
            positions = np.asarray(case["target_positions"]) - 1
            values = {
                "logits": np.asarray(logits[0, positions], dtype=np.float32),
                "last_hidden": np.asarray(hidden[0, len(ids) - 1], dtype=np.float32),
                "mean_hidden": np.asarray(
                    jnp.mean(hidden[0, : len(ids)].astype(jnp.float32), axis=0)
                ),
            }
            if case["kind"] == "af" and not case["full_vocab"]:
                values["precise_logits"] = np.asarray(
                    jnp.einsum(
                        "qh,hv->qv",
                        hidden[0, positions].astype(jnp.float32),
                        head[:, jnp.asarray(dna_ids[:8])].astype(jnp.float32),
                        precision=jax.lax.Precision.HIGHEST,
                    ),
                    dtype=np.float32,
                )
            if case["full_vocab"]:
                pieces = []
                for begin in range(0, len(positions), 256):
                    points = positions[begin : begin + 256]
                    block = jnp.zeros((256, hidden.shape[-1]), dtype=hidden.dtype)
                    block = block.at[: len(points)].set(hidden[0, points])
                    pieces.append(
                        np.asarray(
                            full_projection(block, head)[: len(points)],
                            dtype=np.float32,
                        )
                    )
                values["full_log_probs"] = np.concatenate(pieces)
            callback(case, values, offset + 1, total)


def torch_predictions(model: Path, profile: str, cases, callback, total: int):
    from exp593_evals.profiles import configure

    configure(profile)
    import torch

    from exp593_evals.legacy_readout import _fp32_projection
    from exp593_evals.torch_scorer import FastScorer

    scorer = FastScorer(model, configuration=profile)
    begin = 0
    for group in batched(cases, scorer.batch_size):
        logits, hidden, _ = scorer.forward([c["ids"] for c in group])
        for row, case in enumerate(group):
            positions = np.asarray(case["target_positions"]) - 1
            n = len(case["ids"])
            values = {
                "logits": logits[row, positions].float().cpu().numpy().copy(),
                "last_hidden": hidden[row, n - 1].float().cpu().numpy().copy(),
                "mean_hidden": hidden[row, :n].float().mean(0).cpu().numpy().copy(),
            }
            if case["kind"] == "af" and not case["full_vocab"]:
                with torch.inference_mode():
                    values["precise_logits"] = (
                        _fp32_projection(
                            hidden[row, positions],
                            scorer.full_head[list(scorer.dna_ids[:8])],
                        )
                        .cpu()
                        .numpy()
                        .copy()
                    )
            if case["full_vocab"]:
                pieces = []
                with torch.inference_mode():
                    for start in range(0, len(positions), 256):
                        projected = _fp32_projection(
                            hidden[row, positions[start : start + 256]],
                            scorer.full_head,
                        )
                        selected = projected[:, list(scorer.dna_ids)] - torch.logsumexp(
                            projected, dim=-1, keepdim=True
                        )
                        pieces.append(selected.cpu().numpy().copy())
                values["full_log_probs"] = np.concatenate(pieces)
            callback(case, values, begin + row + 1, total)
        begin += len(group)


def run(args):
    from exp593_dna.pipeline import load_encoder
    from exp593_evals.readouts import canonical_ids

    tokenizer = load_encoder().tokenizer
    from exp593_evals import panel_stream

    panel = panel_stream.header(args.case_stream)
    if panel_stream.digest(args.panel) != panel["source_sha256"]:
        raise ValueError("Fixture stream differs from its immutable source")
    record = json.loads(args.record.read_text())
    if (
        hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
        != record["bindings"]["tokenizer"]
        or panel["tokenizer_sha256"] != record["bindings"]["tokenizer"]
    ):
        raise ValueError("Panel/tokenizer/training identity differs")
    expected = list(range(args.worker, panel["cases"], args.workers))
    cases = panel_stream.cases(args.case_stream, args.worker, args.workers)
    seen = []
    arrays = {}
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def callback(case, values, completed, total):
        seen.append(case["index"])
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite predictions")
            arrays[f"{case['index']}__{name}"] = value
        if completed % 32 == 0 or completed == total:
            status = {
                "backend": args.backend,
                "worker": args.worker,
                "completed": completed,
                "total": total,
                "seconds": time.time() - start,
            }
            print(json.dumps(status), flush=True)
            status_path = args.output / f"{args.backend}-{args.worker}-progress.json"
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status) + "\n")
            temporary.replace(status_path)

    if args.backend == "native":
        native_predictions(
            record, cases, canonical_ids(tokenizer), callback, len(expected)
        )
    else:
        receipt = json.loads((args.model / "native-receipt.json").read_text())
        if receipt["checkpoint"] != record:
            raise ValueError("Model snapshot differs from panel checkpoint")
        torch_predictions(args.model, args.backend, cases, callback, len(expected))
    if seen != expected:
        raise ValueError("Prediction stream did not cover its exact assigned cases")
    target = args.output / f"{args.backend}-{args.worker}.npz"
    np.savez_compressed(target, **arrays)
    receipt = {
        "backend": args.backend,
        "worker": args.worker,
        "workers": args.workers,
        "checkpoint": record,
        "panel_sha256": panel["source_sha256"],
        "array_sha256": panel_stream.digest(target),
        "cases": seen,
        "elapsed_seconds": time.time() - start,
        "host_process_peak_rss_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024,
        "fullgraph": args.backend != "native",
        "head": "BF16 canonical eight rows; separate FP32 full-vocabulary normalization for AR AF",
    }
    target.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--case-stream", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=["native", "torch-fa3", "vllm-fa3", "fast-fa3"],
        required=True,
    )
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.worker < a.workers:
        raise ValueError("Invalid partition")
    run(a)


if __name__ == "__main__":
    main()
