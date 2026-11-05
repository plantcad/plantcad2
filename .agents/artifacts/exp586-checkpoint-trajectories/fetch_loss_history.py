"""Read complete loss histories from W&B without sampling or modifying runs."""

import argparse
import gzip
import hashlib
import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import wandb

RUNS = {
    "216b": ("eric-czech/marin/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1", 216_158_699_520),
    "1t": ("plantcad/marin-dna/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1", 1_000_000_192_512),
}


def paged_history(api: wandb.Api, run: object, metric: str) -> Iterator[dict]:
    """Bound each page below the sampling limit, with no overlapping steps.

    W&B's legacy scan implementation uses sampledHistory. Its inclusive bounds
    can exceed the requested row budget by one and silently omit observations.
    Use 5,000-step windows and a 10,000-row budget instead.
    API source: https://github.com/wandb/wandb/blob/v0.22.3/wandb/apis/public/history.py
    """
    from wandb_gql import gql

    query = gql("""
        query LossHistoryPage($entity: String!, $project: String!, $run: String!, $spec: JSONString!) {
            project(name: $project, entityName: $entity) {
                run(name: $run) { sampledHistory(specs: [$spec]) }
            }
        }
    """)
    stop = run.lastHistoryStep + 1
    for start in range(0, stop, 5_000):
        end = min(start + 5_000, stop)
        spec = {"keys": ["_step", "train/input_tokens", metric],
                "minStep": start, "maxStep": end - 1, "samples": 10_000}
        result = api.client.execute(query, variable_values={
            "entity": run.entity, "project": run.project, "run": run.id, "spec": json.dumps(spec),
        })
        rows = result["project"]["run"]["sampledHistory"][0]
        assert len(rows) <= end - start
        assert all(start <= row["_step"] < end for row in rows)
        yield from rows
        if end % 100_000 == 0:
            print(json.dumps({"metric": metric, "scanned_wandb_step": end}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RUNS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader", choices=["sdk", "paged"], default="sdk")
    args = parser.parse_args()
    run_path, expected_tokens = RUNS[args.model]
    api = wandb.Api(timeout=120)
    run = api.run(run_path)
    summary = dict(run.summary)
    assert run.state == "finished" and summary["run_progress"] >= 1
    assert summary["train/input_tokens"] == expected_tokens
    args.output.mkdir(exist_ok=True, parents=True)
    metadata = {
        "run_id": run.id, "wandb_url": run.url,
        "wandb_sdk_version": wandb.__version__,
        "reader": args.reader,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "expected_final_tokens": expected_tokens,
        "expected_last_wandb_step": run.lastHistoryStep,
        "summary": {key: summary[key] for key in ["train/input_tokens", "train/step", "train/loss",
                                                    "eval/loss", "eval/examples", "run_progress"] if key in summary},
        "sampling": "None; complete W&B scan, with page windows strictly below the row budget when using --reader paged",
        "histories": {},
    }
    for kind, metric in [("validation", "eval/loss"), ("training", "train/loss")]:
        path = args.output / f"{args.model}-{kind}.jsonl.gz"
        temporary = path.with_suffix(path.suffix + ".partial")
        count, previous_step, rollbacks, previous_tokens, null_rows = 0, -1, 0, -1, 0
        repeated_boundaries = 0
        last = None
        with gzip.open(temporary, "wt", compresslevel=3) as stream:
            rows = (run.scan_history(keys=["_step", "train/input_tokens", metric], page_size=10_000)
                    if args.reader == "sdk" else paged_history(api, run, metric))
            for row in rows:
                if any(row.get(key) is None for key in ["_step", "train/input_tokens", metric]):
                    null_rows += 1
                    continue
                step, tokens, loss = int(row["_step"]), int(row["train/input_tokens"]), float(row[metric])
                if step == previous_step:
                    assert last == {"wandb_step": step, "input_tokens": tokens, "loss": loss}
                    repeated_boundaries += 1
                    continue
                assert step > previous_step and math.isfinite(loss) and 0 <= tokens <= expected_tokens, (step, previous_step, tokens, loss)
                if tokens < previous_tokens:
                    rollbacks += 1
                last = {"wandb_step": step, "input_tokens": tokens, "loss": loss}
                stream.write(json.dumps(last, separators=(",", ":")) + "\n")
                count += 1
                previous_step, previous_tokens = step, tokens
                if count % 100_000 == 0:
                    print(json.dumps({"model": args.model, "metric": metric, "rows": count,
                                      "last_tokens": tokens}), flush=True)
        assert count > 0 and last is not None
        if kind == "validation":
            assert last["input_tokens"] == expected_tokens
            assert abs(last["loss"] - summary[metric]) < 1e-12
        temporary.replace(path)
        metadata["histories"][kind] = {
            "file": path.name, "rows": count, "last": last, "token_rollbacks": rollbacks,
            "null_metric_or_coordinate_rows_skipped": null_rows,
            "identical_page_boundary_rows_skipped": repeated_boundaries,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
        }
        (args.output / f"{args.model}-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"model": args.model, "metric": metric, **metadata["histories"][kind]}), flush=True)


if __name__ == "__main__":
    main()
