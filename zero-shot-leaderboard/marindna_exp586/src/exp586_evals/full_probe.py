"""Fit historical and directional frozen-feature probes for one full profile."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp586_evals.bidirectional import directional_probe_feature
from exp586_evals.full_metrics import merge
from exp586_evals.protocol import original


def _validate_roundtrip(estimator, restored, features, expected) -> None:
    """Require exact fitted parameters and float32-stable repeated predictions."""
    fitted_arrays = {
        "scaler": ("mean_", "var_", "scale_"),
        "ridge": ("coef_", "intercept_"),
    }
    for step, attributes in fitted_arrays.items():
        for attribute in attributes:
            np.testing.assert_array_equal(
                getattr(estimator.named_steps[step], attribute),
                getattr(restored.named_steps[step], attribute),
            )
    # Separate BLAS calls may reduce float32 dot products in a different order.
    np.testing.assert_allclose(
        restored.predict(features),
        expected,
        rtol=0,
        atol=8 * np.finfo(np.float32).eps,
    )


def fit_probes(
    preparation_path: Path,
    directory: Path,
    zero_shot_path: Path,
    output: Path,
) -> None:
    protocol = original()
    preparation = json.loads(preparation_path.read_text())
    preparation["samples"] = str(preparation_path.parent.resolve())
    frame = protocol.inputs.read_frame(preparation, {"file": "maize-af.parquet"})
    embeddings = merge(preparation["af_plan"], directory / "chunks", len(frame))
    if not np.array_equal(embeddings.pop("row_id"), frame["row_id"].to_numpy()):
        raise ValueError("Probe embeddings do not align with immutable membership")
    zero_shot = pd.read_parquet(zero_shot_path)
    data = frame.merge(
        zero_shot[["row_id", "llr_acgt_avg"]], on="row_id", validate="1:1"
    ).rename(columns={"llr_acgt_avg": "zero_shot_llr"})
    if not np.array_equal(data["row_id"], frame["row_id"]):
        raise ValueError("Probe join changed row order")

    split_columns = [
        "row_id",
        "block_id",
        "af_bin",
        "stratum",
        "outer_fold",
        "split",
    ]
    split = frame[split_columns]
    train = split["split"].eq("train").to_numpy()
    test = split["split"].eq("test").to_numpy()
    cv_global = protocol.probe.inner_splits(frame, split, protocol.probe.INNER_FOLDS)
    train_rows = np.flatnonzero(train)
    cv = [
        (np.searchsorted(train_rows, fit), np.searchsorted(train_rows, validate))
        for fit, validate in cv_global
    ]
    y_train = data.loc[train, "AF"].to_numpy(dtype=np.float32)
    output.mkdir(parents=True, exist_ok=True)
    fits, metrics = {}, {}
    for representation in protocol.probe.REPRESENTATIONS:
        ref_fwd = embeddings[f"{representation}_ref_fwd"]
        alt_fwd = embeddings[f"{representation}_alt_fwd"]
        ref_rc = embeddings[f"{representation}_ref_rc"]
        alt_rc = embeddings[f"{representation}_alt_rc"]
        averaged_ref = (ref_fwd + ref_rc) * np.float32(0.5)
        averaged_alt = (alt_fwd + alt_rc) * np.float32(0.5)
        contracts = {
            "strand_average": protocol.probe._feature(averaged_ref, averaged_alt),
            "directional": directional_probe_feature(ref_fwd, alt_fwd, ref_rc, alt_rc),
        }
        for contract, features in contracts.items():
            estimator, metadata = protocol.probe._fit_one(features[train], y_train, cv)
            column = f"probe_{representation}_{contract}"
            predictions = estimator.predict(features)
            data[column] = predictions
            estimator_path = output / f"{column}.joblib"
            joblib.dump(estimator, estimator_path, compress=3)
            restored = joblib.load(estimator_path)
            _validate_roundtrip(estimator, restored, features[test], predictions[test])
            key = f"{representation}/{contract}"
            fits[key] = {
                **metadata,
                "features": features.shape[1],
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "feature_contract": (
                    "[mean_strands(REF), mean_strands(ALT)-mean_strands(REF)]"
                    if contract == "strand_average"
                    else "[REF_fwd, ALT_fwd-REF_fwd, REF_rc, ALT_rc-REF_rc]"
                ),
            }
            held_out = data.loc[test]
            metrics[key] = {
                "overall": protocol.probe._bootstrap(held_out, column),
                "by_consequence": {
                    str(name): protocol.probe._bootstrap(group, column)
                    for name, group in held_out.groupby("consequence", sort=True)
                },
            }
            del features
        del averaged_ref, averaged_alt
    columns = [
        "row_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "AF",
        "consequence",
        "block_id",
        "af_bin",
        "outer_fold",
        "split",
        "zero_shot_llr",
        *[name for name in data if name.startswith("probe_")],
    ]
    data[columns].to_parquet(output / "predictions.parquet", index=False)
    split.to_parquet(output / "split.parquet", index=False)
    result = {
        "schema": 1,
        "target": "raw alternate-allele frequency (AF)",
        "split": preparation["af_source"],
        "fits": fits,
        "metrics": metrics,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--zero-shot-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    fit_probes(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
