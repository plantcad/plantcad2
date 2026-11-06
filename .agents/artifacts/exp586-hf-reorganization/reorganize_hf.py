"""Publish exact copies of the selected exp586 exports; preserve the archive."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import RepositoryNotFoundError

HERE = Path(__file__).resolve().parent
ARCHIVE = "marin-dna/marin-dna-exp586"
ARCHIVE_REVISION = "4b8537591261ad14204b28a9c830fa7c6e3f7fbd"
SOURCE_216B_PREFIX = "pretrained-lrm1-seed0/"
RELEASE_216B = "plantcad/marin-dna-exp586-1.5B-60M-216B"
RELEASE_1T = "plantcad/marin-dna-exp586-1.5B-60M"
RELEASE_1T_REVISION = "bd6c44fdc093610f20a892796624de9d96c7ca80"
ARCHIVE_1T_PREFIX = "pretrained-lrm1-seed0-1t/"
EVAL_URL = "https://github.com/plantcad/plantcad2/pull/22"


def get_bytes(repo: str, revision: str, path: str) -> bytes:
    return Path(hf_hub_download(repo_id=repo, revision=revision, filename=path)).read_bytes()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def file_identity(file) -> dict:
    return {"bytes": file.size, "blob_id": file.blob_id,
            "lfs_sha256": file.lfs.sha256 if file.lfs else None}


def verify_copy(api: HfApi, repo: str, revision: str, pairs: list[tuple], prefix: str = "") -> list[dict]:
    files = {file.rfilename: file for file in api.model_info(repo, revision=revision, files_metadata=True).siblings}
    records = []
    for source, target in pairs:
        copied = files[target]
        assert source.size == copied.size, target
        if source.lfs:
            assert copied.lfs and source.lfs.sha256 == copied.lfs.sha256, target
        else:
            assert not copied.lfs and source.blob_id == copied.blob_id, target
        records.append({"source_path": source.rfilename, "destination_path": target, **file_identity(copied)})
    if prefix:
        assert {path for path in files if path.startswith(prefix)} == {
            target for _, target in pairs} | {prefix + "copy-provenance.json"}
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    api = HfApi()
    source_archive = api.model_info(ARCHIVE, revision=ARCHIVE_REVISION, files_metadata=True)
    source_1t = api.model_info(RELEASE_1T, revision=RELEASE_1T_REVISION, files_metadata=True)
    assert not source_archive.private and not source_1t.private
    assert api.model_info(ARCHIVE).sha == ARCHIVE_REVISION, "Archive changed; review current contents before adding files"
    original_manifest = json.loads(get_bytes(ARCHIVE, ARCHIVE_REVISION, "manifest.json"))
    receipt_216b = json.loads(get_bytes(ARCHIVE, ARCHIVE_REVISION, SOURCE_216B_PREFIX + "export-receipt.json"))
    receipt_1t = json.loads(get_bytes(RELEASE_1T, RELEASE_1T_REVISION, "export-receipt.json"))
    assert receipt_216b["input_tokens"] == 216_158_699_520 and receipt_216b["checkpoint_step"] == 412_290
    assert receipt_1t["input_tokens"] == 1_000_000_192_512 and receipt_1t["checkpoint_step"] == 1_907_349
    assert receipt_216b["run_id"] == "exp586-plantcad2-d768-pretrained-lrm1-seed0-v1"
    assert receipt_1t["run_id"] == "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1"
    assert not any(file.rfilename.startswith(ARCHIVE_1T_PREFIX) for file in source_archive.siblings)
    try:
        existing_release = api.model_info(RELEASE_216B).sha
    except RepositoryNotFoundError:
        existing_release = None
    else:
        existing_provenance = json.loads(get_bytes(RELEASE_216B, existing_release, "copy-provenance.json"))
        assert existing_provenance["source_repo"] == ARCHIVE
        assert existing_provenance["source_revision"] == ARCHIVE_REVISION
        assert existing_provenance["source_prefix"] == SOURCE_216B_PREFIX
        assert existing_provenance["destination_repo"] == RELEASE_216B

    created = datetime.now(timezone.utc).isoformat()
    release_216b_card = f'''---
license: apache-2.0
tags:
- biology
- genomics
- dna
- moe
- safetensors
- levanter
---

# MarinDNA exp586 1.5B-60M — 216B tokens

BF16 export of the language-pretrained d768 MoE after 216B DNA tokens: 1.59B total parameters and approximately 60.6M active parameters per token.

- Training input tokens: `216,158,699,520` (10 DNA epochs)
- Final checkpoint: `step-412290`
- Context length: `8192`
- Vocabulary: the original 128,256-token language vocabulary with character-forced DNA inputs
- [Training run](https://forge.coreweave.com/wandb/plantcad/marin-dna/runs/{receipt_216b["run_id"]})

See [MarinDNA issue #586](https://github.com/Open-Athena/marin-dna/issues/586) and the [PlantCAD2 evaluation results]({EVAL_URL}).

The export is copied unchanged from the [research archive](https://huggingface.co/{ARCHIVE}/tree/{ARCHIVE_REVISION}/pretrained-lrm1-seed0).
The [1T-token model](https://huggingface.co/{RELEASE_1T}) is available separately.
'''
    (HERE / "216b-model-card.md").write_text(release_216b_card)
    pairs_216b = [(file, file.rfilename.removeprefix(SOURCE_216B_PREFIX)) for file in source_archive.siblings
                  if file.rfilename.startswith(SOURCE_216B_PREFIX)]
    attributes = next(file for file in source_archive.siblings if file.rfilename == ".gitattributes")
    pairs_216b.append((attributes, ".gitattributes"))
    provenance_216b = {"copied_at": created, "source_repo": ARCHIVE, "source_revision": ARCHIVE_REVISION,
                       "source_prefix": SOURCE_216B_PREFIX, "destination_repo": RELEASE_216B,
                       "input_tokens": receipt_216b["input_tokens"],
                       "files": {target: file_identity(file) for file, target in pairs_216b},
                       "method": "Hugging Face cross-repository copy; model export files unchanged"}
    operations_216b = [CommitOperationCopy(file.rfilename, target, src_repo_id=ARCHIVE,
                                         src_repo_type="model", src_revision=ARCHIVE_REVISION)
                       for file, target in pairs_216b]
    operations_216b.extend([
        CommitOperationAdd("README.md", release_216b_card.encode()),
        CommitOperationAdd("copy-provenance.json", json_bytes(provenance_216b)),
    ])
    if existing_release is None:
        api.create_repo(RELEASE_216B, repo_type="model", private=False, exist_ok=False)
        initial_216b = api.model_info(RELEASE_216B).sha
        result_216b = api.create_commit(RELEASE_216B, operations_216b, parent_commit=initial_216b,
                                        commit_message="Publish the language-pretrained 216B-token exp586 model")
        release_revision = result_216b.oid
    else:
        release_revision = existing_release
    model_pairs_216b = [(file, target) for file, target in pairs_216b if target != ".gitattributes"]
    verified_216b = verify_copy(api, RELEASE_216B, release_revision, model_pairs_216b)
    original_attributes = get_bytes(ARCHIVE, ARCHIVE_REVISION, ".gitattributes").decode().splitlines()
    copied_attributes = get_bytes(RELEASE_216B, release_revision, ".gitattributes").decode().splitlines()
    added_216b_rule = "tokenizer.json filter=lfs diff=lfs merge=lfs -text"
    assert copied_attributes == original_attributes + [added_216b_rule]
    assert get_bytes(RELEASE_216B, release_revision, "README.md").decode() == release_216b_card
    release_receipt = {"repo": RELEASE_216B, "revision": release_revision, "files": verified_216b,
                       "hub_added_lfs_rule": added_216b_rule, "copied_model_files": len(model_pairs_216b), "verified": True}
    (HERE / "216b-copy-receipt.json").write_bytes(json_bytes(release_receipt))
    print(json.dumps({"216b_release": RELEASE_216B, "revision": release_revision, "verified_files": len(verified_216b)}), flush=True)

    manifest = json.loads(json.dumps(original_manifest))
    entry_1t = {key: receipt_1t[key] for key in ["checkpoint", "checkpoint_metadata_digest", "checkpoint_step",
                                               "input_tokens", "run_id", "tokenizer_sha256"]}
    entry_1t["path"] = ARCHIVE_1T_PREFIX
    manifest["models"].append(entry_1t)
    assert manifest["models"][:-1] == original_manifest["models"]
    archive_card = f'''---
license: apache-2.0
tags:
- biology
- genomics
- dna
- moe
- safetensors
- levanter
---

# MarinDNA exp586 archive

BF16 endpoint exports from experiment 586:

| Directory | Initialization and LR | DNA training tokens | Checkpoint | Standalone model |
| --- | --- | ---: | --- | --- |
| `pretrained-lrm0p5-seed0/` | Language-pretrained, 0.5× | 216,158,699,520 | `step-412290` | — |
| `pretrained-lrm1-seed0/` | Language-pretrained, 1× | 216,158,699,520 | `step-412290` | [216B](https://huggingface.co/{RELEASE_216B}) |
| `pretrained-lrm2-seed0/` | Language-pretrained, 2× | 216,158,699,520 | `step-412290` | — |
| `scratch-lrm1-seed0/` | Random, 1× | 216,158,699,520 | `step-412290` | — |
| `pretrained-lrm1-seed0-1t/` | Language-pretrained, 1× | 1,000,000,192,512 | `step-1907349` | [1T](https://huggingface.co/{RELEASE_1T}) |

The 216B-token endpoints have trained for 10 DNA epochs; the 1T-token endpoint has trained for approximately 46.3.
The 1T export is an unchanged copy of the standalone model; per-model receipts preserve the source checkpoints and file identities.

[Experiment issue](https://github.com/Open-Athena/marin-dna/issues/586) · [Evaluation results]({EVAL_URL})
'''
    (HERE / "archive-model-card.md").write_text(archive_card)
    (HERE / "archive-manifest.json").write_bytes(json_bytes(manifest))
    pairs_1t = [(file, ARCHIVE_1T_PREFIX + file.rfilename) for file in source_1t.siblings if file.rfilename != ".gitattributes"]
    provenance_1t = {"copied_at": created, "source_repo": RELEASE_1T, "source_revision": RELEASE_1T_REVISION,
                     "destination_repo": ARCHIVE, "destination_prefix": ARCHIVE_1T_PREFIX,
                     "input_tokens": receipt_1t["input_tokens"],
                     "files": {target: file_identity(file) for file, target in pairs_1t},
                     "method": "Hugging Face cross-repository copy; all source model files unchanged"}
    operations_1t = [CommitOperationCopy(file.rfilename, target, src_repo_id=RELEASE_1T,
                                       src_repo_type="model", src_revision=RELEASE_1T_REVISION)
                     for file, target in pairs_1t]
    operations_1t.extend([
        CommitOperationAdd(ARCHIVE_1T_PREFIX + "copy-provenance.json", json_bytes(provenance_1t)),
        CommitOperationAdd("README.md", archive_card.encode()),
        CommitOperationAdd("manifest.json", json_bytes(manifest)),
    ])
    result_1t = api.create_commit(ARCHIVE, operations_1t, parent_commit=ARCHIVE_REVISION,
                                commit_message="Archive the final 1T-token exp586 export alongside the 216B endpoints")
    verified_1t = verify_copy(api, ARCHIVE, result_1t.oid, pairs_1t, prefix=ARCHIVE_1T_PREFIX)
    final_archive = {file.rfilename: file for file in api.model_info(ARCHIVE, revision=result_1t.oid, files_metadata=True).siblings}
    for file in source_archive.siblings:
        if file.rfilename not in {"README.md", "manifest.json", ".gitattributes"}:
            assert file_identity(file) == file_identity(final_archive[file.rfilename]), file.rfilename
    archive_attributes = get_bytes(ARCHIVE, result_1t.oid, ".gitattributes").decode().splitlines()
    added_1t_rule = ARCHIVE_1T_PREFIX + "tokenizer.json filter=lfs diff=lfs merge=lfs -text"
    assert archive_attributes == original_attributes + [added_1t_rule]
    assert json.loads(get_bytes(ARCHIVE, result_1t.oid, "manifest.json")) == manifest
    assert api.model_info(RELEASE_1T).sha == RELEASE_1T_REVISION, "The standalone 1T source changed"
    archive_receipt = {"repo": ARCHIVE, "previous_revision": ARCHIVE_REVISION, "revision": result_1t.oid,
                       "added_prefix": ARCHIVE_1T_PREFIX, "files": verified_1t,
                       "hub_added_lfs_rule": added_1t_rule,
                       "existing_exports_unchanged": True, "original_manifest_entries_unchanged": True,
                       "source_1t_unchanged": True, "verified": True}
    (HERE / "1t-archive-copy-receipt.json").write_bytes(json_bytes(archive_receipt))
    print(json.dumps({"1t_archive": ARCHIVE, "revision": result_1t.oid, "verified_files": len(verified_1t),
                      "existing_exports_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
