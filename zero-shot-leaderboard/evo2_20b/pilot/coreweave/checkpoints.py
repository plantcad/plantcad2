"""Immutable LR5e-4 / WD0.1 post-cooldown evaluation inputs (stdlib only)."""

CHECKPOINTS = {
    "022t": {
        "run_id": "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2",
        "step": 206144,
        "tokens": 216158699520,
        "revision": "4c71ba81544b93b8a0a0f878b44ac51d1ebb186f",
        "date": "2026.08.20",
    },
    "039t": {
        "run_id": "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1",
        "step": 371065,
        "tokens": 389090902016,
        "revision": "e56696e49dbc4c5d904507983df901fbe9d6d32d",
        "date": "2026.08.25",
    },
    "056t": {
        "run_id": "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1",
        "step": 535985,
        "tokens": 562022055936,
        "revision": "2972ca5abb575ccb9878d2525aafd396e6b73d7c",
        "date": "2026.08.28",
    },
}
DATASET_REPO = "plantcad/PlantCAD2_zero_shot_tasks"
DATASET_REVISION = "d340debe0c8402c84f0696cd2002f87c2f7ba6db"
FULL_ROWS = 1727943


def checkpoint(name: str) -> dict:
    value = dict(CHECKPOINTS[name])
    if not value["revision"]:
        raise ValueError(f"Checkpoint {name} has not been pinned to a verified HF commit")
    value["model_prefix"] = f"{value['run_id']}/hf/step-{value['step']}"
    return value
