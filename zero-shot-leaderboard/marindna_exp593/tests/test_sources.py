import hashlib
import json
from importlib.resources import files


def test_vendored_deployment_and_training_sources_match_their_pins():
    root = files("exp593_evals")
    manifest = json.loads(root.joinpath("sources.json").read_text())
    for family in ("deployment", "training"):
        for relative, expected in manifest[family]["files"].items():
            package, path = relative.split("/", 1)
            assert (
                hashlib.sha256(files(package).joinpath(path).read_bytes()).hexdigest()
                == expected
            )
    assert (
        hashlib.sha256(root.joinpath("legacy_readout.py").read_bytes()).hexdigest()
        == manifest["legacy_readout"]["sha256"]
    )
