"""Short full-node environment and artifact-access probe; never launches child jobs."""

import json
import os
import platform
import subprocess

import boto3
import torch
from botocore.config import Config
from huggingface_hub import HfApi


def main():
    api = HfApi(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    assert api.whoami()["name"] == "eczech"
    api.repo_info("plantcad/marindna-exp472", repo_type="model")
    print(json.dumps({"task": os.environ["IRIS_TASK_ID"], "tasks": os.environ["IRIS_NUM_TASKS"], "python": platform.python_version(), "torch": torch.__version__, "abi": torch._C._GLIBCXX_USE_CXX11_ABI, "s3_environment_keys": sorted(key for key in os.environ if key.startswith(("AWS_", "S3_", "CW_")))}, indent=2), flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=True)
    s3 = boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}))
    response = s3.list_objects_v2(Bucket="marin-us-east-02a", Prefix="MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/2026.08.28/hf/step-535985/", MaxKeys=10)
    print(json.dumps({"model_objects": [{"key": obj["Key"], "bytes": obj["Size"]} for obj in response.get("Contents", [])]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
