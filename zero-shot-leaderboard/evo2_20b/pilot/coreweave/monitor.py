"""Read compact global progress inside a live Iris task; no model/GPU imports."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from common import BUCKET, list_keys, result_prefix, storage


def snapshot(job_name: str) -> dict:
    s3 = storage()
    prefix = result_prefix(job_name)
    keys = list_keys(s3, prefix)
    execution_key = f"{prefix}/execution.json"
    execution = json.loads(s3.get_object(Bucket=BUCKET, Key=execution_key)["Body"].read()) if execution_key in keys else {}
    total = execution.get("total_forwards", 400000)
    wanted = [key for key in keys if "/workers/" in key and key.endswith(("/progress.json", "/done.json", "/environment.json"))]

    def read(key):
        return key, json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())

    with ThreadPoolExecutor(max_workers=16) as pool:
        records = dict(pool.map(read, wanted))
    progress = {record["worker"]: record for key, record in records.items() if key.endswith("/progress.json")}
    done = {record["worker"]: record for key, record in records.items() if key.endswith("/done.json")}
    environments = [record for key, record in records.items() if key.endswith("/environment.json")]
    completed = sum(record["completed_forwards"] for worker, record in progress.items() if worker not in done) + sum(record["completed_rows"] * 2 for record in done.values())
    elapsed = time.time() - min(record["started_epoch"] for record in environments) if environments else 0
    rate = completed / elapsed if elapsed else 0
    return {"job": job_name, "completed_forwards": completed, "total_forwards": total, "percent": round(100 * completed / total, 1), "completed_chunks": sum(key.endswith(".json") and "/chunks/" in key for key in keys), "total_chunks": execution.get("total_chunks"), "ready_workers": len(environments), "done_workers": len(done), "flash_verified_workers": sum(record["flash_verification"]["external_flash_attention_2_verified"] for record in environments), "elapsed_seconds": round(elapsed), "aggregate_forwards_per_second": round(rate, 2), "estimated_remaining_seconds": round((total - completed) / rate) if rate else None, "failures": [key for key in keys if key.endswith("/failed.json")], "complete": f"{prefix}/complete.json" in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_name")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.job_name), indent=2))


if __name__ == "__main__":
    main()
