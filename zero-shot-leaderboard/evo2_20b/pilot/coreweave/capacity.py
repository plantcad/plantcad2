"""Compact H100-only Iris capacity snapshot, following MarinFold's utilization checks."""

import argparse
import json
import subprocess
import time


def summarize(response: dict) -> list[dict]:
    selected = ("cw-us-east-02a", "cw-rno2a")
    peers = {peer["peer_id"]: peer for peer in response["peers"]}
    rows = []
    for name in selected:
        peer = peers[name]
        if not peer["reachable"]:
            raise ValueError(f"Unreachable peer: {name}")
        matched = False
        for backend in peer["backends"]:
            availability = backend["availability"]
            if "h100" not in availability["amounts"]:
                continue
            matched = True
            if availability["version"] not in (2, 3):
                raise ValueError("Unknown Iris availability schema")
            age = time.time() - int(availability["observation_epoch_ms"]) / 1000
            if not 0 <= age <= 90:
                raise ValueError(f"Stale or future capacity reading: {name}, age={age}")
            free = int(availability["amounts"]["h100"])
            total = int(availability["total_amounts"]["h100"])
            held = sum(int(band["amounts"].get("h100", 0)) for band in availability["held_by_band"])
            if min(free, held, total) < 0 or free + held != total:
                raise ValueError(f"Capacity accounting mismatch: {name}")
            rows.append({"cluster": name, "free_h100s": free, "free_node_equivalents": free // 8, "total_h100s": total, "held_h100s": held, "pending_tasks": backend["pending_task_count"], "age_seconds": round(age, 1), "availability_version": availability["version"]})
        if not matched:
            raise ValueError(f"No H100 backend: {name}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-bin", default="iris")
    args = parser.parse_args()
    response = subprocess.run([args.iris_bin, "--cluster", "marin", "rpc", "controller", "list-peers"], check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(summarize(json.loads(response.stdout)), indent=2))


if __name__ == "__main__":
    main()
