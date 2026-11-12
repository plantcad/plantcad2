"""Platform-neutral batch-aligned sample partitioning for persistent GPU workers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    task: str
    start: int
    stop: int
    worker: int

    @property
    def key(self) -> str:
        return f"{self.task}--{self.start:06d}-{self.stop:06d}"


def plan_chunks(task_sizes: dict[str, int], workers: int, chunk_size: int = 1024, batch_size: int = 32) -> list[Chunk]:
    """Preserve original minibatches and greedily balance deterministic row ranges."""
    if workers < 1 or batch_size < 1 or chunk_size < batch_size or chunk_size % batch_size:
        raise ValueError("Require positive workers and a chunk size divisible by batch size")
    if not task_sizes or any(size < 1 for size in task_sizes.values()):
        raise ValueError("Every task must have at least one sample")
    ranges = [(task, start, min(start + chunk_size, size)) for task, size in task_sizes.items() for start in range(0, size, chunk_size)]
    if workers > len(ranges):
        raise ValueError("More workers than chunks; reduce chunk size or node count")
    loads = [0] * workers
    chunks = []
    for task, start, stop in sorted(ranges, key=lambda item: -(item[2] - item[1])):
        worker = min(range(workers), key=lambda index: (loads[index], index))
        chunks.append(Chunk(task, start, stop, worker))
        loads[worker] += stop - start
    return chunks


def serialize_plan(chunks: list[Chunk]) -> list[dict]:
    return [asdict(chunk) for chunk in chunks]


def validate_task_coverage(chunks: list[dict], expected_rows: int) -> list[dict]:
    """Reject missing, overlapping, duplicate, or out-of-order row coverage."""
    ordered = sorted(chunks, key=lambda chunk: chunk["start"])
    next_row = 0
    for chunk in ordered:
        if chunk["start"] != next_row or not chunk["start"] < chunk["stop"] <= expected_rows:
            raise ValueError(f"Invalid sample coverage at row {next_row}: {chunk}")
        next_row = chunk["stop"]
    if next_row != expected_rows:
        raise ValueError(f"Incomplete sample coverage: {next_row}/{expected_rows}")
    return ordered
