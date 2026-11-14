"""Explicit lower-only AR/FIM readouts, sharing the frozen fast-fa3 backbone."""

from __future__ import annotations

import numpy as np

from exp593_anneal_evals.exhaustive import (
    align_reverse,
    combine,
    expand,
    observed_scores,
    predictions,
)
from exp593_anneal_evals.prompts import matched_flanks, prompt_ids
from exp593_evals.readouts import normalize_dna


def rc_dna(text: str) -> str:
    return normalize_dna(text, case="lower").translate(str.maketrans("acgtn", "tgcan"))[
        ::-1
    ]


class LowercaseScorer:
    """Reject case folding at the readout boundary; model kernels are unchanged."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.base_ids = tuple(scorer.dna_ids[4:8])
        self.lookup = dict(
            zip("acgtn", (*self.base_ids, scorer.dna_ids[9]), strict=True)
        )
        self.batch_size = scorer.batch_size
        self.tokenizer = scorer.tokenizer
        if len(set(self.lookup.values())) != 5:
            raise ValueError("Lowercase DNA IDs are not distinct")

    def ids(self, text: str) -> list[int]:
        return [self.lookup[b] for b in normalize_dna(text, case="lower")]

    def next_log_probs(self, requests: list[list[int]]) -> np.ndarray:
        import torch

        output = []
        for start in range(0, len(requests), self.batch_size):
            group = requests[start : start + self.batch_size]
            if any(not ids or len(ids) > 8192 for ids in group):
                raise ValueError("Invalid causal prefix length")
            with torch.inference_mode():
                logits, _, lengths = self.scorer.forward(group)
                rows = torch.arange(len(group), device=self.scorer.device)
                # Slice BEFORE softmax: uppercase logits must have no influence.
                selected = logits[rows, lengths[: len(group)] - 1, 4:8].float()
                output.append(
                    torch.log_softmax(selected, dim=-1).cpu().numpy().astype(np.float64)
                )
        return np.concatenate(output)

    def teacher(self, prompts: list[list[int]], answers: list[str]) -> np.ndarray:
        import torch

        if len(prompts) != len(answers) or len({len(a) for a in answers}) != 1:
            raise ValueError(
                "Teacher-forced requests must align and share target length"
            )
        output = []
        for start in range(0, len(prompts), self.batch_size):
            p, a = (
                prompts[start : start + self.batch_size],
                answers[start : start + self.batch_size],
            )
            ids = [
                prefix + self.ids(answer) for prefix, answer in zip(p, a, strict=True)
            ]
            with torch.inference_mode():
                logits, _, _ = self.scorer.forward(ids)
                queries = torch.tensor(
                    [
                        [len(prefix) + i - 1 for i in range(len(answer))]
                        for prefix, answer in zip(p, a, strict=True)
                    ],
                    device=self.scorer.device,
                )
                rows = torch.arange(len(p), device=self.scorer.device)[:, None]
                selected = logits[rows, queries, 4:8].float()
                output.append(
                    torch.softmax(selected, dim=-1).cpu().numpy().astype(np.float64)
                )
        return np.concatenate(output)


def contexts(
    adapter: LowercaseScorer,
    sequences: list[str],
    positions: list[int],
    presentation: str,
):
    if (
        presentation not in {"ar", "fim"}
        or positions != list(range(min(positions), max(positions) + 1))
        or len(positions) not in (1, 2, 3)
    ):
        raise ValueError("Expected AR/FIM and a contiguous1–3-base benchmark target")
    forward, reverse, answers = [], [], []
    begin, end = min(positions), max(positions) + 1
    for sequence in sequences:
        sequence = normalize_dna(sequence, case="lower")
        if len(sequence) != 8192 or not 0 < begin < end < len(sequence):
            raise ValueError("Benchmark source/gap geometry differs")
        left, answer, right = sequence[:begin], sequence[begin:end], sequence[end:]
        answers.append(answer)
        if presentation == "fim":
            left, right = matched_flanks(adapter.tokenizer, left, right, end - begin)
            forward.append(prompt_ids(adapter.tokenizer, left, right, end - begin))
            reverse.append(
                prompt_ids(adapter.tokenizer, rc_dna(right), rc_dna(left), end - begin)
            )
        else:
            forward.append(adapter.ids(left))
            reverse.append(adapter.ids(rc_dna(right)))
    return forward, reverse, answers


def standard(
    adapter: LowercaseScorer,
    sequences: list[str],
    positions: list[int],
    *,
    presentation: str,
    decoding: str,
) -> dict[str, np.ndarray]:
    forward, reverse, answers = contexts(adapter, sequences, positions, presentation)
    length = len(positions)
    if decoding == "tf":
        fwd = adapter.teacher(forward, answers)
        rev = adapter.teacher(reverse, [rc_dna(a) for a in answers])
        return {"left__probs": fwd, "right_reverse_complement__probs": rev}
    if decoding != "exact":
        raise ValueError("No greedy or beam decoding is permitted")
    fwd = expand(forward, length, adapter.base_ids, adapter.next_log_probs)
    rev = expand(reverse, length, adapter.base_ids, adapter.next_log_probs)
    return {"forward__candidate_log_probs": fwd, "rc__candidate_log_probs": rev}


def exact_task_scores(
    arrays: dict[str, np.ndarray], answers: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    length = len(answers[0])
    fwd, rev = arrays["forward__candidate_log_probs"], arrays["rc__candidate_log_probs"]
    distributions = {
        "forward": fwd,
        "reverse_complement": align_reverse(rev, length),
        "forward_rc_product": combine(fwd, rev, length),
    }
    result = {}
    for name, logs in distributions.items():
        result[name] = {
            **observed_scores(logs, answers),
            "exact_match": np.asarray(
                [
                    a == b
                    for a, b in zip(predictions(logs, length), answers, strict=True)
                ]
            ),
        }
    return result


def sv_positions(row: dict) -> tuple[list[int], list[int]]:
    return (
        [
            *range(int(row["left"]) - 6, int(row["left"]) - 1),
            *range(int(row["right"]), int(row["right"]) + 5),
        ],
        list(range(4091, 4101)),
    )


def sv_fim(adapter: LowercaseScorer, rows: list[dict]) -> dict[str, np.ndarray]:
    """One withheld boundary base per query, no joint10-base answer approximation."""
    requests, indices = [], []
    output = {
        name: np.zeros((len(rows), 2, 10, 4), dtype=np.float64)
        for name in ("forward", "reverse_complement")
    }
    for row_id, row in enumerate(rows):
        for allele, (sequence, positions) in enumerate(
            zip((row["RefSeq"], row["MutSeq"]), sv_positions(row), strict=True)
        ):
            sequence = normalize_dna(sequence, case="lower")
            if len(sequence) != 8192:
                raise ValueError("SV source window differs")
            for index, pos in enumerate(positions):
                if pos <= 0:
                    continue
                if pos >= len(sequence) - 1:
                    raise ValueError("SV boundary has no right flank")
                left, right = matched_flanks(
                    adapter.tokenizer, sequence[:pos], sequence[pos + 1 :], 1
                )
                requests.extend(
                    [
                        prompt_ids(adapter.tokenizer, left, right, 1),
                        prompt_ids(adapter.tokenizer, rc_dna(right), rc_dna(left), 1),
                    ]
                )
                indices.extend(
                    [
                        (row_id, allele, index, "forward"),
                        (row_id, allele, index, "reverse_complement"),
                    ]
                )
    probs = np.exp(adapter.next_log_probs(requests))
    for index, probability in zip(indices, probs, strict=True):
        row, allele, position, strand = index
        output[strand][row, allele, position] = (
            probability if strand == "forward" else probability[[3, 2, 1, 0]]
        )
    return sv_reduce(rows, output)


def sv_reduce(rows: list[dict], output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The historical ALT-base boundary reducer, shared with native diagnostics."""
    output = dict(output)
    if "reverse_complement" in output:
        combined = np.sqrt(output["forward"] * output["reverse_complement"])
        combined /= np.maximum(combined.sum(-1, keepdims=True), np.finfo(float).tiny)
        output["forward_rc_product"] = combined
    scores = {}
    for name, values in output.items():
        selected = []
        for i, row in enumerate(rows):
            answer = normalize_dna(row["MutSeq"], case="lower")[4091:4101]
            terms = []
            for j, base in enumerate(answer):
                ref_pos = sv_positions(row)[0][j]
                if base not in "acgt" or ref_pos <= 0:
                    terms.append(0.0)
                else:
                    b = "acgt".index(base)
                    terms.append(
                        np.log(
                            max(values[i, 1, j, b], 1e-12)
                            / max(values[i, 0, j, b], 1e-12)
                        )
                    )
            selected.append(-np.mean(terms))
        scores[name + "__scores"] = np.asarray(selected)
    return scores


def af_contexts(adapter, rows: list[dict], protocol) -> dict:
    """Trim a variant's flanks once, then reverse that exact retained pair."""
    views = {name: ([], [], []) for name in ("fwd", "rc")}
    for row in rows:
        sequence = normalize_dna(row["sequence"], case="lower")
        position = protocol.af.variant_position(8192, "fwd")
        reference, alternate = row["ref"].lower(), row["alt"].lower()
        if (
            len(sequence) != 8192
            or sequence[position] != reference
            or reference not in "acgt"
            or alternate not in "acgt"
            or protocol.af.variant_position(8192, "rc") != 8191 - position
        ):
            raise ValueError("AF source position or alleles differ")
        left, right = matched_flanks(
            adapter.tokenizer, sequence[:position], sequence[position + 1 :], 1
        )
        for strand, l, r, ref, alt in [
            ("fwd", left, right, reference, alternate),
            ("rc", rc_dna(right), rc_dna(left), rc_dna(reference), rc_dna(alternate)),
        ]:
            prompts, refs, alts = views[strand]
            prompts.append(prompt_ids(adapter.tokenizer, l, r, 1))
            refs.append("acgt".index(ref))
            alts.append("acgt".index(alt))
    return views


def af_fim(adapter: LowercaseScorer, frame, protocol) -> dict[str, np.ndarray]:
    """One-base site odds; FP32 lower-only head matches the retained AF contract."""
    import torch

    from exp593_evals.legacy_readout import _fp32_projection

    scorer = adapter.scorer
    result = {"row_id": frame.row_id.to_numpy(dtype=np.int64)}
    views = af_contexts(adapter, frame.to_dict("records"), protocol)
    for strand in ("fwd", "rc"):
        prompts, refs, alts = views[strand]
        bf16, precise, target_probs = [], [], []
        for start in range(0, len(prompts), adapter.batch_size):
            group = prompts[start : start + adapter.batch_size]
            with torch.inference_mode():
                logits, hidden, lengths = scorer.forward(group)
                indices = torch.arange(len(group), device=scorer.device)
                positions = lengths[: len(group)] - 1
                values = logits[indices, positions, 4:8].float()
                target_probs.append(torch.softmax(values, dim=-1).cpu().numpy())
                projected = _fp32_projection(
                    hidden[indices, positions], scorer.full_head[list(adapter.base_ids)]
                )
                r = torch.tensor(refs[start : start + len(group)], device=scorer.device)
                a = torch.tensor(alts[start : start + len(group)], device=scorer.device)
                bf16.extend(
                    (values[indices, a] - values[indices, r])
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                precise.extend(
                    (projected[indices, a] - projected[indices, r])
                    .cpu()
                    .numpy()
                    .astype(float)
                )
        result["target_probs_" + strand] = np.concatenate(target_probs)
        result["llr_acgt_" + strand] = np.asarray(bf16)
        result["llr_full_vocab_" + strand] = np.asarray(precise)
        result["non_acgt_targets_" + strand] = np.zeros(len(frame))
    return result
