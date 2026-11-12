"""Task-independent combinations of causal DNA predictions and representations."""

import numpy as np

COMPLEMENT_INDICES = np.asarray([3, 2, 1, 0])


def align_reverse_complement_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Map reverse-complement positions and alleles back to forward orientation."""
    if probabilities.ndim < 2 or probabilities.shape[-1] != 4:
        raise ValueError("Expected probabilities with a final A/C/G/T axis")
    return probabilities[..., ::-1, :][..., COMPLEMENT_INDICES]


def two_sided_probabilities(
    forward: np.ndarray, reverse_complement: np.ndarray, *, forward_weight: float = 0.5
) -> np.ndarray:
    """Normalize a weighted product of left- and right-context probabilities."""
    if not 0 <= forward_weight <= 1 or forward.shape != reverse_complement.shape:
        raise ValueError("Two-sided inputs or weight differ")
    reverse = align_reverse_complement_probabilities(reverse_complement)
    tiny = np.finfo(np.float64).tiny
    logits = forward_weight * np.log(np.maximum(forward, tiny))
    logits += (1 - forward_weight) * np.log(np.maximum(reverse, tiny))
    logits -= logits.max(axis=-1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=-1, keepdims=True)
    return result


def target_log_likelihood_ratio(
    forward: np.ndarray,
    reverse_complement: np.ndarray,
    reference_index: np.ndarray,
    alternate_index: np.ndarray,
    *,
    forward_weight: float = 0.5,
) -> np.ndarray:
    """Score ALT versus REF at one site after two-sided probability fusion."""
    probabilities = two_sided_probabilities(
        forward[:, None, :],
        reverse_complement[:, None, :],
        forward_weight=forward_weight,
    )[:, 0]
    rows = np.arange(len(probabilities))
    return np.log(probabilities[rows, alternate_index]) - np.log(
        probabilities[rows, reference_index]
    )


def directional_probe_feature(
    forward_reference: np.ndarray,
    forward_alternate: np.ndarray,
    reverse_reference: np.ndarray,
    reverse_alternate: np.ndarray,
) -> np.ndarray:
    """Keep both strand views while retaining the historical REF/delta contract."""
    arrays = [
        forward_reference,
        forward_alternate,
        reverse_reference,
        reverse_alternate,
    ]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Directional probe arrays differ")
    return np.concatenate(
        [
            forward_reference,
            forward_alternate - forward_reference,
            reverse_reference,
            reverse_alternate - reverse_reference,
        ],
        axis=-1,
    ).astype(np.float32)
