"""Minimal accuracy metrics used by the released evaluation pipeline."""

import numpy as np


def _accuracy(labels, predictions):
    """Return accuracy as a fraction, using ``0.0`` for an empty input."""
    if labels.size == 0:
        return 0.0
    return float(np.mean(predictions == labels))


def _conditional_accuracy(condition, correct):
    """Return ``P(correct | condition)``, or ``0.0`` for an empty subset."""
    count = int(np.count_nonzero(condition))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(correct & condition) / count)


def build_evaluation_summary(labels, pred_audio, pred_visual, pred_fusion):
    """Build the compact public evaluation summary from class predictions.

    ``fusion_on_audio_only_acc`` is fusion accuracy among samples for which
    only the audio branch is correct. ``fusion_on_visual_only_acc`` is the
    analogous quantity for the visual branch.
    """
    labels = np.asarray(labels).reshape(-1)
    pred_audio = np.asarray(pred_audio).reshape(-1)
    pred_visual = np.asarray(pred_visual).reshape(-1)
    pred_fusion = np.asarray(pred_fusion).reshape(-1)

    expected = labels.shape
    for name, predictions in (
        ("pred_audio", pred_audio),
        ("pred_visual", pred_visual),
        ("pred_fusion", pred_fusion),
    ):
        if predictions.shape != expected:
            raise ValueError(
                f"{name} must contain {labels.size} predictions; "
                f"found {predictions.size}."
            )

    audio_correct = pred_audio == labels
    visual_correct = pred_visual == labels
    fusion_correct = pred_fusion == labels
    audio_only = audio_correct & ~visual_correct
    visual_only = ~audio_correct & visual_correct

    return {
        "acc_audio": _accuracy(labels, pred_audio),
        "acc_visual": _accuracy(labels, pred_visual),
        "acc_fusion": _accuracy(labels, pred_fusion),
        "utility": {
            "fusion_on_audio_only_acc": _conditional_accuracy(
                audio_only, fusion_correct
            ),
            "fusion_on_visual_only_acc": _conditional_accuracy(
                visual_only, fusion_correct
            ),
        },
    }
