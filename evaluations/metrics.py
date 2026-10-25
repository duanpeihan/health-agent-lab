"""Dependency-free metrics. Invocation failures count as incorrect predictions."""

import math
import statistics

from .dataset import LABELS


def _divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarize(rows):
    matrix = {label: {prediction: 0 for prediction in (*LABELS, "error")} for label in LABELS}
    for row in rows:
        matrix[row["expected_label"]][row["predicted_label"]] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = support - tp
        precision, recall = _divide(tp, tp + fp), _divide(tp, support)
        per_class[label] = {
            "precision": precision, "recall": recall,
            "f1": _divide(2 * precision * recall, precision + recall),
            "support": support, "tp": tp, "fp": fp, "fn": fn,
        }
    elapsed = sorted(row["elapsed_seconds"] for row in rows)
    correct = sum(matrix[label][label] for label in LABELS)
    return {
        "total": len(rows), "correct": correct, "accuracy": _divide(correct, len(rows)),
        "error_count": sum(row["predicted_label"] == "error" for row in rows),
        "per_class": per_class, "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
        "confusion_matrix": matrix,
        "latency_seconds": {
            "total": sum(elapsed), "mean": statistics.mean(elapsed) if elapsed else 0.0,
            "p50": statistics.median(elapsed) if elapsed else 0.0,
            "p95": elapsed[max(0, math.ceil(0.95 * len(elapsed)) - 1)] if elapsed else 0.0,
            "max": max(elapsed, default=0.0),
        },
    }
