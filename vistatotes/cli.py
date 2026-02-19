"""Command-line interface utilities for VistaTotes."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vistatotes.datasets.loader import load_dataset_from_pickle


def run_autodetect(dataset_path: str, detector_path: str) -> list[dict[str, Any]]:
    """Load a dataset and detector, run the detector, and return positive hits.

    Args:
        dataset_path: Path to a pickle file containing the dataset.
        detector_path: Path to a JSON file containing detector weights and threshold.

    Returns:
        A list of dicts for clips predicted as "Good", each containing
        the clip's ``id``, ``filename``, ``category``, and ``score``.

    Raises:
        FileNotFoundError: If the dataset or detector file does not exist.
        ValueError: If the dataset is empty or the detector file is invalid.
    """
    dataset_file = Path(dataset_path)
    detector_file = Path(detector_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not detector_file.exists():
        raise FileNotFoundError(f"Detector file not found: {detector_path}")

    # Load dataset
    clips: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(dataset_file, clips)

    if not clips:
        raise ValueError(f"No clips loaded from dataset: {dataset_path}")

    # Load detector
    with open(detector_file, "r") as f:
        detector_data = json.load(f)

    if "weights" not in detector_data:
        raise ValueError("Detector file missing 'weights' field")
    if "threshold" not in detector_data:
        raise ValueError("Detector file missing 'threshold' field")

    weights = detector_data["weights"]
    threshold = detector_data["threshold"]

    # Reconstruct the MLP model from weights
    input_dim = len(weights["0.weight"][0])

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    state_dict = {}
    for key, value in weights.items():
        state_dict[key] = torch.tensor(value, dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()

    # Score all clips
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    # Collect positive hits (score >= threshold)
    positive_hits = []
    for cid, score in zip(all_ids, scores):
        if score >= threshold:
            clip = clips[cid]
            positive_hits.append(
                {
                    "id": cid,
                    "filename": clip.get("filename", f"clip_{cid}"),
                    "category": clip.get("category", "unknown"),
                    "score": round(score, 4),
                }
            )

    # Sort by score descending
    positive_hits.sort(key=lambda x: x["score"], reverse=True)

    return positive_hits


def autodetect_main(dataset_path: str, detector_path: str) -> None:
    """CLI entry point: run autodetect and print results to stdout.

    Prints each predicted-Good clip as a line with its filename and score.
    Exits with code 0 on success, 1 on error.

    Args:
        dataset_path: Path to the dataset pickle file.
        detector_path: Path to the detector JSON file.
    """
    try:
        hits = run_autodetect(dataset_path, detector_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not hits:
        print("No items predicted as Good.")
        return

    print(f"Predicted Good ({len(hits)} items):\n")
    for hit in hits:
        print(f"  {hit['filename']}  (score: {hit['score']}, category: {hit['category']})")
