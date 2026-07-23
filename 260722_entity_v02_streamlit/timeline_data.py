"""Build evaluator-consistent entity timelines from prediction fingerprints."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable


NAME_APPEARANCE_IOU_KEY = "entity_coverage::name_appearance_iou"
PERSON_ENTITY_TYPES = {"person", "character"}


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def temporal_iou(
    predicted_intervals: Iterable[tuple[float, float]],
    ground_truth_intervals: Iterable[tuple[float, float]],
) -> float:
    predicted = merge_intervals(predicted_intervals)
    ground_truth = merge_intervals(ground_truth_intervals)
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    intersection = 0.0
    predicted_index = 0
    ground_truth_index = 0
    while predicted_index < len(predicted) and ground_truth_index < len(ground_truth):
        predicted_start, predicted_end = predicted[predicted_index]
        ground_truth_start, ground_truth_end = ground_truth[ground_truth_index]
        intersection += max(
            0.0,
            min(predicted_end, ground_truth_end)
            - max(predicted_start, ground_truth_start),
        )
        if predicted_end <= ground_truth_end:
            predicted_index += 1
        else:
            ground_truth_index += 1
    predicted_duration = sum(end - start for start, end in predicted)
    ground_truth_duration = sum(end - start for start, end in ground_truth)
    union = predicted_duration + ground_truth_duration - intersection
    return intersection / union if union > 0 else 0.0


def predicted_entity_spans(
    payload: dict[str, Any],
) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for shot in payload.get("shot_metadata") or []:
        if not isinstance(shot, dict):
            continue
        try:
            start = float(shot["start_time"])
            end = float(shot["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        for entity in shot.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            if (
                str(entity.get("entity_type") or "").strip().lower()
                not in PERSON_ENTITY_TYPES
            ):
                continue
            name = str(entity.get("canonical_name") or "").strip()
            if name:
                spans[name].append((start, end))
    return dict(spans)


def ground_truth_spans(
    ground_truth: dict[str, Any],
) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for span in ground_truth.get("spans") or []:
        if not isinstance(span, dict):
            continue
        try:
            start = float(span["start"])
            end = float(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        label_id = str(span.get("label_id") or "").strip()
        if label_id and start >= 0 and end > start:
            spans[label_id].append((start, end))
    return dict(spans)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _name_score(predicted_names: Iterable[str], roster_entry: dict[str, Any]) -> int:
    ground_truth_names = {
        _normalized_name(str(value))
        for value in [
            roster_entry.get("label_id"),
            roster_entry.get("name"),
            *(roster_entry.get("aliases") or []),
        ]
        if value
    }
    return sum(
        1
        for predicted_name in predicted_names
        if _normalized_name(predicted_name) in ground_truth_names
    )


def _mapping_error(
    mapping: dict[str, str | None],
    predicted_spans: dict[str, list[tuple[float, float]]],
    ground_truth_by_label: dict[str, list[tuple[float, float]]],
    score_by_label: dict[str, dict[str, Any]],
) -> float:
    mapped_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for predicted_name, label_id in mapping.items():
        if label_id is not None:
            mapped_spans[label_id].extend(predicted_spans[predicted_name])
    error = 0.0
    for label_id, score in score_by_label.items():
        intervals = mapped_spans.get(label_id, [])
        error += abs(
            len(intervals) - int(score.get("name_appearance_predicted_span_count") or 0)
        )
        error += abs(
            temporal_iou(intervals, ground_truth_by_label.get(label_id, []))
            - float(score.get(NAME_APPEARANCE_IOU_KEY) or 0.0)
        )
    return error


def _repair_mapping(
    initial_mapping: dict[str, str | None],
    predicted_spans: dict[str, list[tuple[float, float]]],
    ground_truth_by_label: dict[str, list[tuple[float, float]]],
    score_by_label: dict[str, dict[str, Any]],
) -> tuple[dict[str, str | None], float]:
    mapping = dict(initial_mapping)
    possible_labels: list[str | None] = [None, *score_by_label]
    current_error = _mapping_error(
        mapping, predicted_spans, ground_truth_by_label, score_by_label
    )
    while current_error > 1e-9:
        best_error = current_error
        best_change: tuple[str, str | None] | None = None
        for predicted_name, original_label in mapping.items():
            for candidate_label in possible_labels:
                if candidate_label == original_label:
                    continue
                mapping[predicted_name] = candidate_label
                candidate_error = _mapping_error(
                    mapping, predicted_spans, ground_truth_by_label, score_by_label
                )
                if candidate_error < best_error - 1e-12:
                    best_error = candidate_error
                    best_change = (predicted_name, candidate_label)
            mapping[predicted_name] = original_label
        if best_change is not None:
            mapping[best_change[0]] = best_change[1]
            current_error = best_error
            continue

        predicted_names = list(mapping)
        best_pair: tuple[str, str | None, str, str | None] | None = None
        for first_index, first_name in enumerate(predicted_names):
            first_original_label = mapping[first_name]
            for second_name in predicted_names[first_index + 1 :]:
                second_original_label = mapping[second_name]
                for first_candidate_label in possible_labels:
                    if first_candidate_label == first_original_label:
                        continue
                    mapping[first_name] = first_candidate_label
                    for second_candidate_label in possible_labels:
                        if second_candidate_label == second_original_label:
                            continue
                        mapping[second_name] = second_candidate_label
                        candidate_error = _mapping_error(
                            mapping,
                            predicted_spans,
                            ground_truth_by_label,
                            score_by_label,
                        )
                        if candidate_error < best_error - 1e-12:
                            best_error = candidate_error
                            best_pair = (
                                first_name,
                                first_candidate_label,
                                second_name,
                                second_candidate_label,
                            )
                    mapping[second_name] = second_original_label
                mapping[first_name] = first_original_label
        if best_pair is None:
            break
        mapping[best_pair[0]] = best_pair[1]
        mapping[best_pair[2]] = best_pair[3]
        current_error = best_error
    return mapping, current_error


def _exact_subset_for_label(
    label_id: str,
    mapping: dict[str, str | None],
    predicted_spans: dict[str, list[tuple[float, float]]],
    ground_truth_intervals: list[tuple[float, float]],
    expected_span_count: int,
    expected_iou: float,
    roster_entry: dict[str, Any],
) -> tuple[str, ...] | None:
    available = sorted(
        (
            (name, spans)
            for name, spans in predicted_spans.items()
            if mapping.get(name) in {None, label_id}
            and len(spans) <= expected_span_count
        ),
        key=lambda item: -len(item[1]),
    )
    matches: list[tuple[str, ...]] = []

    def search(
        item_index: int,
        remaining_count: int,
        names: list[str],
        intervals: list[tuple[float, float]],
    ) -> None:
        if remaining_count == 0:
            if (
                abs(temporal_iou(intervals, ground_truth_intervals) - expected_iou)
                <= 1e-8
            ):
                matches.append(tuple(names))
            return
        if item_index == len(available) or remaining_count < 0:
            return
        name, spans = available[item_index]
        if len(spans) <= remaining_count:
            search(
                item_index + 1,
                remaining_count - len(spans),
                [*names, name],
                [*intervals, *spans],
            )
        search(item_index + 1, remaining_count, names, intervals)

    search(0, expected_span_count, [], [])
    if not matches:
        return None
    current_names = {
        name for name, mapped_label in mapping.items() if mapped_label == label_id
    }
    return min(
        matches,
        key=lambda names: (
            -_name_score(names, roster_entry),
            -len(current_names.intersection(names)),
            len(names),
            names,
        ),
    )


def recover_name_appearance_mapping(
    payload: dict[str, Any],
    ground_truth: dict[str, Any],
    character_scores: list[dict[str, Any]],
    initial_mapping: dict[str, str | None] | None = None,
) -> dict[str, list[str]]:
    """Recover the scorer's mapping from saved span-count and IoU fingerprints."""
    predicted_spans = predicted_entity_spans(payload)
    ground_truth_by_label = ground_truth_spans(ground_truth)
    roster_by_label = {
        str(entry.get("label_id")): entry
        for entry in ground_truth.get("roster") or []
        if isinstance(entry, dict) and entry.get("label_id")
    }
    score_by_label = {
        str(score.get("label_id")): score
        for score in character_scores
        if score.get("label_id") and score.get("scored")
    }
    normalized_labels: dict[str, str] = {}
    for label_id, roster_entry in roster_by_label.items():
        for value in [
            label_id,
            roster_entry.get("name"),
            *(roster_entry.get("aliases") or []),
        ]:
            if value:
                normalized_labels[_normalized_name(str(value))] = label_id

    exact_mapping = {
        predicted_name: normalized_labels.get(_normalized_name(predicted_name))
        for predicted_name in predicted_spans
    }
    temporal_mapping = dict(exact_mapping)
    for predicted_name, intervals in predicted_spans.items():
        if temporal_mapping[predicted_name] is not None:
            continue
        best_label = max(
            score_by_label,
            key=lambda label_id: temporal_iou(
                intervals, ground_truth_by_label.get(label_id, [])
            ),
        )
        if temporal_iou(intervals, ground_truth_by_label.get(best_label, [])) > 0:
            temporal_mapping[predicted_name] = best_label

    attempts = [
        *([initial_mapping] if initial_mapping is not None else []),
        exact_mapping,
        temporal_mapping,
        {predicted_name: None for predicted_name in predicted_spans},
    ]
    repaired_attempts = [
        _repair_mapping(attempt, predicted_spans, ground_truth_by_label, score_by_label)
        for attempt in attempts
    ]
    mapping, error = min(repaired_attempts, key=lambda result: result[1])
    if error > 1e-8:
        for label_id, score in score_by_label.items():
            mapped_intervals = [
                interval
                for predicted_name, mapped_label in mapping.items()
                if mapped_label == label_id
                for interval in predicted_spans[predicted_name]
            ]
            expected_span_count = int(
                score.get("name_appearance_predicted_span_count") or 0
            )
            expected_iou = float(score.get(NAME_APPEARANCE_IOU_KEY) or 0.0)
            if (
                len(mapped_intervals) == expected_span_count
                and abs(
                    temporal_iou(
                        mapped_intervals, ground_truth_by_label.get(label_id, [])
                    )
                    - expected_iou
                )
                <= 1e-8
            ):
                continue
            exact_subset = _exact_subset_for_label(
                label_id,
                mapping,
                predicted_spans,
                ground_truth_by_label.get(label_id, []),
                expected_span_count,
                expected_iou,
                roster_by_label.get(label_id, {}),
            )
            if exact_subset is None:
                continue
            for predicted_name, mapped_label in mapping.items():
                if mapped_label == label_id:
                    mapping[predicted_name] = None
            for predicted_name in exact_subset:
                mapping[predicted_name] = label_id
        error = _mapping_error(
            mapping, predicted_spans, ground_truth_by_label, score_by_label
        )
    if error > 1e-8:
        raise ValueError(f"could not recover evaluator mapping (error={error:.9f})")
    by_label: dict[str, list[str]] = defaultdict(list)
    for predicted_name, label_id in mapping.items():
        if label_id is not None:
            by_label[label_id].append(predicted_name)
    for label_id, predicted_names in by_label.items():
        predicted_names.sort(
            key=lambda predicted_name: (
                -_name_score([predicted_name], roster_by_label.get(label_id, {})),
                predicted_name,
            )
        )
    return dict(by_label)


def mapping_fingerprint_error(
    payload: dict[str, Any],
    ground_truth: dict[str, Any],
    character_scores: list[dict[str, Any]],
    mapping: dict[str, list[str]],
) -> float:
    """Return zero when a mapping exactly reproduces saved evaluator details."""
    predicted_spans = predicted_entity_spans(payload)
    inverse_mapping = {
        predicted_name: label_id
        for label_id, predicted_names in mapping.items()
        for predicted_name in predicted_names
    }
    complete_mapping = {
        predicted_name: inverse_mapping.get(predicted_name)
        for predicted_name in predicted_spans
    }
    score_by_label = {
        str(score.get("label_id")): score
        for score in character_scores
        if score.get("label_id") and score.get("scored")
    }
    return _mapping_error(
        complete_mapping,
        predicted_spans,
        ground_truth_spans(ground_truth),
        score_by_label,
    )


def timeline_records(
    payload: dict[str, Any],
    ground_truth: dict[str, Any],
    character_scores: list[dict[str, Any]],
    mapping: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    predicted_spans = predicted_entity_spans(payload)
    ground_truth_by_label = ground_truth_spans(ground_truth)
    mapping = mapping or recover_name_appearance_mapping(
        payload, ground_truth, character_scores
    )
    score_by_label = {str(score.get("label_id")): score for score in character_scores}
    records: list[dict[str, Any]] = []
    roster = [
        entry for entry in ground_truth.get("roster") or [] if isinstance(entry, dict)
    ]
    for entity_index, entry in enumerate(roster):
        label_id = str(entry.get("label_id") or "")
        if label_id not in score_by_label or not score_by_label[label_id].get("scored"):
            continue
        ground_truth_name = str(entry.get("name") or label_id)
        predicted_names = mapping.get(label_id, [])
        prediction_label = (
            ", ".join(predicted_names) if predicted_names else "No matched prediction"
        )
        for source, intervals, lane in (
            (
                "GT",
                ground_truth_by_label.get(label_id, []),
                f"{ground_truth_name} · GT",
            ),
            (
                "Prediction",
                [
                    interval
                    for name in predicted_names
                    for interval in predicted_spans[name]
                ],
                f"{ground_truth_name} ← {prediction_label}",
            ),
        ):
            for start, end in intervals:
                records.append(
                    {
                        "entity_index": entity_index,
                        "ground_truth_name": ground_truth_name,
                        "predicted_names": prediction_label,
                        "source": source,
                        "lane": lane,
                        "start": start,
                        "end": end,
                    }
                )
    duration = max(
        [
            float(ground_truth.get("video_duration") or 0),
            *(record["end"] for record in records),
        ]
    )
    return records, duration
