# Entity duration ratio analysis

This analysis measures duration-volume bias after applying the Entity coverage v0.2 `name_and_desc` entity mapping.

For each ground-truth entity in each sample:

```text
entity ratio = union duration of mapped predicted spans / union duration of GT spans
```

- The checkpoint **macro ratio** is the mean of the 106 entity/sample ratios. A missed GT entity or parse-failed sample contributes `0`.
- The **micro ratio** is total mapped predicted union duration divided by total GT union duration. It weights long-duration entities more heavily.
- Predicted entities that do not map to a GT entity have no valid denominator, so they are excluded from the ratios and counted separately.
- Raw span sums and overlap-deduplicated union durations are both retained in `results.json`.

Ground truth: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf at revision `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`.

## Results

The snapshot contains 30 checkpoints, 18 samples per checkpoint, 106 GT entity/sample entries per checkpoint, and 3,180 total entity/checkpoint pairs.

| Statistic across checkpoints | Result |
|---|---:|
| Median macro union ratio | 1.336 |
| Macro union ratio range | 0.690–1.733 |
| Checkpoints with macro ratio above 1 | 23/30 |
| Median micro union ratio | 0.886 |
| Micro union ratio range | 0.759–1.312 |
| Mean missing-GT-entity fraction | 34.7% |
| Mean fraction of entity ratios above 1 | 38.6% |

The macro mean is very sensitive to short GT entities. The largest example is `CrystalFriend` / Jay: 9.1 seconds of GT duration and 546 seconds of predicted duration, producing a 60.0 ratio. This is why a checkpoint can have a macro ratio far above 1 while its micro ratio remains below 1.

## Interpretation

The proposed macro ratio is useful as an **over/under-production diagnostic**, but it should not be treated as a performance score:

- `1.0` only means the predicted and GT duration volumes match; the predicted timestamps may still be wrong.
- Short GT entities can dominate the arithmetic mean.
- Missing entities contribute `0`, while extreme over-prediction has no upper bound.

For a dashboard diagnostic, show the micro ratio together with the missing-entity fraction and Name + appearance IoU. Keep the macro distribution or median available for finding entity-level outliers rather than ranking models by its mean.

## Validation

Re-running the GPT mapper can change an ambiguous entity assignment. The scripts therefore reconstruct the mapping and repair it against two values already saved by the original evaluator for every GT entity: Name + appearance IoU and predicted span count. All 30 checkpoints have `0` remaining IoU mismatches and `0` unresolved mapping errors.

- `report.html`: inspection-friendly checkpoint table and largest entity-level outliers
- `results.json`: complete per-checkpoint, per-sample, and per-entity data
- `build_ground_truth.py`: exports the pinned HF ground truth
- `analyze_entity_duration_ratios.py`: reconstructs mappings and computes statistics
