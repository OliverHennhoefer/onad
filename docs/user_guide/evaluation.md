# Evaluation

Install evaluation extras first:

```bash
pip install "onad[eval]"
```

## Common metrics

- `average_precision_score` (recommended for imbalanced anomaly data)
- `roc_auc_score` (useful secondary metric)

## Example

```python
from sklearn.metrics import average_precision_score, roc_auc_score

labels, scores = [], []
for x, y in dataset.stream():
    model.learn_one(x)
    labels.append(y)
    scores.append(model.score_one(x))

print("PR-AUC:", average_precision_score(labels, scores))
print("ROC-AUC:", roc_auc_score(labels, scores))
```

## Streaming pitfalls

- Separate warmup from evaluation.
- Do not leak future labels into threshold calibration.
- Compare models under the same stream order and seed.
