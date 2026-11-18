# Model Card: Census Income Classifier

Format follows [Mitchell et al., *Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993).

## Model details
- Random forest classifier (scikit-learn, 100 trees, `min_samples_split=20`, `max_features="sqrt"`).
- Built by Ross Fitzgerald as part of the Udacity ML DevOps Engineer Nanodegree.

## Intended use
Educational demonstration of an ML deployment pipeline: predicting whether a person's income is above $50K a year from census attributes. **Not for real decisions** about individuals (lending, hiring and so on).

## Data
[UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset, about 32.5k records after removing duplicates. 14 features (6 numeric, 8 categorical). Label: `<=50K` / `>50K`.

- **Training:** 80%, stratified on the label.
- **Evaluation:** 20% held out.

Categorical features are one-hot encoded. The label is binarized.

## Metrics (test set)
| Precision | Recall | F1 |
|---|---|---|
| 0.781 | 0.651 | 0.710 |

Per-slice metrics for every categorical value are in [`slice_output.txt`](slice_output.txt).

## Ethical considerations
The dataset reflects 1994 US census demographics and contains sensitive attributes (race, sex, native country). Performance varies between slices, and small groups have unstable metrics. No bias mitigation has been applied.

## Caveats and recommendations
- Recall is noticeably lower than precision, so the model misses many high earners.
- Slices with few samples should not be read as reliable.
- Before any real use, add fairness metrics (e.g. equalized odds) and consider rebalancing or reweighting.
