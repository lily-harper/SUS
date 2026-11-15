# SUS: Screening Unverified Speedruns

Cost sensitive machine learning for prioritizing human review of speedrun submissions.

**Lily Holmes · CSCI 5622: Machine Learning · Fall 2026**

## Decision and data

Game moderators decide which submissions receive priority human review. A run that is not flagged remains in the normal review queue. The model does not approve or reject runs. The positive label means a run was historically **rejected**; it does not establish cheating, since rejections can include incomplete or duplicate submissions.

The project uses the [Kaggle Speedrun.com dataset](https://www.kaggle.com/datasets/alexmerren1/speedrun-com-data), reportedly collected from Speedrun.com through an API. The cleaning stage keeps resolved verified and rejected runs dated 2018–2024. It selects eligible games using submissions and reviewer counts through 2020, then keeps runs for those games through 2024. Review outcome fields such as `statusReason`, `verifiedDate`, and `examiner` are excluded from model inputs. The dataset license still needs verification for the final report.

## Evaluation plan

| File | Years | Use |
| --- | --- | --- |
| `data/train.csv` | 2018–2020 | Fit models and preprocessing |
| `data/tune.csv` | 2021 | Choose hyperparameters |
| `data/threshold.csv` | 2022 | Choose a decision threshold for each model |
| `data/selection.csv` | 2023 | Compare models and select one |
| `data/test.csv` | 2024 | Final evaluation of the selected model |

The primary measure is **cost per 1,000 runs**: `1,000 × (false positives + 2 × false negatives) / number of runs`. A false positive is an unnecessary priority review; a false negative is a historically rejected run left in the normal queue. The 2:1 costs are assumptions about moderator effort and delay, not measured costs. The dummy classifier provides a simple comparison; reviewing no runs and reviewing every run are useful cost references. The original assignment proposal in `logs/rubric/ml.pdf` was superseded by the [revised proposal](logs/proposal.md), which defines this human review decision and metric.

The main comparison includes logistic regression, a decision tree, and a random forest. Their preprocessing is fitted on training rows inside each model pipeline. Hyperparameters use the 2021 split, thresholds use 2022, and the lowest cost eligible model on 2023 is saved to `outputs/selected_model_2023.json`. Isolation forest and a Gaussian mixture model are separate experiments; they are not part of that automatic model selection or the 2024 test script.

## Run the project

From the repository root, with Python and the packages in `requirements.txt` installed:

```bash
python -m pip install -r requirements.txt
python pipeline.py
```

If the raw CSV files are absent, `pipeline.py` calls the Kaggle download script, which requires the Kaggle CLI and credentials. It then cleans the data, writes the five split files, tunes and evaluates the three main models, and saves the 2023 selection. The 2024 test is a separate step to run **after** reviewing that selection:

```bash
python pipeline/05_test_2024.py
```

The pipeline also writes threshold tables and 2023 confusion matrix, ROC, and precision–recall plots under `outputs/`. The raw and cleaned CSVs are local data products rather than committed project files.

## Limits and project record

The `date` field describes a run and is not confirmed to be its submission timestamp, so the year split measures performance on later **dated** runs. The 2024 test should be examined once after the model and threshold have been fixed. The current scripts report the assumed 2:1 cost; the [revised proposal](logs/proposal.md) also calls for sensitivity checks at 5:1 and 10:1, calibration, and failure analysis by game and developer. Those analyses are not yet implemented in the main pipeline.

See the [decision log](decision_log.md) for changes to the original proposal and [AI use note](logs/ai_note.md) for the assignment disclosure record.
