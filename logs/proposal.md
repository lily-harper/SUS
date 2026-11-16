# Revised proposal — November 11, 2026


## 1. Decision

Game moderators decide which runs receive **priority human review**. Others stay in the normal queue; the model never approves or rejects runs. The positive class is a historically **rejected** run, not proven fraud. Rejections include duplicate and incomplete submissions.

| Actual outcome | Normal queue | Priority review |
| -------------- | -----------: | --------------: |
| Verified       |            0 |               1 |
| Rejected       |            2 |               0 |

The 2:1 costs represent moderator time versus delayed review. They are assumptions, so I will also test missed-rejection costs of 5 and 10.

## 2. Data

The [Kaggle Speedrun.com Dataset](https://www.kaggle.com/datasets/alexmerren1/speedrun-com-data) contains speedrun.com records reportedly collected through an API. The raw runs table has 4,881,038 rows. Cleaning, candidate selection, and the 2025 exclusion leave 118,064 resolved runs: 105,748 verified and 12,316 rejected. The dataset license needs verification. Unresolved runs are excluded. `examiner`, `verifiedDate`, and `statusReason` reveal later information and are excluded as predictors. Developer and player IDs are not initial predictors.

## 3. Split

The final holdout is **2024**. Expanding validation folds train through 2020/test 2021, through 2021/test 2022, and through 2022/test 2023. Six undated runs and 471 dated in 2025 are excluded. Models and thresholds are chosen before examining 2024 outcomes. `date` appears to describe the run, not necessarily submission time; this tests later **dated** runs only. Fitted preprocessing stays inside each training fold.

## 4. Metric and baseline

The primary metric is **realized cost per 1,000 runs**: `1,000 × (unnecessary reviews + 2 × missed rejections) / runs`. The baseline is **review none**; **review all** is a second reference. Both appear beside results. “Reduce recall” in the prior proposal was wrong: lower rejection recall means more missed rejections. For rejection probability `p`, expected cost is `2p` in the normal queue and `1−p` for priority review. I will tune thresholds on validation data, report expected cost separately, and check calibration. Secondary measures are rejection recall, precision, review rate, and confusion counts. If no model beats review none, I will recommend no model-guided triage at these costs.

## 5. Comparison and failure analysis

I will compare logistic regression, LDA, and random forest on the same split and metric. PCA with LDA and a GMM anomaly review rule will be tested for decision value. **Developer** and **game** are predeclared failure slices. Each will include size, rejected count, cost, and a 95% interval; I will name the worst adequately sized slice. The report will state a recommendation and reversal condition and include the AI appendix.

## 6. Implementation record — November 15, 2026

Sections 1–5 record the decisions made before modeling; this section describes the later implementation and does not retroactively make later choices pre-specified.

The current pipeline uses disjoint calendar-year stages: 2018–2020 for fitting candidate models, 2021 for hyperparameter tuning, 2022 for threshold choice, 2023 for model selection, and 2024 as the final holdout. Preprocessing is fitted within each training stage. The selected supervised model is then refitted on all pre-2024 rows with its saved hyperparameters and threshold before the 2024 evaluation. Game eligibility is calculated using only pre-2021 runs; the current submission cutoff is 1,250 and the examiner cutoff is fewer than 20. These operational cutoffs were not stated in the original proposal. The current split summary contains 615,485 resolved runs across the five stages, so the 118,064 count in section 2 describes an earlier clean-table version and must not be used as the current sample size.

The implemented supervised comparison is logistic regression, a decision tree, and a random forest. The decision tree replaces the proposed LDA; LDA and PCA with LDA have not been implemented. The implemented threshold selector also requires F1 above 0.1 before minimizing cost. That constraint was not specified before modeling, so results should identify it and show the unrestricted cost choice as a sensitivity analysis. A GMM was explored; they are not included in the current supervised model-selection step.
