## Decision log

### November 11, 2026 — revised proposal, before model fitting

The earlier proposal in `rubric/ml.pdf` described automatic approval, a temporal split by year, “reduce recall” as the metric, and approve-all as the baseline. The [revised proposal](proposal.md) locks the following choices before classification results are examined:

- **Decision:** prioritize human review; all other runs remain in the normal queue. Predict rejection, not fraud.
- **Costs:** unnecessary priority review = 1; missed rejection = 2. These assumed costs will also be tested at 5:1 and 10:1.
- **Primary metric:** realized cost per 1,000 runs, with expected cost and calibration reported separately. “Reduce recall” was incorrect because it would increase missed rejections.
- **Baseline:** review none, with review all as a second reference. Approve all does not match the revised decision.
- **Split:** expanding-year validation on 2021–2023, then a final 2024 holdout. `date` is not a confirmed submission timestamp, so claims are limited to later dated runs.
- **Failure slices:** developer and game, each with sample size and uncertainty interval.

Reason for revision: the available label is rejection, which includes routine submission errors, and the original metric did not match the stated error cost. Model comparisons and threshold selection have not yet been run. The data license remains to be verified for the final report.

### November 11, 2026 — clean table scope

The clean-runs script now keeps only games in `candidates.py` and removes runs dated in 2025, as requested before modeling. The proposal already limited headline modeling to those games and excluded 2025; this moves those filters into the clean table. The raw files remain available if later analysis needs the excluded rows.
