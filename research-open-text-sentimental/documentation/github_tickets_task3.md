# GitHub Tickets — Task 3 (VADER Validation)

---

## Ticket 1 — Task 3: Confusion matrix visualizations

**Title:** `[Research] Task 3 — Generate VADER validation confusion matrices`

**Description:**

Completed **Task 3** (validate VADER with manual labels): generated confusion matrix visualizations comparing human sentiment labels with VADER predictions.

**Deliverables:**
- **`vader_validation_confusion_title.png`** — Confusion matrix: Human vs VADER title label (3×3 heatmap showing agreement/disagreement for Positive/Neutral/Negative).
- **`vader_validation_confusion_author.png`** — Confusion matrix: Human vs VADER author-mean bucket (3×3 heatmap comparing human labels to VADER author sentiment buckets).

**Context:** Manual labels from 25 posts (`data/vader_validation_template.csv`). Script: `scripts/validate_vader_agreement.py`. These visualizations show where VADER agrees/disagrees with human judgment, highlighting potential issues (sarcasm, domain language, context).

**Attachments:** Please attach both PNG files:
- `visualizations/vader_validation_confusion_title.png`
- `visualizations/vader_validation_confusion_author.png`

---

## Ticket 2 — Task 3: VADER validation report

**Title:** `[Research] Task 3 — Generate VADER validation report (accuracy & confusion matrices)`

**Description:**

Completed **Task 3** (validate VADER with manual labels): computed agreement metrics and generated validation report comparing human sentiment labels with VADER predictions.

**Deliverable:** **`vader_validation_report.md`** — Report includes:
- **Accuracy metrics:** VADER title label accuracy (48.0%) and VADER author-mean bucket accuracy (72.0%) vs human labels.
- **Confusion matrices:** Two tables (Human vs VADER title label; Human vs VADER author-mean bucket) showing agreement/disagreement breakdown.
- **Interpretation:** Notes on what disagreements may indicate (sarcasm, domain language, context).

**Context:** Manual labels from 25 posts (stratified: 8 Positive, 8 Negative, 9 Neutral by VADER title). Script: `scripts/validate_vader_agreement.py`. This report quantifies VADER reliability for the sentiment analysis pipeline.

**Attachment:** Please attach `documentation/vader_validation_report.md` to this issue.

---

## Summary

| Ticket | Task | Deliverable | Attach |
|--------|------|------------|--------|
| 1 | Task 3 | 2 confusion matrix PNGs | `vader_validation_confusion_title.png`, `vader_validation_confusion_author.png` |
| 2 | Task 3 | Validation report (markdown) | `vader_validation_report.md` |
