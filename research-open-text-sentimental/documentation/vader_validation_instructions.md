# VADER Validation — Manual Labeling Instructions (Task 3)

## Template

**File:** `data/vader_validation_template.csv`  
**Posts:** 25 (stratified by VADER title sentiment)

## How to label

1. Open `data/vader_validation_template.csv` in Excel, Google Sheets, or a text editor.
2. For each row, read the **title** (and optionally open the **url** to read the post and a few comments).
3. Fill the **human_sentiment** column with exactly one of:
   - **Positive** — overall sentiment is positive (satisfaction, praise, optimism).
   - **Negative** — overall sentiment is negative (frustration, criticism, complaint).
   - **Neutral** — mixed or factual/neutral (no clear positive or negative).
4. Optionally fill **human_notes** with a short note (e.g. "sarcasm", "title misleading").
5. Save the CSV (keep the same columns and headers).

## After labeling

Run the validation script to compute agreement with VADER:

```bash
python3 scripts/validate_vader_agreement.py
```

This produces `documentation/vader_validation_report.md` with accuracy, confusion matrix, and interpretation.
