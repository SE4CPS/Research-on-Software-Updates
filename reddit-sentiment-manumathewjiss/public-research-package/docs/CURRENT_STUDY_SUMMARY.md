# Current study summary

## Research question

The study examines whether sentiment in Reddit software-update discussions depends on the discussion purpose, speaker, reply stage, elapsed time, text length, and sentiment instrument.

## Dataset summary

- 37,596 candidate posts in the broad capture frame.
- 6,425 preserved original analytical records.
- 12,053 additional purpose-labeled records retained with documented provenance.
- 18,478 unique records in the analytical union.
- 11,051 TPS and 999 strict GDS records in the expanded RQ1 population.
- 6,996 matched original-post and first-author-reply pairs.

## Main findings

- Expanded VADER original-post contrast, GDS minus TPS: +0.161, 95% interval [+0.114, +0.206].
- Length-adjusted VADER contrast: +0.120, 95% interval [+0.074, +0.165].
- First-author-reply change contrast: VADER −0.115, Cardiff −0.174, Senti4SD −0.227, and TextBlob +0.003.
- Expanded author-community contrast: approximately −0.011, with an interval crossing zero.
- Balanced macro-F1 was 0.752 for text alone; adding sentiment did not provide a reliable improvement.

## Interpretation

The results support a context-aware interpretation of sentiment. Sentiment is descriptive and measurement-sensitive; it should not be treated as a universal purpose classifier or as evidence that a technical issue was resolved.

The public summary intentionally omits raw Reddit text, restricted labels, protected evaluation records, and model caches.
