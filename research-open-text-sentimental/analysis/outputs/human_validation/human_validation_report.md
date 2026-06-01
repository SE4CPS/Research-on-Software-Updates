# Human Validation Report

**Source:** `human_eval_template_80.csv`
**Generated:** 2026-05-27T02:37:03.902171+00:00

**Rows:** 80 | **Label ratings filled:** 80 | **Divergence ratings filled:** 80

## TPS/GDS agreement with gold (`corrected_label`)

- **Accuracy:** 91.2% (73/80)
- **Cohen's κ:** 0.688 (substantial agreement)

| | Human TPS | Human GDS |
|--|-----------|-----------|
| **Gold TPS** | 10 | 3 |
| **Gold GDS** | 4 | 63 |

### Example label disagreements (up to 15)

- **1onnoh7** (r/Android): "Android November security update is out, fixes two vulnerabilities…" gold=TPS human=GDS
- **1ozp5qj** (r/angular): "I still can't get used to it 😀…" gold=TPS human=GDS
- **1oualms** (r/linux): "Firefox 145.0, See All New Features, Updates and Fixes…" gold=TPS human=GDS
- **1oqaia9** (r/immich): "Thumbnails greyed out sometimes- this happen to anyone else?…" gold=GDS human=TPS
- **1qone71** (r/chrome): "this past week whenever I am AFK for a while chrome will open a new tab, type so…" gold=GDS human=TPS
- **1nu697z** (r/vscode): "rich printing different colors depending on if i'm in light or dark mode.…" gold=GDS human=TPS
- **1nqxc6f** (r/chrome): "Trash Samsung ruined chromium based Browsers.…" gold=GDS human=TPS

## Divergence: human vs VADER proxy bucket

- **Exact bucket agreement:** 32.5% (n=80)

VADER bucket = `aligned` if divergence < 0.15 else author/community more negative.
