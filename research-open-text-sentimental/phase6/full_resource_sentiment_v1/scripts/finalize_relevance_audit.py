"""Lock the completed two-reviewer relevance audit; never reads protected outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path('/Users/manumathewjiss/Documents/Codex/2026-10-28/ple/outputs')
SAMPLE_DIR = ROOT / 'restricted/relevance_audit_sample_2026-10-28'
PUBLIC_DIR = ROOT / 'results/relevance_audit_sample_2026-10-28'
FINAL_DIR = ROOT / 'restricted/relevance_audit_final_2026-10-28'
LABELS = {'relevant', 'not_relevant', 'unclear', 'unassessable'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    return rows


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def main() -> None:
    if FINAL_DIR.exists():
        raise SystemExit('Final audit already exists; refusing to overwrite locked files')
    sample = json.loads((PUBLIC_DIR / 'selected_ids.json').read_text())
    sample_hash = sample['selected_ids_sha256']
    if sample_hash != '9f7081261492be2bae5ddac6a825504d811e61568f5923fb1874a89054c24b0a':
        raise ValueError('Selected-ID hash changed')
    key = {x['case_code']: x for x in map(json.loads, (SAMPLE_DIR / 'private_id_key.jsonl').open())}
    if len(key) != 240 or {x['reddit_id'] for x in key.values()} != set(sample['all_ids_sorted']):
        raise ValueError('Private key does not match frozen sample')

    reviews = {}
    review_hashes = {}
    for reviewer in ('A', 'B'):
        source = SOURCE_DIR / f'reviewer_{reviewer}_packet_completed.csv'
        packet = {x['case_code']: x for x in read_csv(SAMPLE_DIR / f'reviewer_{reviewer}_packet.csv')}
        rows = read_csv(source)
        if len(rows) != 240 or len({x['case_code'] for x in rows}) != 240:
            raise ValueError(f'Reviewer {reviewer}: missing or duplicate cases')
        if {x['case_code'] for x in rows} != set(packet):
            raise ValueError(f'Reviewer {reviewer}: case codes changed')
        for row in rows:
            original = packet[row['case_code']]
            if (row['title'], row['body']) != (original['title'], original['body']):
                raise ValueError(f'Reviewer {reviewer}: saved text changed')
            if row['relevance'] not in LABELS or not row['evidence_phrase'].strip():
                raise ValueError(f'Reviewer {reviewer}: invalid or incomplete coding')
        reviews[reviewer] = {x['case_code']: x for x in rows}
        review_hashes[reviewer] = digest(source)

    adjudication = read_csv(SAMPLE_DIR / 'HUMAN_ADJUDICATION_REQUIRED.csv')
    if len(adjudication) != 1 or adjudication[0]['case_code'] != 'C-639299':
        raise ValueError('Unexpected adjudication records')
    decision = adjudication[0]
    if decision['adjudicated_label'] != 'unassessable' or not decision['reason'].strip():
        raise ValueError('Human decision incomplete')

    final_rows = []
    for code in sorted(key):
        a = reviews['A'][code]['relevance']
        b = reviews['B'][code]['relevance']
        if a != b:
            raise ValueError(f'Unadjudicated reviewer disagreement: {code}')
        final = decision['adjudicated_label'] if code == decision['case_code'] else a
        final_rows.append({
            'case_code': code,
            'stratum': key[code]['stratum'],
            'reviewer_A_label': a,
            'reviewer_B_label': b,
            'final_label': final,
            'adjudicated': code == decision['case_code'],
        })

    by_stratum = defaultdict(list)
    for row in final_rows:
        by_stratum[row['stratum']].append(row['final_label'])
    counts = {'october_2026': 20337, 'older': 16566}
    if {s: len(v) for s, v in by_stratum.items()} != {'october_2026': 120, 'older': 120}:
        raise ValueError('Stratum sample sizes changed')
    eligible_n = sum(counts.values())
    weights = {s: n / eligible_n for s, n in counts.items()}
    period = {}
    for s, labels in by_stratum.items():
        tally = Counter(labels)
        period[s] = {
            'sample_n': len(labels),
            'frame_n': counts[s],
            'weight': weights[s],
            'counts': {label: tally[label] for label in sorted(LABELS)},
            'confirmed_relevant_fraction': tally['relevant'] / len(labels),
            'wilson_95': wilson(tally['relevant'], len(labels)),
        }
    point = sum(weights[s] * period[s]['confirmed_relevant_fraction'] for s in counts)
    upper = sum(weights[s] * (period[s]['counts']['relevant'] + period[s]['counts']['unclear'] + period[s]['counts']['unassessable']) / period[s]['sample_n'] for s in counts)
    rng = random.Random(20261028)
    replicates = []
    for _ in range(10000):
        replicates.append(sum(weights[s] * sum(x == 'relevant' for x in rng.choices(by_stratum[s], k=120)) / 120 for s in ('october_2026', 'older')))
    replicates.sort()
    # Linear-interpolated 2.5th and 97.5th percentiles.
    def quantile(p: float) -> float:
        position = (len(replicates) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(replicates) - 1)
        return replicates[lo] + (replicates[hi] - replicates[lo]) * (position - lo)

    label_counts = Counter(row['final_label'] for row in final_rows)
    result = {
        'status': 'final_locked',
        'frame': 'nonprotected Reddit-URL ReleaseTrain candidate capture',
        'capture_n': 37596,
        'protected_excluded_n': 693,
        'eligible_frame_n': eligible_n,
        'sample_n': 240,
        'reviewer_independence': 'reported by user; not independently provable from files',
        'first_pass_agreement_fraction': 1.0,
        'cohens_kappa': 1.0,
        'total_final_counts': {label: label_counts[label] for label in sorted(LABELS)},
        'periods': period,
        'weighted_confirmed_relevant_fraction': point,
        'weighted_upper_if_ambiguous_counted_relevant': upper,
        'weighted_bootstrap_95': [quantile(.025), quantile(.975)],
        'bootstrap_seed': 20261028,
        'bootstrap_replicates': 10000,
        'interval_scope': 'sampling uncertainty within this stratified candidate frame, not labeling or frame-selection uncertainty',
        'adjudicated_case': 'C-639299',
    }
    FINAL_DIR.mkdir(mode=0o700)
    shutil.copy2(SOURCE_DIR / 'reviewer_A_packet_completed.csv', FINAL_DIR / 'reviewer_A_completed.csv')
    shutil.copy2(SOURCE_DIR / 'reviewer_B_packet_completed.csv', FINAL_DIR / 'reviewer_B_completed.csv')
    shutil.copy2(SAMPLE_DIR / 'HUMAN_ADJUDICATION_REQUIRED.csv', FINAL_DIR / 'human_adjudication.csv')
    with (FINAL_DIR / 'final_labels.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    (FINAL_DIR / 'final_statistics.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    files = sorted(x for x in FINAL_DIR.iterdir() if x.is_file())
    manifest = {
        'status': 'final_locked',
        'locked_at_utc': datetime.now(timezone.utc).isoformat(),
        'sample_id_sha256': sample_hash,
        'completed_source_sha256': review_hashes,
        'files': {x.name: digest(x) for x in files},
        'note': 'Read-only local archive; hashes detect later changes. Original completed files were not changed.',
    }
    (FINAL_DIR / 'LOCK_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    for path in FINAL_DIR.iterdir():
        os.chmod(path, 0o400)
    os.chmod(FINAL_DIR, 0o500)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
