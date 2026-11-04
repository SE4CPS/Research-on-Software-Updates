"""Tightened Cardiff resource proposal — no install, no download, no inference."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    OUT_ROOT,
    RESTRICTED_ROOT,
    load_jsonl,
    utcnow,
    write_json,
)


PINNED_REV = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HUB = Path.home() / ".cache/huggingface/hub/models--cardiffnlp--twitter-roberta-base-sentiment-latest"


def _resolve_size(path: Path) -> int:
    try:
        return path.resolve().stat().st_size if path.exists() else 0
    except OSError:
        return 0


def run_cardiff_proposal(join_report: dict[str, Any]) -> dict[str, Any]:
    dest = OUT_ROOT / "phase3_cardiff_proposal"
    dest.mkdir(parents=True, exist_ok=True)

    snap = HUB / "snapshots" / PINNED_REV
    files = {}
    tokenizer_ok = False
    weights_bytes = 0
    if snap.exists():
        for p in snap.iterdir():
            sz = _resolve_size(p)
            files[p.name] = {"bytes": sz, "resolved": str(p.resolve()) if p.exists() else None}
            if p.name in {"pytorch_model.bin", "model.safetensors"}:
                weights_bytes = max(weights_bytes, sz)
            if p.name in {"tokenizer.json", "vocab.json", "merges.txt", "config.json"}:
                tokenizer_ok = tokenizer_ok or True
        # tokenizer completeness
        needed = {"config.json", "tokenizer.json"}
        present = {p.name for p in snap.iterdir()}
        tokenizer_ok = needed.issubset(present) or {"vocab.json", "merges.txt", "config.json"}.issubset(present)

    # Prefer safetensors snap if pinned missing weights listing
    alt = HUB / "snapshots/d616e2bdfcdb0ca89d9b6efc4909e1db063af290"
    alt_files = {}
    if alt.exists():
        for p in alt.iterdir():
            alt_files[p.name] = _resolve_size(p)

    # License from cached README if present
    license_text = None
    for cand in [
        snap / "README.md" if snap.exists() else None,
        HUB / "snapshots" / PINNED_REV / "README.md",
    ]:
        if cand and cand.exists():
            raw = cand.read_text(encoding="utf-8", errors="replace")[:4000]
            for line in raw.splitlines():
                if line.lower().startswith("license:"):
                    license_text = line.split(":", 1)[1].strip()
                    break
            if license_text is None and "license" in raw.lower():
                license_text = "see_cached_README_license_section"
            break

    free = shutil.disk_usage("/").free
    torch_ok = importlib.util.find_spec("torch") is not None
    tf_ok = importlib.util.find_spec("transformers") is not None
    tok_ok = importlib.util.find_spec("tokenizers") is not None

    # Unit counts from exact join
    units = join_report.get("unit_counts") or {}
    n_posts = int(units.get("original_post") or 3259)
    n_first_a = int(units.get("first_author_reply") or 0)
    n_second_a = int(units.get("second_author_reply") or 0)
    n_later_a = int(units.get("later_author_reply") or 0)
    n_comm = int(units.get("first_community_reply") or 0)
    n_pilot_pool = n_posts + n_first_a + n_second_a + n_later_a + n_comm

    # Length proxy: word counts from analysis units if available
    analysis_path = RESTRICTED_ROOT / "analysis_units_exact_join.jsonl"
    word_proxy = {"label": "whitespace_word_count_proxy_NOT_model_tokens", "n": 0}
    if analysis_path.exists():
        chars = []
        for u in load_jsonl(analysis_path):
            if u.get("unit") in {
                "original_post",
                "first_author_reply",
                "second_author_reply",
                "later_author_reply",
                "first_community_reply",
            }:
                chars.append(int(u.get("text_chars") or 0))
        if chars:
            import numpy as np

            # Rough word proxy: chars/5
            words = [c / 5.0 for c in chars]
            word_proxy = {
                "label": "chars_div_5_proxy_NOT_model_tokens",
                "n": len(chars),
                "mean_chars": float(np.mean(chars)),
                "p50_chars": float(np.median(chars)),
                "p90_chars": float(np.percentile(chars, 90)),
                "max_chars": int(max(chars)),
                "mean_word_proxy": float(np.mean(words)),
                "note": "tokenizer not invoked; torch/transformers not installed",
            }

    # Dependency size estimates (wheels; not installed)
    dep_estimates = {
        "torch_cpu_macos_wheel_gib": "0.15–0.25 (approx; verify at pip download)",
        "transformers_gib": "0.05–0.15",
        "tokenizers_gib": "0.01–0.05",
        "total_install_headroom_gib_recommended": 2.0,
        "weights_already_cached_bytes": weights_bytes or (max(alt_files.values()) if alt_files else 0),
        "weights_already_cached_mib": round(
            (weights_bytes or (max(alt_files.values()) if alt_files else 0)) / (1024**2), 1
        ),
    }

    proposal = {
        "utc": utcnow(),
        "status": "AWAITING_APPROVAL — no torch/transformers install; no inference",
        "model_id": MODEL_ID,
        "pinned_revision": PINNED_REV,
        "alt_snapshot_safetensors": "d616e2bdfcdb0ca89d9b6efc4909e1db063af290",
        "cache_files_pinned": files,
        "cache_files_alt": alt_files,
        "tokenizer_complete_locally": tokenizer_ok,
        "model_card_license": license_text or "NOT_FOUND_IN_CACHE — verify on Hugging Face model card before use",
        "environment": {
            "torch_installed": torch_ok,
            "transformers_installed": tf_ok,
            "tokenizers_installed": tok_ok,
            "free_disk_bytes": free,
            "free_disk_gib": round(free / (1024**3), 3),
        },
        "dependency_size_estimates": dep_estimates,
        "text_units_after_exact_join": {
            "original_posts": n_posts,
            "first_author_replies": n_first_a,
            "second_author_replies": n_second_a,
            "later_author_replies": n_later_a,
            "first_community_replies": n_comm,
            "pilot_pool_total": n_pilot_pool,
            "length_summary": word_proxy,
        },
        "separate_approval_requests": {
            "A_canary_20": {
                "n": 20,
                "seed": 20261103,
                "est_runtime_min_cpu": "2–5 after install",
                "est_ram_gib": "2–3",
                "stopping_checks": ["all 20 produce labels", "no OOM", "license confirmed"],
            },
            "B_bounded_representative_pilot": {
                "n_units": "200–400 stratified by label×unit",
                "compare": ["Cardiff", "VADER", "TextBlob"],
                "identical_text_units": True,
                "include_technical_language_disagreements": True,
                "est_runtime_min_cpu": "15–45",
                "est_ram_gib": "2–4",
                "keep_separate_from_confirmatory": True,
                "stopping_checks": ["hash of unit IDs fixed a priori", "pilot ≠ main results"],
            },
            "C_full_eligible_matched_units": {
                "n_units": n_pilot_pool,
                "est_runtime_min_cpu": "60–180",
                "est_ram_gib": "2–4",
                "est_disk_outputs_mib": "50–200",
                "requires": "prior A+B acceptance + disk≥2 GiB free headroom",
            },
        },
        "money": 0,
        "privacy": "local restricted outputs only; no public reply text",
        "rollback": "delete isolated venv + result shards under phase3_cardiff_*",
        "blockers_now": [
            "torch not installed",
            "transformers not installed",
            "explicit approval required before install/inference",
            f"free disk {round(free/(1024**3),2)} GiB — install needs ~2 GiB headroom recommended",
        ],
    }
    write_json(dest / "CARDIFF_APPROVAL_PROPOSAL_TIGHTENED.json", proposal)

    # Also keep SentiStrength-SE gate note
    write_json(
        dest / "SENTISTRENGTH_SE_GATE.json",
        {
            "utc": utcnow(),
            "status": "BLOCKED — no verified local jar/license; do not substitute SentiSE or generic SentiStrength",
            "approval_gate": "Confirm academic license + obtain SentiStrength-SE package before any run",
        },
    )
    return proposal
