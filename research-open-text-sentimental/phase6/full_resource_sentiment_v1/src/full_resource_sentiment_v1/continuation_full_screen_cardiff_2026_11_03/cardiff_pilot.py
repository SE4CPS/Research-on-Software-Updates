"""Cardiff resource check, isolated install, 20-case canary, bounded pilot."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.comments import extract_comment_text
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03 import (
    CARDIFF_MODEL,
    CARDIFF_REV,
    SEED,
)
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.common import (
    CAPTURE_DIR,
    EXACT_JOIN_RESTRICTED,
    OUT_ROOT,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    VENV_CARDIFF,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)

MIN_FREE_GIB = 2.0
CANARY_N = 20
PILOT_N = 300


def _resource_check() -> dict[str, Any]:
    hub = Path.home() / ".cache/huggingface/hub/models--cardiffnlp--twitter-roberta-base-sentiment-latest"
    snap = hub / "snapshots" / CARDIFF_REV
    weights = 0
    files: dict[str, int] = {}
    if snap.exists():
        for p in snap.iterdir():
            try:
                sz = p.resolve().stat().st_size if p.exists() else 0
            except OSError:
                sz = 0
            files[p.name] = sz
            if p.name in {"pytorch_model.bin", "model.safetensors"}:
                weights = max(weights, sz)
    free = shutil.disk_usage("/").free
    return {
        "utc": utcnow(),
        "model": CARDIFF_MODEL,
        "pinned_revision": CARDIFF_REV,
        "license": "cc-by-4.0",
        "license_source": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cache_files_bytes": files,
        "weights_bytes": weights,
        "weights_mib": round(weights / (1024**2), 1),
        "free_disk_gib": round(free / (1024**3), 3),
        "min_free_gib_required": MIN_FREE_GIB,
        "disk_ok": free >= MIN_FREE_GIB * (1024**3),
        "probed_wheels_mib": {"torch_2.9.0_cp314_arm64": 74.4, "transformers": 13.1, "tokenizers": 3.0},
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _ensure_venv(resource: dict[str, Any]) -> dict[str, Any]:
    if not resource["disk_ok"]:
        return {"status": "BLOCKED_DISK", "resource": resource}
    py = sys.executable
    if not VENV_CARDIFF.exists():
        subprocess.run([py, "-m", "venv", str(VENV_CARDIFF)], check=True)
    pip = VENV_CARDIFF / "bin" / "pip"
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True, capture_output=True, text=True)
    wheel_dir = Path("/tmp/cardiff_wheel_probe")
    if wheel_dir.exists() and (wheel_dir / "torch-2.9.0-cp314-cp314-macosx_11_0_arm64.whl").exists():
        install = [
            str(pip),
            "install",
            "--find-links",
            str(wheel_dir),
            "torch==2.9.0",
            "transformers",
            "tokenizers",
            "safetensors",
            "numpy",
        ]
    else:
        install = [
            str(pip),
            "install",
            "torch==2.9.0",
            "transformers",
            "tokenizers",
            "safetensors",
            "numpy",
        ]
    proc = subprocess.run(install, capture_output=True, text=True)
    return {
        "status": "INSTALLED" if proc.returncode == 0 else "INSTALL_FAILED",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2500:],
        "stderr_tail": proc.stderr[-2500:],
        "venv": str(VENV_CARDIFF),
    }


def _select_units(n: int, seed: int) -> list[dict[str, Any]]:
    units = [
        u
        for u in load_jsonl(EXACT_JOIN_RESTRICTED / "analysis_units_exact_join.jsonl")
        if u.get("unit")
        in {
            "original_post",
            "first_author_reply",
            "second_author_reply",
            "later_author_reply",
            "first_community_reply",
        }
        and u.get("label") in {"TPS", "GDS"}
    ]
    rng = np.random.default_rng(seed)
    by_key: dict[tuple[str, str], list] = {}
    for u in units:
        by_key.setdefault((u["label"], u["unit"]), []).append(u)
    keys = sorted(by_key.keys())
    selected: list[dict[str, Any]] = []
    per = max(1, n // max(len(keys), 1))
    for k in keys:
        pool = by_key[k]
        take = min(per, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        selected.extend(pool[int(i)] for i in idx)
    if len(selected) > n:
        idx = rng.choice(len(selected), size=n, replace=False)
        selected = [selected[int(i)] for i in sorted(idx)]
    elif len(selected) < n:
        chosen = {(u["reddit_id"], u["unit"], u.get("original_index")) for u in selected}
        remaining = [
            u
            for u in units
            if (u["reddit_id"], u["unit"], u.get("original_index")) not in chosen
        ]
        need = n - len(selected)
        if remaining:
            idx = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            selected.extend(remaining[int(i)] for i in idx)
    return selected


def _attach_texts(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    matched = load_jsonl(EXACT_JOIN_RESTRICTED / "matched_units_exact_join.jsonl")
    meta = {(m["reddit_id"], m.get("original_index")): m for m in matched}
    need_ids = {u["reddit_id"] for u in selected if u["unit"] != "original_post"}
    pages_needed = {
        int(meta[(u["reddit_id"], u.get("original_index"))]["capture_page"])
        for u in selected
        if u["unit"] != "original_post" and (u["reddit_id"], u.get("original_index")) in meta
    }
    cap: dict[str, list] = {}
    for page in pages_needed:
        path = CAPTURE_DIR / f"page_{page:05d}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid in need_ids:
                cap[rid] = [c for c in (row.get("comments") or []) if isinstance(c, dict)]

    out = []
    for u in selected:
        if u["unit"] == "original_post":
            doc = texts.get(u["reddit_id"]) or {}
            text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        else:
            text = ""
            m = meta.get((u["reddit_id"], u.get("original_index")))
            if m and u["reddit_id"] in cap:
                oi = int(m["original_index"])
                comments = cap[u["reddit_id"]]
                if 0 <= oi < len(comments):
                    text, _ = extract_comment_text(comments[oi])
                    text = text or ""
        row = {
            "reddit_id": u["reddit_id"],
            "label": u["label"],
            "unit": u["unit"],
            "original_index": u.get("original_index"),
            "vader": u.get("vader"),
            "textblob": u.get("textblob"),
            "text_chars": len(text),
            "text_hash": sha256_text(text),
            "text": text,
        }
        out.append(row)
    return out


def _run_inference_script(units_path: Path, out_path: Path) -> dict[str, Any]:
    script = OUT_ROOT / "phase3_cardiff" / "_infer_worker.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"""#!/usr/bin/env python3
import json, time, resource
from pathlib import Path
MODEL = "{CARDIFF_MODEL}"
REV = "{CARDIFF_REV}"
units = [json.loads(l) for l in open(r"{units_path}") if l.strip()]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, local_files_only=True)
model.eval()
load_s = time.time() - t0
id2label = {{int(k): v for k, v in model.config.id2label.items()}} if model.config.id2label else {{}}
rows = []
max_rss = 0
for u in units:
    text = u.get("text") or ""
    t1 = time.time()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    n_tok = int(enc["input_ids"].shape[1])
    full_len = len(tok(text, truncation=False)["input_ids"]) if text else 0
    truncated = full_len > 512
    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
    pred_i = int(logits.argmax())
    label = id2label.get(pred_i, str(pred_i))
    name = str(label).lower()
    if "neg" in name:
        pol = -1.0
    elif "pos" in name:
        pol = 1.0
    else:
        pol = 0.0
    latency = time.time() - t1
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_rss = max(max_rss, rss)
    except Exception:
        rss = None
    rows.append({{
        "reddit_id": u.get("reddit_id"),
        "label": u.get("label"),
        "unit": u.get("unit"),
        "vader": u.get("vader"),
        "textblob": u.get("textblob"),
        "text_hash": u.get("text_hash"),
        "text_chars": u.get("text_chars"),
        "cardiff_label": str(label),
        "cardiff_polarity_sign": pol,
        "cardiff_probs": probs,
        "n_tokens": n_tok,
        "truncated": bool(truncated),
        "latency_s": latency,
        "text_chars_runtime": len(text),
    }})
meta = {{
    "load_seconds": load_s,
    "n": len(rows),
    "max_rss_raw": max_rss,
    "rss_note": "ru_maxrss units platform-dependent (bytes on macOS)",
    "model": MODEL,
    "revision": REV,
}}
Path(r"{out_path}").write_text("\\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\\n")
Path(r"{out_path}").with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
print(json.dumps({{"ok": True, "n": len(rows), "load_s": load_s}}))
""",
        encoding="utf-8",
    )
    py = VENV_CARDIFF / "bin" / "python"
    proc = subprocess.run([str(py), str(script)], capture_output=True, text=True, timeout=3600)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-1500:],
        "stderr": proc.stderr[-2500:],
    }


def _compare_instruments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def sign(x):
        if x is None:
            return None
        return int(np.sign(float(x)))

    pairs = {"cardiff_vs_vader": 0, "cardiff_vs_textblob": 0, "vader_vs_textblob": 0, "n": 0}
    tech_disagree = []
    for r in rows:
        if r.get("vader") is None or r.get("textblob") is None:
            continue
        pairs["n"] += 1
        cs, vs, ts = sign(r.get("cardiff_polarity_sign")), sign(r["vader"]), sign(r["textblob"])
        if cs is not None and vs is not None and cs == vs:
            pairs["cardiff_vs_vader"] += 1
        if cs is not None and ts is not None and cs == ts:
            pairs["cardiff_vs_textblob"] += 1
        if vs is not None and ts is not None and vs == ts:
            pairs["vader_vs_textblob"] += 1
        text_chars = int(r.get("text_chars_runtime") or r.get("text_chars") or 0)
        if cs != vs and text_chars > 400:
            tech_disagree.append(
                {
                    "reddit_id": r["reddit_id"],
                    "unit": r["unit"],
                    "cardiff": r.get("cardiff_label"),
                    "vader_sign": vs,
                    "textblob_sign": ts,
                    "text_chars": text_chars,
                }
            )
    n = max(pairs["n"], 1)
    return {
        "n_with_all_scores": pairs["n"],
        "sign_agreement_rate": {
            "cardiff_vs_vader": pairs["cardiff_vs_vader"] / n,
            "cardiff_vs_textblob": pairs["cardiff_vs_textblob"] / n,
            "vader_vs_textblob": pairs["vader_vs_textblob"] / n,
        },
        "technical_long_text_cardiff_vader_disagreements_n": len(tech_disagree),
        "technical_disagreement_examples_capped": tech_disagree[:25],
        "note": "Instrument agreement ≠ ground-truth sentiment",
    }


def run_cardiff() -> dict[str, Any]:
    dest = OUT_ROOT / "phase3_cardiff"
    dest.mkdir(parents=True, exist_ok=True)
    resource = _resource_check()
    write_json(dest / "RESOURCE_CHECK.json", resource)
    if not resource["disk_ok"]:
        write_json(dest / "CARDIFF_STATUS.json", {"status": "BLOCKED_DISK", "resource": resource})
        return {"status": "BLOCKED_DISK", "resource": resource}

    print("  installing Cardiff venv...", flush=True)
    install = _ensure_venv(resource)
    write_json(dest / "INSTALL_LOG.json", install)
    if install["status"] != "INSTALLED":
        write_json(dest / "CARDIFF_STATUS.json", {"status": install["status"], "install": install})
        return {"status": install["status"], "install": install}

    print("  Cardiff canary 20...", flush=True)
    canary_sel = _attach_texts(_select_units(CANARY_N, SEED + 1))
    canary_path = RESTRICTED_ROOT / "cardiff_canary_units.jsonl"
    # Strip full text from public hashes later; keep text only in restricted for inference
    write_jsonl(canary_path, canary_sel)
    canary_out = RESTRICTED_ROOT / "cardiff_canary_scores.jsonl"
    canary_run = _run_inference_script(canary_path, canary_out)
    write_json(dest / "CANARY_RUN.json", canary_run)
    if canary_run["returncode"] != 0 or not canary_out.exists():
        write_json(dest / "CARDIFF_STATUS.json", {"status": "CANARY_FAILED", "run": canary_run})
        return {"status": "CANARY_FAILED", "run": canary_run}

    canary_rows = load_jsonl(canary_out)
    canary_meta = json.loads(canary_out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    canary_report = {
        "utc": utcnow(),
        "n": len(canary_rows),
        "n_expected": CANARY_N,
        "all_produced_labels": all(r.get("cardiff_label") for r in canary_rows),
        "n_truncated": sum(1 for r in canary_rows if r.get("truncated")),
        "token_length": {
            "mean": float(np.mean([r["n_tokens"] for r in canary_rows])),
            "max": int(max(r["n_tokens"] for r in canary_rows)),
        },
        "latency_s": {
            "mean": float(np.mean([r["latency_s"] for r in canary_rows])),
            "sum": float(np.sum([r["latency_s"] for r in canary_rows])),
        },
        "meta": canary_meta,
        "comparison": _compare_instruments(canary_rows),
        "sha256_scores": sha256_file(canary_out),
    }
    write_json(dest / "CANARY_RESULTS.json", canary_report)
    if not canary_report["all_produced_labels"]:
        write_json(dest / "CARDIFF_STATUS.json", {"status": "CANARY_INCOMPLETE"})
        return {"status": "CANARY_INCOMPLETE", "canary": canary_report}

    print("  Cardiff pilot 300...", flush=True)
    pilot_sel = _attach_texts(_select_units(PILOT_N, SEED + 2))
    pilot_path = RESTRICTED_ROOT / "cardiff_pilot_units.jsonl"
    write_jsonl(pilot_path, pilot_sel)
    pilot_out = RESTRICTED_ROOT / "cardiff_pilot_scores.jsonl"
    pilot_run = _run_inference_script(pilot_path, pilot_out)
    write_json(dest / "PILOT_RUN.json", pilot_run)
    if pilot_run["returncode"] != 0 or not pilot_out.exists():
        write_json(dest / "CARDIFF_STATUS.json", {"status": "PILOT_FAILED", "run": pilot_run, "canary": "ok"})
        return {"status": "PILOT_FAILED", "run": pilot_run, "canary": canary_report}

    pilot_rows = load_jsonl(pilot_out)
    pilot_meta = json.loads(pilot_out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    pilot_report = {
        "utc": utcnow(),
        "n": len(pilot_rows),
        "n_expected": PILOT_N,
        "separate_from_confirmatory_main_results": True,
        "n_truncated": sum(1 for r in pilot_rows if r.get("truncated")),
        "token_length": {
            "mean": float(np.mean([r["n_tokens"] for r in pilot_rows])),
            "p50": float(np.median([r["n_tokens"] for r in pilot_rows])),
            "p90": float(np.percentile([r["n_tokens"] for r in pilot_rows], 90)),
            "max": int(max(r["n_tokens"] for r in pilot_rows)),
        },
        "latency_s": {
            "mean": float(np.mean([r["latency_s"] for r in pilot_rows])),
            "sum": float(np.sum([r["latency_s"] for r in pilot_rows])),
        },
        "meta": pilot_meta,
        "comparison": _compare_instruments(pilot_rows),
        "by_unit": {
            u: sum(1 for r in pilot_rows if r.get("unit") == u)
            for u in sorted({r.get("unit") for r in pilot_rows})
        },
        "sha256_scores": sha256_file(pilot_out),
        "sha256_units": sha256_file(pilot_path),
        "note": "Pilot only; not confirmatory main results; agreement ≠ truth",
    }
    write_json(dest / "PILOT_RESULTS.json", pilot_report)
    status = {
        "status": "CANARY_AND_PILOT_COMPLETE",
        "canary_n": canary_report["n"],
        "pilot_n": pilot_report["n"],
        "license": resource["license"],
        "pinned_revision": CARDIFF_REV,
    }
    write_json(dest / "CARDIFF_STATUS.json", status)
    return {"status": status["status"], "canary": canary_report, "pilot": pilot_report, "resource": resource}
