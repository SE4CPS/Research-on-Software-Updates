"""Run authors' pySenti4SD on unit batches (small chunks to fit disk)."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03 import LABEL_TO_SIGN
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import (
    JAVA_BIN,
    PYSENTI,
    RESTRICTED_ROOT,
    VENV_SENTI,
)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    java_dir = str(JAVA_BIN.parent)
    env["PATH"] = f"{java_dir}:{VENV_SENTI / 'bin'}:{env.get('PATH', '')}"
    env["JAVA_HOME"] = str(JAVA_BIN.parent.parent)
    return env


def score_units_batch(units: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    """Score one batch; returns rows with polarity label/sign joined by sequential ID."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = work_dir / "input.csv"
    jar_csv = work_dir / "input_jar.csv"
    feat_dir = work_dir / "temp_features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = feat_dir / "extractedFeatures.csv"
    pred_name = "batch_predictions.csv"

    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["ID", "Text"])
        for i, u in enumerate(units, start=1):
            # Collapse newlines; Senti4SD CSV is line-oriented
            text = (u.get("text") or "").replace("\r", " ").replace("\n", " ").strip()
            writer.writerow([str(i), text])

    py = str(VENV_SENTI / "bin/python")
    env = _env()
    proc = subprocess.run(
        [py, str(PYSENTI / "python/csv_processing.py"), "-i", str(input_csv), "-d", "sc", "-c", "text"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"csv_processing failed: {proc.stderr[-2000:]}")
    # csv_processing writes alongside input as input_jar.csv in cwd
    produced = work_dir / "input_jar.csv"
    if not produced.exists():
        # may write next to absolute path stem
        alt = Path(str(input_csv).rsplit(".", 1)[0] + "_jar.csv")
        if alt.exists():
            produced = alt
        else:
            raise RuntimeError(f"jar csv missing after csv_processing; stdout={proc.stdout[-500:]}")

    jar = PYSENTI / "java/Senti4SD-fast.jar"
    dsm = PYSENTI / "java/dsm.bin"
    jproc = subprocess.run(
        [
            str(JAVA_BIN),
            "-jar",
            str(jar),
            "-F",
            "A",
            "-i",
            str(produced),
            "-W",
            str(dsm),
            "-oc",
            str(feat_csv),
            "-vd",
            "600",
        ],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    if jproc.returncode != 0 or not feat_csv.exists():
        raise RuntimeError(f"Senti4SD-fast.jar failed: {jproc.stderr[-2000:]}\n{jproc.stdout[-1000:]}")

    # classification_task writes under CWD/predictions/
    pred_dir = work_dir / "predictions"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    cproc = subprocess.run(
        [
            py,
            str(PYSENTI / "python/classification_task.py"),
            "-i",
            str(feat_csv),
            "-i",
            str(input_csv),
            "-d",
            "sc",
            "-m",
            str(PYSENTI / "Senti4SD.model"),
            "-c",
            "200",
            "-j",
            "1",
            "-o",
            pred_name,
        ],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    pred_path = work_dir / "predictions" / pred_name
    if cproc.returncode != 0 or not pred_path.exists():
        raise RuntimeError(f"classification_task failed: {cproc.stderr[-2000:]}\n{cproc.stdout[-1000:]}")

    preds: dict[int, str] = {}
    with pred_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pid = int(row["ID"])
            preds[pid] = str(row["PREDICTED"]).strip().lower()

    out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i, u in enumerate(units, start=1):
        lab = preds.get(i)
        if lab not in LABEL_TO_SIGN:
            failures.append({"reddit_id": u["reddit_id"], "unit": u["unit"], "predicted": lab, "batch_id": i})
            continue
        out.append(
            {
                "cohort": u["cohort"],
                "reddit_id": u["reddit_id"],
                "label": u["label"],
                "unit": u["unit"],
                "text_hash": u["text_hash"],
                "text_chars": u["text_chars"],
                "source": u.get("source"),
                "date_band": u.get("date_band"),
                "senti4sd_label": lab,
                "senti4sd_sign": LABEL_TO_SIGN[lab],
                "vader": u.get("vader"),
                "textblob": u.get("textblob"),
                "cardiff_sign": u.get("cardiff_sign"),
                "join_key": f"{u['reddit_id']}|{u['text_hash']}|{u['unit']}",
            }
        )
    # reclaim feature disk immediately
    shutil.rmtree(feat_dir, ignore_errors=True)
    for p in (produced, input_csv, jar_csv):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    return out


def score_all(units: list[dict[str, Any]], cohort_tag: str, batch_size: int = 80) -> dict[str, Any]:
    out_path = RESTRICTED_ROOT / f"{cohort_tag}_senti4sd_scores.jsonl"
    fail_path = RESTRICTED_ROOT / f"{cohort_tag}_senti4sd_failures.jsonl"
    if out_path.exists():
        out_path.unlink()
    if fail_path.exists():
        fail_path.unlink()

    scored = 0
    failed = 0
    work_root = RESTRICTED_ROOT / f"_work_{cohort_tag}"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(units), batch_size):
        batch = units[start : start + batch_size]
        print(f"  Senti4SD {cohort_tag} batch {start // batch_size + 1}: {len(batch)} texts...", flush=True)
        work = work_root / f"batch_{start:05d}"
        try:
            rows = score_units_batch(batch, work)
            from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import append_jsonl

            append_jsonl(out_path, rows)
            scored += len(rows)
            miss = len(batch) - len(rows)
            if miss:
                failed += miss
                append_jsonl(
                    fail_path,
                    [
                        {
                            "reddit_id": u["reddit_id"],
                            "unit": u["unit"],
                            "text_hash": u["text_hash"],
                            "reason": "missing_prediction",
                        }
                        for u in batch
                        if not any(r["reddit_id"] == u["reddit_id"] and r["unit"] == u["unit"] for r in rows)
                    ],
                )
        except Exception as exc:  # noqa: BLE001 — record batch failure, continue
            failed += len(batch)
            from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import append_jsonl

            append_jsonl(
                fail_path,
                [
                    {
                        "reddit_id": u["reddit_id"],
                        "unit": u["unit"],
                        "text_hash": u["text_hash"],
                        "reason": f"batch_error: {exc}",
                    }
                    for u in batch
                ],
            )
            print(f"  BATCH FAILED: {exc}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    shutil.rmtree(work_root, ignore_errors=True)
    return {
        "cohort": cohort_tag,
        "n_input_units": len(units),
        "n_scored": scored,
        "n_failed": failed,
        "scores_path": str(out_path),
        "failures_path": str(fail_path),
        "batch_size": batch_size,
    }
