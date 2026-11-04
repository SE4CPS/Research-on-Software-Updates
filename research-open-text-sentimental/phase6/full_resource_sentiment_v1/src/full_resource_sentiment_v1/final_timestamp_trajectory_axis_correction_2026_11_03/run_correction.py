"""Revise Figures B/E to categorical elapsed-time groups; preserve sealed numbers."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import RUN_ID, SEALED_TRAJECTORY

ROOT = Path(__file__).resolve().parents[3]
SEALED = ROOT / "results" / SEALED_TRAJECTORY
OUT = ROOT / "results" / "final_timestamp_trajectory_axis_correction_2026-11-03"

# Display order for categorical axis (not measured hours)
EVENT0_CAT = "Event 0\n(full cohort)"
BIN_ORDER = ["0–6 h", "6–24 h", "1–3 d", ">3 d"]
BIN_KEY_TO_LABEL = {
    "0-6h": "0–6 h",
    "6-24h": "6–24 h",
    "1-3d": "1–3 d",
    ">3d": ">3 d",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: as_str(r.get(k)) for k in fieldnames})


def as_str(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    return v


def _f(v):
    if v is None or v == "":
        return None
    return float(v)


def _i(v):
    if v is None or v == "":
        return None
    return int(float(v))


def _bool(v) -> bool:
    return str(v).lower() in {"true", "1", "yes"}


def transform_elapsed_rows(rows: list[dict]) -> list[dict]:
    """Preserve sealed means/n/CIs; replace midpoint metadata with category index."""
    out = []
    for r in rows:
        bin_key = r["bin"]
        if bin_key == "Event0":
            cat = EVENT0_CAT.replace("\n", " ")
            cat_index = 0
            x_group = "Event0"
        else:
            cat = BIN_KEY_TO_LABEL[bin_key]
            cat_index = 1 + list(BIN_KEY_TO_LABEL).index(bin_key)
            x_group = bin_key
        out.append(
            {
                "cohort": r["cohort"],
                "label": r["label"],
                "role": r["role"],
                "bin": bin_key,
                "x_group": x_group,
                "x_category_label": cat,
                "x_category_index": cat_index,
                # Retained for provenance only; NOT a measured midpoint for open-ended >3d
                "legacy_bin_mid_hours_do_not_plot_as_measured": r.get("bin_mid_hours"),
                "n_threads": _i(r["n_threads"]),
                "n_comments": _i(r["n_comments"]),
                "mean": _f(r["mean"]),
                "median": _f(r.get("median")),
                "ci95_low": _f(r["ci95_low"]),
                "ci95_high": _f(r["ci95_high"]),
                "sparse": _bool(r["sparse"]),
                "scale": r["scale"],
            }
        )
    return out


def transform_gap_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        bin_key = r["bin"]
        cat = BIN_KEY_TO_LABEL[bin_key]
        cat_index = list(BIN_KEY_TO_LABEL).index(bin_key)
        out.append(
            {
                "cohort": r["cohort"],
                "label": r["label"],
                "bin": bin_key,
                "x_group": bin_key,
                "x_category_label": cat,
                "x_category_index": cat_index,
                "legacy_bin_mid_hours_do_not_plot_as_measured": r.get("bin_mid_hours"),
                "n_threads_both_roles": _i(r["n_threads_both_roles"]),
                "mean_author_minus_community": _f(r["mean_author_minus_community"]),
                "ci95_low": _f(r["ci95_low"]),
                "ci95_high": _f(r["ci95_high"]),
                "sparse": _bool(r["sparse"]),
                "exploratory": _bool(r.get("exploratory", True)),
            }
        )
    return out


def verify_numeric_preservation(orig: list[dict], new: list[dict], *, kind: str) -> list[str]:
    fails = []
    if len(orig) != len(new):
        return [f"{kind}: row count {len(orig)} vs {len(new)}"]
    def _close(a, b) -> bool:
        if a in (None, "") and b is None:
            return True
        if a in (None, "") or b is None:
            return False
        return abs(float(a) - float(b)) <= 1e-12

    for o, n in zip(orig, new):
        key = (o["cohort"], o["label"], o.get("role", ""), o["bin"])
        if kind == "elapsed":
            if _i(o["n_threads"]) != n["n_threads"]:
                fails.append(f"{key}: n_threads mismatch")
            if not _close(o["mean"], n["mean"]):
                fails.append(f"{key}: mean mismatch {o['mean']} vs {n['mean']}")
            if not _close(o["ci95_low"], n["ci95_low"]):
                fails.append(f"{key}: ci low mismatch")
            if not _close(o["ci95_high"], n["ci95_high"]):
                fails.append(f"{key}: ci high mismatch")
            if _bool(o["sparse"]) != n["sparse"]:
                fails.append(f"{key}: sparse mismatch")
            if o["label"] != n["label"] or o["bin"] != n["bin"] or o["role"] != n["role"]:
                fails.append(f"{key}: label/bin/role mismatch")
        else:
            if _i(o["n_threads_both_roles"]) != n["n_threads_both_roles"]:
                fails.append(f"{key}: n mismatch")
            if not _close(o["mean_author_minus_community"], n["mean_author_minus_community"]):
                fails.append(f"{key}: gap mean mismatch")
            if not _close(o["ci95_low"], n["ci95_low"]):
                fails.append(f"{key}: ci low mismatch")
            if not _close(o["ci95_high"], n["ci95_high"]):
                fails.append(f"{key}: ci high mismatch")
            if _bool(o["sparse"]) != n["sparse"]:
                fails.append(f"{key}: sparse mismatch")
    return fails


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig_b(path: Path, rows: list[dict]) -> None:
    _style()
    cohorts = ["ai", "frozen"]
    labels = ["TPS", "GDS"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharey=True)
    role_style = {
        "post": {"color": "#333333", "marker": "D", "ls": "None", "label": "Event 0 (full cohort posts)"},
        "author": {"color": "#1f4e79", "marker": "o", "ls": "-", "label": "Author replies (bin-eligible threads)"},
        "community": {"color": "#2a7a4b", "marker": "s", "ls": "--", "label": "Community replies (bin-eligible threads)"},
    }
    # categorical positions: 0=Event0, 1..4 = bins
    xtick_labels = [EVENT0_CAT] + BIN_ORDER
    xpos = list(range(len(xtick_labels)))

    for i, cohort in enumerate(cohorts):
        for j, lab in enumerate(labels):
            ax = axes[i][j]
            for role, style in role_style.items():
                sub = [
                    r
                    for r in rows
                    if r["cohort"] == cohort and r["label"] == lab and r["role"] == role
                ]
                sub = sorted(sub, key=lambda r: r["x_category_index"])
                xs, means, yerr_lo, yerr_hi, sparse_flags = [], [], [], [], []
                for r in sub:
                    if r["mean"] is None:
                        continue
                    xs.append(r["x_category_index"])
                    means.append(r["mean"])
                    lo = r["ci95_low"]
                    hi = r["ci95_high"]
                    yerr_lo.append(r["mean"] - lo if lo is not None else 0.0)
                    yerr_hi.append(hi - r["mean"] if hi is not None else 0.0)
                    sparse_flags.append(r["sparse"])
                for x, m, elo, ehi, sparse in zip(xs, means, yerr_lo, yerr_hi, sparse_flags):
                    ax.errorbar(
                        [x],
                        [m],
                        yerr=np.array([[elo], [ehi]]),
                        fmt=style["marker"],
                        color=style["color"],
                        markerfacecolor=("none" if sparse else style["color"]),
                        capsize=2,
                        markersize=6,
                    )
                    if sparse:
                        ax.annotate(
                            f"n={next(r['n_threads'] for r in sub if r['x_category_index']==x)}*",
                            (x, m),
                            textcoords="offset points",
                            xytext=(0, -12),
                            ha="center",
                            fontsize=7,
                            color="#8a3b12",
                        )
                # connect non-sparse reply bins only (categories 1–4), not claiming same-thread path
                if role != "post":
                    solid = [
                        (r["x_category_index"], r["mean"])
                        for r in sub
                        if r["mean"] is not None and not r["sparse"]
                    ]
                    if len(solid) >= 2:
                        ax.plot(
                            [p[0] for p in solid],
                            [p[1] for p in solid],
                            linestyle=style["ls"],
                            color=style["color"],
                            linewidth=1.2,
                            alpha=0.85,
                        )
            e0 = next(
                (
                    r
                    for r in rows
                    if r["cohort"] == cohort
                    and r["label"] == lab
                    and r["role"] == "post"
                    and r["bin"] == "Event0"
                ),
                None,
            )
            n_strip = f"Event 0 n={e0['n_threads']} (full cohort)" if e0 else ""
            ax.set_title(f"{'AI' if cohort == 'ai' else 'Frozen'} · {lab}\n{n_strip}", fontsize=10)
            ax.axhline(0, color="#888", linewidth=0.6, linestyle=":")
            ax.set_xticks(xpos)
            ax.set_xticklabels(xtick_labels, fontsize=8)
            ax.set_xlim(-0.4, len(xtick_labels) - 0.6)
            ax.set_xlabel("Elapsed-time group (categorical; not clock midpoints)")
            if j == 0:
                ax.set_ylabel("Mean VADER compound\n(thread-equal weight within group)")
            if i == 0 and j == 1:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=s["color"],
                        marker=s["marker"],
                        linestyle=s["ls"] if s["ls"] != "None" else "None",
                        label=s["label"],
                    )
                    for s in role_style.values()
                ]
                ax.legend(handles=handles, frameon=False, loc="best", fontsize=7)

    fig.suptitle(
        "Elapsed-time author vs community sentiment (VADER) — categorical time groups\n"
        "Descriptive bin summaries (threads may differ by group); hollow / n* = sparse (n_threads < 20); "
        "no imputation; >3 d is open-ended (not a 168 h midpoint)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_e(path: Path, rows: list[dict]) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"TPS": "#1f4e79", "GDS": "#b8572a"}
    xpos = list(range(len(BIN_ORDER)))
    for ax, cohort in zip(axes, ("ai", "frozen")):
        for lab in ("TPS", "GDS"):
            sub = [r for r in rows if r["cohort"] == cohort and r["label"] == lab]
            sub = sorted(sub, key=lambda r: r["x_category_index"])
            plotted = False
            for r in sub:
                m = r["mean_author_minus_community"]
                if m is None:
                    continue
                x = r["x_category_index"]
                elo = m - r["ci95_low"] if r["ci95_low"] is not None else 0.0
                ehi = r["ci95_high"] - m if r["ci95_high"] is not None else 0.0
                ax.errorbar(
                    [x],
                    [m],
                    yerr=np.array([[elo], [ehi]]),
                    fmt="o",
                    color=colors[lab],
                    markerfacecolor=("none" if r["sparse"] else colors[lab]),
                    capsize=2,
                    label=(lab if not plotted else None),
                )
                plotted = True
                if r["sparse"] or (r["n_threads_both_roles"] or 0) < 40:
                    ax.annotate(
                        f"n={r['n_threads_both_roles']}",
                        (x, m),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color="#555",
                    )
            solid = [
                (r["x_category_index"], r["mean_author_minus_community"])
                for r in sub
                if r["mean_author_minus_community"] is not None and not r["sparse"]
            ]
            if len(solid) >= 2:
                ax.plot(
                    [p[0] for p in solid],
                    [p[1] for p in solid],
                    color=colors[lab],
                    linewidth=1.2,
                )
        ax.axhline(0, color="#888", linewidth=0.6, linestyle=":")
        ax.set_xticks(xpos)
        ax.set_xticklabels(BIN_ORDER)
        ax.set_xlim(-0.4, len(BIN_ORDER) - 0.6)
        ax.set_title("AI-assisted" if cohort == "ai" else "Frozen-score")
        ax.set_xlabel("Elapsed-time group (categorical; >3 d open-ended)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Author − community VADER mean\n(same thread & bin only)")
    fig.suptitle(
        "Exploratory within-bin author−community gap (VADER) — categorical time groups\n"
        "Descriptive group summaries (threads may differ by bin); not a within-thread path; "
        "hollow = sparse; not causal; >3 d is not a 168 h midpoint",
        fontsize=11,
        y=1.05,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_captions(path: Path) -> None:
    path.write_text(
        """# Corrected figure captions (axis correction 2026-11-03)

Status: corrected after Codex CONDITIONAL PASS on sealed trajectory package. Numeric means, n, and CIs are unchanged from the sealed tables.

## Figure B (corrected) — Elapsed-time author and community means (VADER)

Thread-equal-weighted mean VADER compound by role within **predeclared categorical elapsed-time groups** (0–6 h, 6–24 h, 1–3 d, >3 d) measured from the Reddit post `created_utc`. The x-axis is **categorical**; the open-ended **>3 d** group is **not** a measured midpoint of 168 hours (or any other clock hour).

**Composition note:** these panels are **descriptive bin summaries, not within-thread paths**. Threads contributing to each time group can differ. The **Event 0** marker uses the **full purpose cohort** with a valid post score/timestamp, whereas author and community points use only threads with at least one timed scored reply in that group. Figure C remains the appropriate same-thread author path.

Sparse cells (n_threads < 20) are shown hollow / annotated and must not be over-interpreted. Missing role×group observations remain missing (not imputed as neutral). Separate AI/frozen and TPS/GDS facets. Uncertainty: 95% whole-thread bootstrap intervals from the sealed analysis.

## Figure E (corrected) — Exploratory within-bin author−community gap

Within-thread author mean minus community mean in the same categorical elapsed-time group, requiring both roles observed in that group. X-axis categories match Figure B; **>3 d is open-ended** and is not plotted as a 168-hour midpoint. Threads can differ across groups, so connecting points summarizes group means only. Exploratory only; not a causal resolution claim. Sparse cells remain marked.

## Unchanged figures (sealed package)

Figure C (fixed-eligibility same-thread author path) and Figure F (four-model two-event) are unchanged and remain in the sealed trajectory directory.
""",
        encoding="utf-8",
    )


def write_change_report(
    path: Path,
    *,
    elapsed_fails: list[str],
    gap_fails: list[str],
    sealed_hashes: dict,
) -> None:
    path.write_text(
        f"""# Change report — trajectory axis/caption correction ({RUN_ID})

## Trigger
Codex independent audit (`planning/independent_final_timestamp_trajectory_audit_2026-11-03/CODEX_INDEPENDENT_AUDIT.md`) gave a **CONDITIONAL PASS**: sealed tables verified; Figures B and E required axis/caption correction before manuscript use.

## What changed
1. **Figures B and E** replotted with a **categorical** x-axis: `Event 0 (full cohort)`, `0–6 h`, `6–24 h`, `1–3 d`, `>3 d`.
2. Removed presentation of **168 hours** (and other values) as a **measured midpoint** for the open-ended `>3 d` group.
3. Plot-data metadata: retained sealed means/n/CIs/sparse flags; replaced plotting field with `x_category_index` / `x_category_label`; renamed legacy midpoint column to `legacy_bin_mid_hours_do_not_plot_as_measured`.
4. Captions state that bin points are **descriptive summaries** (threads may differ by group) and that Figure B’s Event 0 uses the **full cohort**.

## What did not change
- No sentiment models rerun.
- No labels, sealed trajectory results, manuscripts, or source score files modified.
- Verified means, sample sizes, intervals, TPS/GDS labels, sparse flags preserved (see preservation checks below).
- Figures C and F left in the sealed package without alteration.

## Preservation checks
- Elapsed table mismatches: {elapsed_fails or "none"}
- Gap table mismatches: {gap_fails or "none"}

## Sealed source files reused (SHA-256)
```json
{json.dumps(sealed_hashes, indent=2)}
```

## Output directory
`{OUT}`

UTC: {utc_now()}
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "plot_data").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    sealed_elapsed_path = SEALED / "plot_data" / "elapsed_time_role_means.csv"
    sealed_gap_path = SEALED / "plot_data" / "exploratory_author_community_gap.csv"
    sealed_hashes = {
        "elapsed_time_role_means.csv": sha256_file(sealed_elapsed_path),
        "exploratory_author_community_gap.csv": sha256_file(sealed_gap_path),
        "sealed_OUTPUT_HASHES.json": sha256_file(SEALED / "OUTPUT_HASHES.json"),
        "audit_md": sha256_file(
            ROOT
            / "planning"
            / "independent_final_timestamp_trajectory_audit_2026-11-03"
            / "CODEX_INDEPENDENT_AUDIT.md"
        ),
    }

    orig_elapsed = read_csv(sealed_elapsed_path)
    orig_gap = read_csv(sealed_gap_path)
    new_elapsed = transform_elapsed_rows(orig_elapsed)
    new_gap = transform_gap_rows(orig_gap)

    elapsed_fails = verify_numeric_preservation(orig_elapsed, new_elapsed, kind="elapsed")
    gap_fails = verify_numeric_preservation(orig_gap, new_gap, kind="gap")
    if elapsed_fails or gap_fails:
        raise SystemExit(f"Preservation failed: {elapsed_fails + gap_fails}")

    elapsed_fields = [
        "cohort",
        "label",
        "role",
        "bin",
        "x_group",
        "x_category_label",
        "x_category_index",
        "legacy_bin_mid_hours_do_not_plot_as_measured",
        "n_threads",
        "n_comments",
        "mean",
        "median",
        "ci95_low",
        "ci95_high",
        "sparse",
        "scale",
    ]
    gap_fields = [
        "cohort",
        "label",
        "bin",
        "x_group",
        "x_category_label",
        "x_category_index",
        "legacy_bin_mid_hours_do_not_plot_as_measured",
        "n_threads_both_roles",
        "mean_author_minus_community",
        "ci95_low",
        "ci95_high",
        "sparse",
        "exploratory",
    ]
    write_csv(OUT / "plot_data" / "elapsed_time_role_means_categorical.csv", new_elapsed, elapsed_fields)
    write_csv(OUT / "plot_data" / "exploratory_author_community_gap_categorical.csv", new_gap, gap_fields)

    # JSON mirrors for review
    (OUT / "tables" / "elapsed_time_role_means_categorical.json").write_text(
        json.dumps(new_elapsed, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "tables" / "exploratory_author_community_gap_categorical.json").write_text(
        json.dumps(new_gap, indent=2) + "\n", encoding="utf-8"
    )

    fig_b(OUT / "figures" / "fig_B_elapsed_time_author_community_CORRECTED.png", new_elapsed)
    fig_e(OUT / "figures" / "fig_E_exploratory_author_community_gap_CORRECTED.png", new_gap)

    # Copy sealed originals as before/ references (byte copies for review)
    import shutil

    shutil.copy2(
        SEALED / "figures" / "fig_B_elapsed_time_author_community.png",
        OUT / "figures" / "fig_B_BEFORE_sealed.png",
    )
    shutil.copy2(
        SEALED / "figures" / "fig_E_exploratory_author_community_gap.png",
        OUT / "figures" / "fig_E_BEFORE_sealed.png",
    )

    write_captions(OUT / "FIGURE_CAPTIONS_CORRECTED.md")
    write_change_report(
        OUT / "CHANGE_REPORT.md",
        elapsed_fails=elapsed_fails,
        gap_fails=gap_fails,
        sealed_hashes=sealed_hashes,
    )

    (OUT / "RUN.md").write_text(
        f"""# RUN — {RUN_ID}

- UTC: {utc_now()}
- Entry: `python -m full_resource_sentiment_v1.final_timestamp_trajectory_axis_correction_2026_11_03.run_correction`
- Sealed source (unchanged): `results/{SEALED_TRAJECTORY}/`
- Action: categorical axis + caption correction for Figures B and E only
- Sentiment models: not rerun
- Status: ready for final Codex check
""",
        encoding="utf-8",
    )

    (OUT / "CODEX_FINAL_CHECK_PACKET.md").write_text(
        f"""# Codex final check packet — axis correction

## Verdict requested
Confirm CONDITIONAL PASS item is resolved for Figures B and E.

## Verify
1. Corrected figures use categorical labels `0–6 h`, `6–24 h`, `1–3 d`, `>3 d` (plus Event 0 on B).
2. No axis title or tick implies **168 hours** is a measured midpoint for `>3 d`.
3. Captions state changing-thread composition and full-cohort Event 0 for B.
4. `plot_data/*_categorical.csv` means/n/CIs/sparse match sealed CSVs exactly (preservation checks reported none).
5. Sealed tree `results/{SEALED_TRAJECTORY}/` remains unmodified.
6. File hashes in `OUTPUT_HASHES.json`.

## Key outputs
- `figures/fig_B_elapsed_time_author_community_CORRECTED.png`
- `figures/fig_E_exploratory_author_community_gap_CORRECTED.png`
- `FIGURE_CAPTIONS_CORRECTED.md`
- `CHANGE_REPORT.md`
- Before copies: `figures/fig_B_BEFORE_sealed.png`, `figures/fig_E_BEFORE_sealed.png`
""",
        encoding="utf-8",
    )

    file_hashes = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "OUTPUT_HASHES.json":
            file_hashes[str(p.relative_to(OUT))] = sha256_file(p)
    (OUT / "OUTPUT_HASHES.json").write_text(
        json.dumps(
            {
                "utc": utc_now(),
                "run_id": RUN_ID,
                "n_files": len(file_hashes),
                "files": file_hashes,
                "sealed_sources": sealed_hashes,
                "preservation_ok": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"DONE → {OUT}")
    print("preservation_ok True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
