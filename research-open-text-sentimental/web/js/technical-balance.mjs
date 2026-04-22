/**
 * All subreddits from /reddit/meta/subreddits, with balanced minScore vs maxScore
 * cohorts (up to 3+3; single cohort capped at 5; mixed fill to 5 when needed).
 */
import {
  fetchJson,
  getAnalyzer,
  emptyClassFlags,
  analyzePost,
  renderTrajectoryGrid,
  purgeTrajectoryGrid,
} from "./vader-trajectory.mjs";

const CLASS1_PATH = "reddit/query/filter?minComments=3&minScore=0.5&limit=500";
const CLASS2_PATH = "reddit/query/filter?minComments=3&maxScore=0.5&limit=500";
const META_PATH = "reddit/meta/subreddits";

/** Deduplicate by lowercase (e.g. python vs Python); sort for the dropdown. */
function allSubredditNamesFromMeta(dataList) {
  const seen = new Set();
  const out = [];
  for (const raw of dataList || []) {
    const s = String(raw).trim().toLowerCase();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

function postId(p) {
  return p.redditId || p._id;
}

function sortByEngagement(posts) {
  return [...posts].sort((a, b) => {
    const [ncA, sA] = [Number(a.num_comments) || 0, Number(a.score) || 0];
    const [ncB, sB] = [Number(b.num_comments) || 0, Number(b.score) || 0];
    if (ncB !== ncA) return ncB - ncA;
    return sB - sA;
  });
}

function uniqByIdSorted(posts) {
  const m = new Map();
  for (const p of posts) {
    const i = postId(p);
    if (i == null) continue;
    if (!m.has(i)) m.set(i, p);
  }
  return sortByEngagement([...m.values()]);
}

/**
 * Picks: up to 3 from l1, 3 from l2, deduped. If <5 total, add best remaining from
 * union until 5. If 5 and both cohorts can reach 3+3, add one more for 6.
 */
export function pickBalancedMinMax(l1, l2) {
  l1 = sortByEngagement(l1);
  l2 = sortByEngagement(l2);
  if (!l1.length && !l2.length) return [];
  if (!l1.length) {
    return l2.slice(0, 5).map((p) => ({ post: p, c1: false, c2: true }));
  }
  if (!l2.length) {
    return l1.slice(0, 5).map((p) => ({ post: p, c1: true, c2: false }));
  }

  const seen = new Set();
  const out = [];
  const add = (p, c1, c2) => {
    const i = postId(p);
    if (i == null || seen.has(i)) return;
    seen.add(i);
    out.push({ post: p, c1, c2 });
  };

  for (const p of l1) {
    if (out.filter((x) => x.c1).length >= 3) break;
    add(p, true, false);
  }
  for (const p of l2) {
    if (out.filter((x) => x.c2).length >= 3) break;
    add(p, false, true);
  }

  if (out.length < 5) {
    for (const p of uniqByIdSorted([...l1, ...l2])) {
      if (out.length >= 5) break;
      if (seen.has(postId(p))) continue;
      const i = postId(p);
      const in1 = l1.some((q) => postId(q) === i);
      const in2 = l2.some((q) => postId(q) === i);
      if (!in1 && !in2) continue;
      add(p, in1, in2);
    }
  }
  return out.length > 6 ? out.slice(0, 6) : out;
}

function postsForSub(rows, name) {
  const n = name.toLowerCase();
  return rows.filter((p) => (p.subreddit || "").trim().toLowerCase() === n);
}

function buildClassFlags(c1, c2) {
  return { ...emptyClassFlags(), class1: Boolean(c1), class2: Boolean(c2) };
}

export async function runTechnicalBalancePanel(opts) {
  const { gridEl, statusEl, subSelect, metaOut } = opts;
  statusEl.textContent = "Loading meta subreddits, min slice, and max slice…";

  const [metaRes, c1p, c2p] = await Promise.all([
    fetchJson(META_PATH),
    fetchJson(CLASS1_PATH),
    fetchJson(CLASS2_PATH),
  ]);

  const dataList = Array.isArray(metaRes.data) ? metaRes.data : Array.isArray(metaRes) ? metaRes : [];
  const allNames = allSubredditNamesFromMeta(dataList);
  if (!allNames.length) {
    statusEl.textContent = "ReleaseTrain meta returned no subreddit names. Check the API.";
    if (metaOut) metaOut.textContent = "Meta: 0 names.";
    return;
  }

  const rows1 = c1p.data || [];
  const rows2 = c2p.data || [];

  const withAny = allNames.filter(
    (name) => postsForSub(rows1, name).length + postsForSub(rows2, name).length > 0
  );

  subSelect.innerHTML = "";
  for (const name of allNames) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = `r/${name}`;
    subSelect.appendChild(o);
  }
  subSelect.value = (withAny.length ? withAny[0] : allNames[0]) || "";

  if (metaOut) {
    const hint = withAny.length
      ? ` · ${withAny.length} with ≥1 post in min or max slice (first selected)`
      : " · no meta sub had posts in current min/max rows — pick another or Refresh";
    metaOut.textContent = `Meta: ${dataList.length} raw · ${allNames.length} unique (deduped) · min-slice: ${rows1.length} · max-slice: ${rows2.length}${hint}`;
  }
  if (!withAny.length) {
    statusEl.textContent =
      "No subreddit in the current min/max filter rows matches any meta name. Open the list and try again after Refresh, or the ReleaseTrain sample may be empty.";
  }

  const SentimentClass = await getAnalyzer();
  const plotConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  const render2 = () => {
    const sub = (subSelect.value || "").toLowerCase();
    if (!sub) {
      statusEl.textContent = "Select a subreddit.";
      return;
    }
    const l1 = postsForSub(rows1, sub);
    const l2 = postsForSub(rows2, sub);
    const picks = pickBalancedMinMax(l1, l2);
    const analyzed = [];
    for (const row of picks) {
      const f = buildClassFlags(row.c1, row.c2);
      const p = { ...row.post, _rrClasses: f };
      analyzed.push(analyzePost(SentimentClass, p));
    }
    if (!analyzed.length) {
      statusEl.textContent = `No posts for r/${sub} in min (≥0.5) or max (≤0.5) slices. Try another.`;
      purgeTrajectoryGrid(gridEl);
      return;
    }
    const maxCharts = Math.min(6, analyzed.length);
    renderTrajectoryGrid(gridEl, analyzed, plotConfig, maxCharts);
    const n1 = l1.length;
    const n2 = l2.length;
    statusEl.textContent = `r/${sub} · ${analyzed.length} chart(s) (target 3 Class 1 + 3 Class 2, or ≤5 if one class) · rows in min slice: ${n1} · in max slice: ${n2} · VADER in browser · click point = open Reddit.`;
  };

  subSelect.onchange = () => render2();
  render2();
}

function bootTechnical() {
  const gridEl = document.getElementById("technicalBalanceGrid");
  const statusEl = document.getElementById("technicalBalanceStatus");
  const subSelect = document.getElementById("technicalBalanceFilter");
  const metaOut = document.getElementById("technicalBalanceMeta");
  const refresh = document.getElementById("technicalBalanceRefresh");
  if (!gridEl || !statusEl || !subSelect) return;
  if (refresh) {
    refresh.addEventListener("click", () => window.location.reload());
  }
  runTechnicalBalancePanel({ gridEl, statusEl, subSelect, metaOut }).catch((e) => {
    statusEl.textContent = `Error: ${e.message || e}`;
    console.error(e);
  });
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootTechnical);
  } else {
    bootTechnical();
  }
}
