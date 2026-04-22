/**
 * Live multi-subreddit VADER: fetch ReleaseTrain filter JSON, pick top posts per subreddit,
 * run VADER per comment for comment-index trajectories.
 */

import {
  fetchJson,
  getAnalyzer,
  emptyClassFlags,
  analyzePost,
  renderTrajectoryGrid,
  purgeTrajectoryGrid,
} from "./vader-trajectory.mjs";

const TARGET_SUBREDDITS = [
  "android",
  "rust",
  "mongodb",
  "nginx",
  "linux",
  "openclaw",
  "chrome",
  "firefox",
  "langchain",
];

const FETCH_SLICES = [
  { path: "reddit/query/filter?minComments=1&limit=500", key: "pool" },
  { path: "reddit/query/filter?minComments=3&minScore=0.5&limit=500", key: "class1" },
  { path: "reddit/query/filter?minComments=3&maxScore=0.5&limit=500", key: "class2" },
];

function engagementTuple(post) {
  return [Number(post.num_comments) || 0, Number(post.score) || 0];
}

function dedupeMerge(postsById, rows) {
  for (const p of rows) {
    const id = p.redditId || p._id;
    if (!id) continue;
    const prev = postsById.get(id);
    if (!prev) {
      postsById.set(id, p);
      continue;
    }
    const a = engagementTuple(prev);
    const b = engagementTuple(p);
    if (b[0] > a[0] || (b[0] === a[0] && b[1] > a[1])) {
      postsById.set(id, p);
    }
  }
}

function pickTopKWithClassDiversity(list, k) {
  if (!list.length || k <= 0) return [];
  if (list.length <= k) return list.slice();
  const postId = (p) => p.redditId || p._id;
  const hasC1 = (p) => p._rrClasses && p._rrClasses.class1;
  const hasC2 = (p) => p._rrClasses && p._rrClasses.class2;
  const indexOf = (id) => list.findIndex((p) => postId(p) === id);
  if (k < 2 || !list.some(hasC1) || !list.some(hasC2)) {
    return list.slice(0, k);
  }
  const p1 = list.find(hasC1);
  const p2 = list.find(hasC2);
  const out = [];
  const seen = new Set();
  for (const p of [p1, p2]) {
    if (!p) continue;
    const id = postId(p);
    if (seen.has(id)) continue;
    if (id == null) continue;
    out.push(p);
    seen.add(id);
  }
  for (const p of list) {
    if (out.length >= k) break;
    const id = postId(p);
    if (id == null || seen.has(id)) continue;
    out.push(p);
    seen.add(id);
  }
  out.sort((a, b) => indexOf(postId(a)) - indexOf(postId(b)));
  return out.slice(0, k);
}

function pickTopPerSubreddit(postsById, targets, k) {
  const buckets = new Map();
  for (const t of targets) {
    buckets.set(t.toLowerCase(), []);
  }

  for (const p of postsById.values()) {
    const sub = (p.subreddit || "").trim().toLowerCase();
    if (!buckets.has(sub)) continue;
    buckets.get(sub).push(p);
  }

  const out = [];
  for (const t of targets) {
    const key = t.toLowerCase();
    const list = buckets.get(key) || [];
    list.sort((a, b) => {
      const [ncA, sA] = engagementTuple(a);
      const [ncB, sB] = engagementTuple(b);
      if (ncB !== ncA) return ncB - ncA;
      return sB - sA;
    });
    const top = pickTopKWithClassDiversity(list, k);
    for (const p of top) {
      out.push({ post: p });
    }
  }
  return out;
}

function recordFlagsForRows(rows, sliceKey, flagsById) {
  for (const p of rows) {
    const id = p.redditId || p._id;
    if (!id) continue;
    let f = flagsById.get(id);
    if (!f) {
      f = emptyClassFlags();
      flagsById.set(id, f);
    }
    if (sliceKey === "pool") f.pool = true;
    if (sliceKey === "class1") f.class1 = true;
    if (sliceKey === "class2") f.class2 = true;
  }
}

async function loadAllPosts() {
  const postsById = new Map();
  const flagsById = new Map();
  const errors = [];

  await Promise.all(
    FETCH_SLICES.map(async (slice) => {
      try {
        const payload = await fetchJson(slice.path);
        const rows = payload.data || [];
        recordFlagsForRows(rows, slice.key, flagsById);
        dedupeMerge(postsById, rows);
      } catch (e) {
        errors.push(String(e.message || e));
      }
    })
  );
  for (const p of postsById.values()) {
    const id = p.redditId || p._id;
    p._rrClasses = flagsById.get(id) || emptyClassFlags();
  }
  if (!postsById.size && errors.length) {
    throw new Error(errors.join(" | "));
  }
  return postsById;
}

const MAX_TRAJECTORY_CELLS = 5;

function filterAnalyzedForDisplay(analyzed, subredditFilter) {
  const want = (subredditFilter || TARGET_SUBREDDITS[0]).toLowerCase();
  return analyzed.filter((a) => (a.subreddit || "").toLowerCase() === want);
}

export async function runLiveSubredditDashboard(opts) {
  const { trajectoryGridEl, statusEl, subSelect, onMeta } = opts;

  statusEl.textContent = "Fetching Reddit slices from ReleaseTrain…";
  const postsById = await loadAllPosts();
  const picked = pickTopPerSubreddit(postsById, TARGET_SUBREDDITS, 5);

  const SentimentClass = await getAnalyzer();
  const analyzed = picked.map(({ post }) => analyzePost(SentimentClass, post));

  const meta = {
    fetchedUniquePosts: postsById.size,
    targets: TARGET_SUBREDDITS,
    perSubredditCounts: {},
  };
  for (const t of TARGET_SUBREDDITS) {
    const k = t.toLowerCase();
    meta.perSubredditCounts[t] = analyzed.filter(
      (a) => (a.subreddit || "").toLowerCase() === k
    ).length;
  }
  if (onMeta) onMeta(meta);

  const plotConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  function render() {
    const v = subSelect.value;
    const subset = filterAnalyzedForDisplay(analyzed, v);
    if (!subset.length) {
      statusEl.textContent = "No posts for this filter (API slice may lack that subreddit). Try Refresh.";
      if (trajectoryGridEl) purgeTrajectoryGrid(trajectoryGridEl);
      return;
    }
    if (trajectoryGridEl) {
      renderTrajectoryGrid(trajectoryGridEl, subset, plotConfig, MAX_TRAJECTORY_CELLS);
    }
    statusEl.textContent = `Showing ${subset.length} trajectory chart(s) for r/${(v || TARGET_SUBREDDITS[0]).toLowerCase()} · ${analyzed.length} post(s) scored total · unique posts in API pool: ${meta.fetchedUniquePosts} · Class 1/2/Pool — see green strip on each chart · Per-subreddit pick uses class diversity when available · Click a point or header to open Reddit · Refresh for a new sample.`;
  }

  subSelect.innerHTML = "";
  for (const t of TARGET_SUBREDDITS) {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = `r/${t}`;
    subSelect.appendChild(o);
  }
  subSelect.value = TARGET_SUBREDDITS[0];

  subSelect.onchange = () => render();
  render();
}

function bootLivePanel() {
  const trajectoryGridEl = document.getElementById("liveSubredditTrajectoryGrid");
  const statusEl = document.getElementById("liveSubredditStatus");
  const subSelect = document.getElementById("liveSubredditFilter");
  const metaEl = document.getElementById("liveSubredditMeta");
  const refreshBtn = document.getElementById("liveSubredditRefresh");
  if (!trajectoryGridEl || !statusEl || !subSelect) return;

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => window.location.reload());
  }

  runLiveSubredditDashboard({
    trajectoryGridEl,
    statusEl,
    subSelect,
    onMeta: (meta) => {
      if (!metaEl) return;
      const parts = TARGET_SUBREDDITS.map(
        (t) => `${t}: ${meta.perSubredditCounts[t] ?? 0}/5`
      );
      metaEl.textContent = `Posts per target (max 5 each): ${parts.join(" · ")}`;
    },
  }).catch((err) => {
    statusEl.textContent = `Live panel error: ${err.message || err}`;
    console.error(err);
  });
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootLivePanel);
  } else {
    bootLivePanel();
  }
}
