/**
 * Live multi-subreddit VADER: fetch ReleaseTrain filter JSON, pick top posts per subreddit,
 * run VADER per comment for comment-index trajectories.
 */

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
  "reddit/query/filter?minComments=1&limit=500",
  "reddit/query/filter?minComments=3&minScore=0.5&limit=500",
  "reddit/query/filter?minComments=3&maxScore=0.5&limit=500",
];

/** ReleaseTrain exposes CORS `Access-Control-Allow-Origin: *`, so we call the API directly. Same-origin Netlify proxies were unreliable (404) when redirects did not match the deploy layout. */
const RELEASETRAIN_API_BASE = "https://releasetrain.io/api/";

function apiBaseUrl() {
  return RELEASETRAIN_API_BASE;
}

function cleanText(s) {
  if (!s) return "";
  return String(s)
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isOpComment(c, post) {
  const v = c.is_submitter;
  if (v === true) return true;
  if (v === false) return false;
  const pa = (post.author || "").trim().toLowerCase();
  const ca = (c.author || "").trim().toLowerCase();
  return Boolean(pa) && pa === ca;
}

function commentSortKey(c) {
  if (c.created_utc_ts != null) {
    const t = Number(c.created_utc_ts);
    if (!Number.isNaN(t)) return t;
  }
  const raw = c.created_utc;
  if (!raw) return 0;
  const s = String(raw).replace("Z", "+00:00");
  const d = Date.parse(s);
  return Number.isNaN(d) ? 0 : d / 1000;
}

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
    const top = list.slice(0, k);
    for (const p of top) {
      out.push({ post: p });
    }
  }
  return out;
}

/** Distinct Plotly line colors (up to 5 posts). */
const TRAJECTORY_LINE_COLORS = [
  "#f59e0b",
  "#3b82f6",
  "#10b981",
  "#ec4899",
  "#a855f7",
];

function analyzePost(SentimentClass, post) {
  const title = cleanText(post.title || "");
  const comments = Array.isArray(post.comments) ? [...post.comments] : [];
  comments.sort((a, b) => commentSortKey(a) - commentSortKey(b));

  const trajX = [];
  const trajY = [];
  const trajCustom = [];

  let commentIndex = 0;
  for (const c of comments) {
    const body = cleanText(c.body || "");
    if (!body) continue;
    const pol = SentimentClass.polarity_scores(body).compound;
    commentIndex += 1;
    trajX.push(commentIndex);
    trajY.push(pol);
    trajCustom.push([
      post.url || "",
      post.redditId || "",
      (post.title || "").slice(0, 100),
      post.subreddit || "",
      commentIndex,
      isOpComment(c, post) ? "author" : "community",
    ]);
  }

  return {
    redditId: post.redditId || "",
    subreddit: post.subreddit || "",
    titleText: title.slice(0, 120),
    url: post.url || "",
    num_comments: post.num_comments,
    score: post.score,
    trajectory: {
      x: trajX,
      y: trajY,
      customdata: trajCustom,
      nComments: trajX.length,
    },
  };
}

async function fetchJson(path) {
  const url = `${apiBaseUrl()}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`HTTP ${res.status} for ${path}: ${t.slice(0, 200)}`);
  }
  return res.json();
}

async function loadAllPosts() {
  const postsById = new Map();
  const errors = [];
  await Promise.all(
    FETCH_SLICES.map(async (path) => {
      try {
        const payload = await fetchJson(path);
        dedupeMerge(postsById, payload.data || []);
      } catch (e) {
        errors.push(String(e.message || e));
      }
    })
  );
  if (!postsById.size && errors.length) {
    throw new Error(errors.join(" | "));
  }
  return postsById;
}

function filterAnalyzed(analyzed, subredditFilter) {
  if (!subredditFilter || subredditFilter === "__all__") return analyzed;
  const want = subredditFilter.toLowerCase();
  return analyzed.filter((a) => (a.subreddit || "").toLowerCase() === want);
}

const TRAJECTORY_SHAPES = [
  {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: 0,
    y1: 0,
    yref: "y",
    line: { color: "#666", width: 1, dash: "dash" },
  },
  {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: 0.05,
    y1: 0.05,
    yref: "y",
    line: { color: "#2f855a", width: 1, dash: "dot" },
  },
  {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: -0.05,
    y1: -0.05,
    yref: "y",
    line: { color: "#c53030", width: 1, dash: "dot" },
  },
];

/** One Plotly figure: single post, comment index vs VADER compound. */
function buildSinglePostTrajectoryPlot(a, postIndex) {
  const color = TRAJECTORY_LINE_COLORS[postIndex % TRAJECTORY_LINE_COLORS.length];
  const tr = a.trajectory;

  if (!tr || !tr.x || !tr.x.length) {
    return {
      traces: [
        {
          type: "scatter",
          x: [0],
          y: [0],
          mode: "text",
          text: ["No comment bodies"],
          textfont: { color: "#f87171", size: 12 },
          showlegend: false,
        },
      ],
      layout: {
        paper_bgcolor: "#0a0a0a",
        plot_bgcolor: "#101010",
        font: { color: "#fde68a", size: 10 },
        margin: { t: 40, r: 12, b: 40, l: 44 },
        title: {
          text: `Post ${postIndex + 1} · r/${a.subreddit} · ${a.redditId}`,
          font: { size: 11, color: "#fef08a" },
        },
        xaxis: { visible: false },
        yaxis: { visible: false },
        annotations: [
          {
            text: "No comments in API payload for this thread.",
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
            showarrow: false,
            font: { color: "#94a3b8", size: 11 },
          },
        ],
      },
    };
  }

  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      x: tr.x,
      y: tr.y,
      customdata: tr.customdata,
      line: { color, width: 2.2 },
      marker: { size: 4, color },
      showlegend: false,
      hovertemplate:
        "comment #%{customdata[4]} · %{customdata[5]}<br>" +
        "%{customdata[2]}<br>" +
        "<b>VADER</b>: %{y:.3f}<br>" +
        "<extra>Click to open post</extra>",
    },
  ];

  const layout = {
    paper_bgcolor: "#0a0a0a",
    plot_bgcolor: "#101010",
    font: { color: "#fde68a", size: 10 },
    margin: { t: 44, r: 12, b: 44, l: 48 },
    title: {
      text: `Post ${postIndex + 1} · r/${a.subreddit} · ${a.redditId}`,
      font: { size: 11, color: "#fef08a" },
    },
    xaxis: {
      title: "Comment # (chronological)",
      gridcolor: "#333",
      zeroline: false,
      dtick: Math.max(1, Math.ceil(tr.x.length / 8)),
    },
    yaxis: {
      title: "VADER compound",
      gridcolor: "#333",
      zeroline: true,
      zerolinecolor: "#666",
      range: [-1, 1],
    },
    shapes: TRAJECTORY_SHAPES,
  };

  return { traces, layout };
}

function purgeTrajectoryGrid(container) {
  if (!container) return;
  container.querySelectorAll(".js-trajectory-plot").forEach((gd) => {
    try {
      Plotly.purge(gd);
    } catch (_) {
      /* ignore */
    }
  });
  container.innerHTML = "";
}

function attachTrajectoryClick(gd) {
  gd.on("plotly_click", (ev) => {
    const p = ev.points && ev.points[0];
    if (!p || p.customdata == null) return;
    const row = p.customdata;
    const url = Array.isArray(row) ? row[0] : null;
    if (url && String(url).startsWith("http")) {
      window.open(url, "_blank", "noopener");
    }
  });
}

function renderTrajectoryGrid(gridEl, subset, plotConfig) {
  purgeTrajectoryGrid(gridEl);
  const rows = subset.slice(0, 5);
  rows.forEach((a, i) => {
    const cell = document.createElement("div");
    cell.className = "trajectory-cell";

    const head = document.createElement("div");
    head.className = "trajectory-cell-head";
    const safeUrl = a.url && String(a.url).startsWith("http") ? a.url : "#";
    const strong = document.createElement("strong");
    strong.textContent = `Post ${i + 1}`;
    head.appendChild(strong);
    head.appendChild(document.createTextNode(" · "));
    const link = document.createElement("a");
    link.href = safeUrl;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `r/${a.subreddit} · ${a.redditId}`;
    head.appendChild(link);
    head.appendChild(document.createElement("br"));
    const sub = document.createElement("span");
    sub.style.opacity = "0.9";
    sub.textContent = `${(a.titleText || "").slice(0, 72)}…`;
    head.appendChild(sub);

    const plotDiv = document.createElement("div");
    plotDiv.className = "js-trajectory-plot";

    cell.appendChild(head);
    cell.appendChild(plotDiv);
    gridEl.appendChild(cell);

    const { traces, layout } = buildSinglePostTrajectoryPlot(a, i);
    Plotly.newPlot(plotDiv, traces, layout, plotConfig).then((gd) => {
      attachTrajectoryClick(gd);
    });
  });
}

let siaPromise = null;

/** Bundled with esbuild from vader-sentiment@1.1.3 (CDN esm.sh wrapped the wrong export). */
function getAnalyzer() {
  if (!siaPromise) {
    siaPromise = import("../vendor/vader-sentiment.bundle.mjs").then((mod) => {
      const C = mod.default;
      if (!C || typeof C.polarity_scores !== "function") {
        throw new Error("vader bundle: SentimentIntensityAnalyzer.polarity_scores missing");
      }
      return C;
    });
  }
  return siaPromise;
}

export async function runLiveSubredditDashboard(opts) {
  const {
    trajectoryGridEl,
    statusEl,
    subSelect,
    onMeta,
  } = opts;

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
    const subset = filterAnalyzed(analyzed, v);
    if (!subset.length) {
      statusEl.textContent = "No posts for this filter (API slice may lack that subreddit). Try Refresh.";
      if (trajectoryGridEl) purgeTrajectoryGrid(trajectoryGridEl);
      return;
    }

    if (trajectoryGridEl) {
      renderTrajectoryGrid(trajectoryGridEl, subset, plotConfig);
    }

    statusEl.textContent = `Analyzed ${subset.length} post(s) · unique posts merged from API: ${meta.fetchedUniquePosts} · Up to 5 separate trajectory charts · Click a point or the header link to open Reddit · Refresh for a new sample.`;
  }

  subSelect.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "__all__";
  allOpt.textContent = "All subreddits";
  subSelect.appendChild(allOpt);
  for (const t of TARGET_SUBREDDITS) {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = `r/${t}`;
    subSelect.appendChild(o);
  }

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
