/**
 * Live multi-subreddit VADER dashboard: fetch ReleaseTrain filter JSON via same-origin proxy,
 * pick top posts per subreddit, score title / description / author vs community comments.
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

function compoundLabel(c) {
  if (c >= 0.05) return "Positive";
  if (c <= -0.05) return "Negative";
  return "Neutral";
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

function meanCompounds(scores) {
  if (!scores.length) return null;
  const sum = scores.reduce((a, b) => a + b, 0);
  return sum / scores.length;
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
  const desc = cleanText(
    [post.author_description || "", post.body || ""].filter(Boolean).join("\n")
  );

  const titleScores = title
    ? SentimentClass.polarity_scores(title)
    : { compound: 0, pos: 0, neu: 1, neg: 0 };
  const descScores = desc
    ? SentimentClass.polarity_scores(desc)
    : { compound: 0, pos: 0, neu: 1, neg: 0 };

  const comments = Array.isArray(post.comments) ? [...post.comments] : [];
  comments.sort((a, b) => commentSortKey(a) - commentSortKey(b));

  const authorCompounds = [];
  const communityCompounds = [];
  const trajX = [];
  const trajY = [];
  const trajCustom = [];

  let commentIndex = 0;
  for (const c of comments) {
    const body = cleanText(c.body || "");
    if (!body) continue;
    const pol = SentimentClass.polarity_scores(body).compound;
    if (isOpComment(c, post)) authorCompounds.push(pol);
    else communityCompounds.push(pol);
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
    title: { ...titleScores, label: compoundLabel(titleScores.compound) },
    description: { ...descScores, label: compoundLabel(descScores.compound) },
    authorComments: {
      mean: meanCompounds(authorCompounds),
      n: authorCompounds.length,
      label:
        authorCompounds.length === 0
          ? "N/A"
          : compoundLabel(meanCompounds(authorCompounds)),
    },
    communityComments: {
      mean: meanCompounds(communityCompounds),
      n: communityCompounds.length,
      label:
        communityCompounds.length === 0
          ? "N/A"
          : compoundLabel(meanCompounds(communityCompounds)),
    },
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

function buildGroupedBarPlot(analyzed) {
  const labels = analyzed.map(
    (a) =>
      `${a.subreddit} · ${a.redditId}<br><span style="font-size:11px">${(a.titleText || "").slice(0, 60)}…</span>`
  );

  const titleC = analyzed.map((a) => a.title.compound);
  const descC = analyzed.map((a) => a.description.compound);
  const authC = analyzed.map((a) =>
    a.authorComments.mean == null ? null : a.authorComments.mean
  );
  const commC = analyzed.map((a) =>
    a.communityComments.mean == null ? null : a.communityComments.mean
  );

  const traces = [
    {
      type: "bar",
      name: "Post title",
      x: labels,
      y: titleC,
      marker: { color: "#f59e0b" },
    },
    {
      type: "bar",
      name: "Description / body",
      x: labels,
      y: descC,
      marker: { color: "#84cc16" },
    },
    {
      type: "bar",
      name: "Author comments (mean)",
      x: labels,
      y: authC,
      marker: { color: "#ea580c" },
    },
    {
      type: "bar",
      name: "Community comments (mean)",
      x: labels,
      y: commC,
      marker: { color: "#2563eb" },
    },
  ];

  const layout = {
    barmode: "group",
    paper_bgcolor: "#0a0a0a",
    plot_bgcolor: "#101010",
    font: { color: "#fde68a", size: 11 },
    margin: { t: 48, r: 24, b: 140, l: 56 },
    title: {
      text: "VADER compound by text layer (live fetch)",
      font: { size: 14, color: "#fef08a" },
    },
    xaxis: { tickangle: -35, gridcolor: "#333" },
    yaxis: {
      title: "VADER compound",
      range: [-1, 1],
      gridcolor: "#333",
      zeroline: true,
      zerolinecolor: "#666",
    },
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      x: 0,
      font: { size: 10, color: "#fde68a" },
    },
    shapes: [
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
    ],
  };

  return { traces, layout };
}

function buildCommentIndexLinePlot(subset) {
  const traces = [];
  const maxLines = 5;
  const rows = subset.slice(0, maxLines);

  rows.forEach((a, i) => {
    const tr = a.trajectory;
    if (!tr || !tr.x || !tr.x.length) return;
    const color = TRAJECTORY_LINE_COLORS[i % TRAJECTORY_LINE_COLORS.length];
    const name = `r/${a.subreddit} · ${a.redditId} (${tr.nComments} comments)`;
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name,
      legendgroup: a.redditId,
      x: tr.x,
      y: tr.y,
      customdata: tr.customdata,
      line: { color, width: 2.4 },
      marker: { size: 5, color },
      hovertemplate:
        `<b>${name}</b><br>` +
        "%{customdata[4]} · %{customdata[5]}<br>" +
        "%{customdata[2]}<br>" +
        "<b>VADER compound</b>: %{y:.3f}<br>" +
        "<extra>Click to open Reddit post</extra>",
    });
  });

  const layout = {
    paper_bgcolor: "#0a0a0a",
    plot_bgcolor: "#101010",
    font: { color: "#fde68a", size: 11 },
    margin: { t: 52, r: 24, b: 56, l: 56 },
    title: {
      text: "Comment-index trajectory (raw VADER per comment) — up to 5 posts",
      font: { size: 13, color: "#fef08a" },
    },
    xaxis: {
      title: "Comment number (chronological within post)",
      gridcolor: "#333",
      zeroline: false,
      dtick: 5,
    },
    yaxis: {
      title: "VADER compound",
      gridcolor: "#333",
      zeroline: true,
      zerolinecolor: "#666",
      range: [-1, 1],
    },
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      x: 0,
      font: { size: 9, color: "#fde68a" },
    },
    shapes: [
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
    ],
  };

  if (!traces.length) {
    traces.push({
      type: "scatter",
      x: [0],
      y: [0],
      mode: "text",
      text: ["No comments in API payload for these posts"],
      textfont: { color: "#f87171", size: 13 },
      showlegend: false,
    });
    layout.annotations = [
      {
        text: "Threads have no comment bodies in this slice — try Refresh.",
        xref: "paper",
        yref: "paper",
        x: 0.5,
        y: 0.55,
        showarrow: false,
        font: { color: "#fde68a", size: 12 },
      },
    ];
  }

  return { traces, layout };
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
    plotEl,
    plotLinesEl,
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
      Plotly.purge(plotEl);
      if (plotLinesEl) Plotly.purge(plotLinesEl);
      Plotly.newPlot(
        plotEl,
        [
          {
            type: "scatter",
            x: [0],
            y: [0],
            mode: "text",
            text: ["No data"],
            textfont: { color: "#f87171", size: 14 },
          },
        ],
        {
          paper_bgcolor: "#0a0a0a",
          plot_bgcolor: "#101010",
          xaxis: { visible: false },
          yaxis: { visible: false },
          annotations: [
            {
              text: "No posts — try another subreddit or refresh.",
              xref: "paper",
              yref: "paper",
              x: 0.5,
              y: 0.5,
              showarrow: false,
              font: { color: "#fde68a", size: 14 },
            },
          ],
        },
        { responsive: true, displaylogo: false }
      );
      return;
    }

    const { traces, layout } = buildGroupedBarPlot(subset);
    Plotly.purge(plotEl);
    Plotly.newPlot(plotEl, traces, layout, plotConfig);

    if (plotLinesEl) {
      const linePlot = buildCommentIndexLinePlot(subset);
      Plotly.purge(plotLinesEl);
      Plotly.newPlot(plotLinesEl, linePlot.traces, linePlot.layout, plotConfig).then(
        (gd) => {
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
      );
    }

    statusEl.textContent = `Analyzed ${subset.length} post(s) · unique posts merged from API: ${meta.fetchedUniquePosts} · Line chart: up to 5 posts · Click a point to open Reddit · Refresh for a new sample.`;
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
  const plotEl = document.getElementById("liveSubredditPlot");
  const plotLinesEl = document.getElementById("liveSubredditPlotLines");
  const statusEl = document.getElementById("liveSubredditStatus");
  const subSelect = document.getElementById("liveSubredditFilter");
  const metaEl = document.getElementById("liveSubredditMeta");
  const refreshBtn = document.getElementById("liveSubredditRefresh");
  if (!plotEl || !statusEl || !subSelect) return;

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => window.location.reload());
  }

  runLiveSubredditDashboard({
    plotEl,
    plotLinesEl,
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
