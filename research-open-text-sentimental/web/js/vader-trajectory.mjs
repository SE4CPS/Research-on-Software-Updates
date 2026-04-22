/**
 * Shared VADER + Plotly helpers for live dashboard panels.
 */

export const RELEASETRAIN_API_BASE = "https://releasetrain.io/api/";

export function apiBaseUrl() {
  return RELEASETRAIN_API_BASE;
}

export async function fetchJson(path) {
  const url = `${apiBaseUrl()}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`HTTP ${res.status} for ${path}: ${t.slice(0, 200)}`);
  }
  return res.json();
}

export function cleanText(s) {
  if (!s) return "";
  return String(s)
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function isOpComment(c, post) {
  const v = c.is_submitter;
  if (v === true) return true;
  if (v === false) return false;
  const pa = (post.author || "").trim().toLowerCase();
  const ca = (c.author || "").trim().toLowerCase();
  return Boolean(pa) && pa === ca;
}

export function commentSortKey(c) {
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

export function emptyClassFlags() {
  return { pool: false, class1: false, class2: false };
}

export function formatSourceClassLine(flags) {
  const f = flags || emptyClassFlags();
  const parts = [];
  if (f.pool) parts.push("Pool (no post-score filter)");
  if (f.class1) parts.push("Class 1: min post score ≥0.5");
  if (f.class2) parts.push("Class 2: max post score ≤0.5");
  if (!parts.length) return "Source: (slice unknown)";
  return `Source: ${parts.join(" · ")}`;
}

export const TRAJECTORY_LINE_COLORS = [
  "#f59e0b",
  "#3b82f6",
  "#10b981",
  "#ec4899",
  "#a855f7",
  "#0ea5e9",
];

export const TRAJECTORY_SHAPES = [
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

let siaPromise = null;

export function getAnalyzer() {
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

export function analyzePost(SentimentClass, post) {
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

  const classFlags = post._rrClasses || emptyClassFlags();

  return {
    redditId: post.redditId || "",
    subreddit: post.subreddit || "",
    titleText: title.slice(0, 120),
    url: post.url || "",
    num_comments: post.num_comments,
    score: post.score,
    classFlags,
    classLabel: formatSourceClassLine(classFlags),
    trajectory: {
      x: trajX,
      y: trajY,
      customdata: trajCustom,
      nComments: trajX.length,
    },
  };
}

function chartTitlePlain(postIndex, a) {
  return `Post ${postIndex + 1} · r/${a.subreddit} · ${a.redditId}`;
}

export function buildSinglePostTrajectoryPlot(a, postIndex) {
  const color = TRAJECTORY_LINE_COLORS[postIndex % TRAJECTORY_LINE_COLORS.length];
  const tr = a.trajectory;
  const titleText = chartTitlePlain(postIndex, a);

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
        margin: { t: 50, r: 12, b: 40, l: 44 },
        title: {
          text: titleText,
          font: { size: 11, color: "#fef08a" },
          x: 0.01,
          xanchor: "left",
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
    margin: { t: 50, r: 12, b: 44, l: 48 },
    title: {
      text: titleText,
      font: { size: 11, color: "#fef08a" },
      x: 0.01,
      xanchor: "left",
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

export function purgeTrajectoryGrid(container) {
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

export function attachTrajectoryClick(gd) {
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

export function renderTrajectoryGrid(gridEl, subset, plotConfig, maxCells = 5) {
  purgeTrajectoryGrid(gridEl);
  const rows = subset.slice(0, maxCells);
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

    const sourceStrip = document.createElement("div");
    sourceStrip.className = "trajectory-source-strip";
    sourceStrip.setAttribute("role", "status");
    const label = a.classLabel && String(a.classLabel).trim() ? a.classLabel : "Source: not available";
    sourceStrip.textContent = label;

    const plotDiv = document.createElement("div");
    plotDiv.className = "js-trajectory-plot";

    cell.appendChild(head);
    cell.appendChild(sourceStrip);
    cell.appendChild(plotDiv);
    gridEl.appendChild(cell);

    const { traces, layout } = buildSinglePostTrajectoryPlot(a, i);
    Plotly.newPlot(plotDiv, traces, layout, plotConfig).then((gd) => {
      attachTrajectoryClick(gd);
    });
  });
}
