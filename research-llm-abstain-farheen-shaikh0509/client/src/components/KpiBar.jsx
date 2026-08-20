// KpiBar — compact inline metrics for the header bar
// BERTScore + Composite side by side, no cards, tight layout

const ZERO_SCORE = { precision: 0, recall: 0, f1: 0, risk: "high" };

export default function KpiBar({ bertscore, compositeScore, decision }) {
  if (compositeScore === null || compositeScore === undefined) return null;

  const score = (decision === "confident" && bertscore) ? bertscore : ZERO_SCORE;

  return (
    <div className="kpi-inline">
      <BertKpi bertscore={score} />
      <div className="kpi-sep" />
      <CompositeKpi score={compositeScore} decision={decision} />
    </div>
  );
}

// ── BERTScore metric ─────────────────────────────

function BertKpi({ bertscore }) {
  const { precision, recall, f1, risk } = bertscore;

  const color =
    risk === "low"    ? "var(--accent-teal)"
    : risk === "medium" ? "var(--accent-amber)"
    : "var(--accent-coral)";

  const riskLabel = risk === "low" ? "Low" : risk === "medium" ? "Med" : "High";

  return (
    <div className="kpi-metric">
      <span className="kpi-metric-label">Hallucination</span>
      <div className="kpi-metric-row">
        <SmallRing value={f1} color={color} />
        <div className="kpi-metric-details">
          <span className="kpi-metric-main" style={{ color }}>
            {(f1 * 100).toFixed(1)}%
          </span>
          <span className="kpi-metric-sub">
            Grnd&nbsp;<b>{(precision * 100).toFixed(0)}%</b>
            &nbsp;·&nbsp;
            Cov&nbsp;<b>{(recall * 100).toFixed(0)}%</b>
          </span>
          <span className="kpi-risk-pill" style={{ color, background: `${color}15`, borderColor: `${color}25` }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: color, display: "inline-block", marginRight: 4 }} />
            {riskLabel} Risk
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Small SVG ring ───────────────────────────────

function SmallRing({ value, color }) {
  const r    = 22;
  const circ = 2 * Math.PI * r;
  const off  = circ * (1 - Math.min(1, Math.max(0, value)));
  return (
    <svg width="54" height="54" viewBox="0 0 54 54" style={{ flexShrink: 0 }}>
      <circle cx="27" cy="27" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
      <circle
        cx="27" cy="27" r={r} fill="none"
        stroke={color} strokeWidth="5"
        strokeDasharray={circ} strokeDashoffset={off}
        strokeLinecap="round"
        transform="rotate(-90 27 27)"
        style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.16,1,0.3,1)" }}
      />
      <text x="27" y="32" textAnchor="middle" fill={color}
        fontSize="11" fontWeight="700" fontFamily="var(--font-mono)">
        {Math.round(value * 100)}
      </text>
    </svg>
  );
}

// ── Composite metric ─────────────────────────────

function CompositeKpi({ score, decision }) {
  const pct = Math.min(100, Math.round((score || 0) * 100));
  const color =
    decision === "confident" ? "var(--accent-teal)"
    : decision === "suggest"  ? "var(--accent-amber)"
    : "var(--accent-coral)";

  const label =
    decision === "confident" ? "Confident"
    : decision === "suggest"  ? "Suggestion"
    : "Abstained";

  return (
    <div className="kpi-metric">
      <span className="kpi-metric-label">Confidence</span>
      <div className="kpi-metric-row" style={{ alignItems: "center", gap: 10 }}>
        <span className="kpi-composite-num" style={{ color }}>{pct}<span style={{ fontSize: 12 }}>%</span></span>
        <div className="kpi-metric-details">
          <div className="kpi-mini-track">
            <div className="kpi-mini-fill" style={{ width: `${pct}%`, background: color }} />
            {[60, 80].map(b => (
              <div key={b} className="kpi-mini-marker" style={{ left: `${b}%` }} />
            ))}
          </div>
          <span className="kpi-risk-pill" style={{ color, background: `${color}15`, borderColor: `${color}25` }}>
            {label}
          </span>
        </div>
      </div>
    </div>
  );
}
