// KpiBar — center-top dashboard strip
// BERTScore hallucination detection + composite confidence bar

const ZERO_SCORE = { precision: 0, recall: 0, f1: 0, risk: "high", method: "n/a" };

export default function KpiBar({ bertscore, compositeScore, decision }) {
  if (compositeScore === null || compositeScore === undefined) return null;

  // Abstain / suggest → show zeroed BERTScore, not N/A
  const score = (decision === "confident" && bertscore) ? bertscore : ZERO_SCORE;

  return (
    <div className="kpi-bar">
      <BertScoreKpi bertscore={score} decision={decision} />
      <div className="kpi-divider" />
      <CompositeKpi score={compositeScore} decision={decision} />
    </div>
  );
}

// ── BERTScore panel ──────────────────────────────

function BertScoreKpi({ bertscore, decision }) {
  const { precision, recall, f1, risk, method } = bertscore;

  const riskColor =
    risk === "low"    ? "var(--accent-teal)"
    : risk === "medium" ? "var(--accent-amber)"
    : "var(--accent-coral)";

  const riskLabel =
    risk === "low" ? "Low Risk" : risk === "medium" ? "Medium Risk" : "High Risk";

  return (
    <div className="kpi-panel">
      <div className="kpi-label">Hallucination Detection</div>
      <div className="kpi-bert-row">
        <RingGauge value={f1} color={riskColor} />
        <div className="kpi-bert-details">
          <div className="kpi-bert-f1">
            BERTScore&nbsp;<span style={{ color: riskColor }}>{(f1 * 100).toFixed(1)}%</span>
          </div>
          <div className="kpi-bert-pr">
            <span title="Grounded: % of response words found in source data">
              Grounded&nbsp;<strong>{(precision * 100).toFixed(0)}%</strong>
            </span>
            &nbsp;·&nbsp;
            <span title="Coverage: % of source data represented in response (lower is normal for summaries)">
              Coverage&nbsp;<strong>{(recall * 100).toFixed(0)}%</strong>
            </span>
          </div>
          <div
            className="kpi-risk-badge"
            style={{ background: `${riskColor}18`, color: riskColor, borderColor: `${riskColor}30` }}
          >
            <span className="kpi-risk-dot" style={{ background: riskColor }} />
            {riskLabel}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── SVG ring gauge ───────────────────────────────

function RingGauge({ value, color }) {
  const r    = 22;
  const circ = 2 * Math.PI * r;
  const off  = circ * (1 - Math.min(1, Math.max(0, value)));
  return (
    <svg width="56" height="56" viewBox="0 0 56 56" className="kpi-ring">
      <circle cx="28" cy="28" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
      <circle
        cx="28" cy="28" r={r} fill="none"
        stroke={color} strokeWidth="5"
        strokeDasharray={circ} strokeDashoffset={off}
        strokeLinecap="round"
        transform="rotate(-90 28 28)"
        style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.16,1,0.3,1)" }}
      />
      <text x="28" y="33" textAnchor="middle" fill={color}
        fontSize="11" fontWeight="700" fontFamily="var(--font-mono)">
        {Math.round(value * 100)}
      </text>
    </svg>
  );
}

// ── Composite score panel ────────────────────────

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
    <div className="kpi-panel">
      <div className="kpi-label">Composite Confidence</div>

      <div className="kpi-composite-score" style={{ color }}>
        {pct}<span className="kpi-pct-unit">%</span>
      </div>

      <div className="kpi-bar-wrap">
        <div className="kpi-bar-track">
          <div className="kpi-bar-fill" style={{ width: `${pct}%`, background: color }} />
          {[60, 80].map(b => (
            <div key={b} className="kpi-band-marker" style={{ left: `${b}%` }}>
              <div className="kpi-band-line" />
              <span className="kpi-band-label">{b}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="kpi-composite-meta">
        <span
          className="kpi-decision-pill"
          style={{ color, background: `${color}18`, borderColor: `${color}30` }}
        >
          {label}
        </span>
      </div>
    </div>
  );
}
