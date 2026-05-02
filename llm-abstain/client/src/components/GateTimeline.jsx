// ============================================
// FILE: client/src/components/GateTimeline.jsx
// Displays the 5-gate abstain pipeline results
// ============================================

export default function GateTimeline({ gates }) {
  return (
    <div className="gates-section">
      <div className="gates-title">Abstain Pipeline Gates</div>
      {gates.map((gate) => (
        <div key={gate.gate} className="gate-item">
          <div className={`gate-num ${gate.result}`}>{gate.gate}</div>
          <div className="gate-info">
            <div className="gate-name">{gate.name}</div>
            <div className="gate-detail">
              {gate.reason ||
                (gate.result === "pass" ? "Passed validation" : "")}
            </div>
          </div>
          <div
            className="gate-score"
            style={{
              color:
                gate.result === "pass"
                  ? "var(--accent-teal)"
                  : gate.result === "fail"
                    ? "var(--accent-coral)"
                    : "var(--accent-amber)",
            }}
          >
            {(gate.score * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
}