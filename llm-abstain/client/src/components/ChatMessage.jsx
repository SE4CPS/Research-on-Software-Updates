import { useState } from "react";
import GateTimeline from "./GateTimeline";
import SoftwareCard from "./SoftwareCard";

export default function ChatMessage({ message, onSend }) {
  const { role, type, content } = message;

  if (role === "user") {
    return (
      <div className="message user msg-enter">
        <div className="message-bubble user-bubble">{content}</div>
        <div className="message-avatar user-avatar">U</div>
      </div>
    );
  }

  if (type === "text") {
    return (
      <div className="message assistant msg-enter">
        <div className="message-avatar bot-avatar" title="Release Master">RM</div>
        <div className="message-bubble">{content}</div>
      </div>
    );
  }

  if (type === "pipeline-result") {
    return (
      <div className="message assistant msg-enter">
        <div className="message-avatar bot-avatar" title="Release Master">RM</div>
        <PipelineResult data={content} onSend={onSend} />
      </div>
    );
  }

  return null;
}

const TOOL_LABELS = {
  search_releases:         "Releases",
  search_cves:             "CVEs",
  search_breaking_changes: "Breaking Changes",
};

function PipelineResult({ data, onSend }) {
  const { decision, compositeScore, reason, suggestions, gates, processedAs, queryType, agentDecision } = data;
  const [gatesOpen, setGatesOpen] = useState(false);

  const confidenceColor =
    decision === "confident" ? "var(--accent-teal)"
    : decision === "suggest" ? "var(--accent-amber)"
    : "var(--accent-coral)";

  const badgeLabel =
    decision === "confident" ? "Confident"
    : decision === "suggest" ? "Suggestion"
    : "I don't know";

  return (
    <div className="message-bubble pipeline-bubble">
      {/* Interpreted as */}
      {processedAs && (
        <div className="processed-as">
          Interpreted as: <strong>{processedAs}</strong>
        </div>
      )}

      {/* Decision + confidence */}
      <div className="decision-row">
        <span className={`decision-badge ${decision}`}>{badgeLabel}</span>
        <div className="confidence-inline">
          <div className="confidence-track">
            <div className="confidence-fill" style={{ width: `${(compositeScore || 0) * 100}%`, background: confidenceColor }} />
          </div>
          <span className="confidence-label" style={{ color: confidenceColor }}>
            {((compositeScore || 0) * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Collapsible gate timeline */}
      {gates && gates.length > 0 && (
        <div className="gates-section">
          <button className="gates-toggle" onClick={() => setGatesOpen((o) => !o)}>
            <span>Validation Gates</span>
            <span className="gates-toggle-counts">
              {gates.filter((g) => g.result === "pass").length} passed ·{" "}
              {gates.filter((g) => g.result === "fail").length} failed
            </span>
            <svg
              width="12" height="12" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
              style={{ transform: gatesOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}
            >
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>
          {gatesOpen && <GateTimeline gates={gates} />}
        </div>
      )}

      {/* Confident: LLM natural-language summary */}
      {decision === "confident" && data.llmResponse && (
        <div className="llm-summary">
          <span className="llm-summary-label">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61z"/>
            </svg>
            AI Summary
          </span>
          <p className="llm-summary-text">{data.llmResponse.text}</p>
          <span className="llm-summary-model">{data.llmResponse.model}</span>
        </div>
      )}

      {/* Confident: software data card */}
      {decision === "confident" && data.data && (
        <SoftwareCard data={data.data} queryType={queryType} />
      )}

      {/* I don't know */}
      {decision === "abstain" && (
        <div className="abstain-inline">
          <p className="reason">
            {reason || "I don't know — releasetrain.io doesn't have enough information for this query."}
          </p>
          <p className="abstain-note">
            Only verified data from releasetrain.io is returned. The system refuses to guess.
          </p>
        </div>
      )}

      {/* Suggestions — clickable */}
      {decision === "suggest" && suggestions && (
        <div className="suggest-inline">
          <p className="reason">{reason}</p>
          <div className="suggest-label">Did you mean:</div>
          <ul className="suggest-list">
            {suggestions.map((s, i) => (
              <li key={i} className="suggest-item" onClick={() => onSend?.(s.name)}>
                <span>{s.name}</span>
                <span className="suggest-score">{(s.score * 100).toFixed(0)}% match · click to search</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
