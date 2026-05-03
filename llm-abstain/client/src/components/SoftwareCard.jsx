import { useState } from "react";

export default function SoftwareCard({ data, queryType }) {
  if (!data) return null;
  const type = queryType || data.queryType || "general";
  if (type === "cve")      return <CveCard data={data} />;
  if (type === "patch")    return <PatchCard data={data} />;
  if (type === "version")  return <VersionCard data={data} />;
  if (type === "critical" || type === "breaking") return <CriticalCard data={data} />;
  return <GeneralCard data={data} />;
}

// ── Date banner ──
function DateBanner({ dateLabel, dateFilter }) {
  if (!dateFilter) return null;
  return (
    <div className="date-banner">
      Showing results for <strong>{dateLabel}</strong> — {dateFilter}
    </div>
  );
}

// ── Copy button ──
function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  function copy(e) {
    e.stopPropagation();
    navigator.clipboard.writeText(text).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }
  return (
    <button className="copy-btn" onClick={copy} title="Copy">
      {copied ? "✓" : "Copy"}
    </button>
  );
}

// ── CVE Card ──
function CveCard({ data }) {
  const [showAll, setShowAll] = useState(false);
  const severityColors = { security: "#ff8800", critical: "#ff4444", high: "#ff8800", medium: "#ffcc00", low: "#44cc44" };
  const severityKey = String(data.severity || "").toLowerCase().split(/[\s,]/)[0];
  const severityColor = severityColors[severityKey] || "#ff8800";

  const cves = data.recentCves || [];
  const visible = showAll ? cves : cves.slice(0, 3);

  return (
    <div className="software-card">
      <h3>Security Advisory</h3>
      <DateBanner dateLabel={data.dateLabel} dateFilter={data.dateFilter} />
      <div className="data-grid">
        {data.software && <Field label="Software" value={data.software} />}
        {data.cveId && (
          <div className="data-field">
            <div className="data-field-label">Top CVE ID</div>
            <div className="data-field-value" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <a className="cve-link" href={data.sourceUrl} target="_blank" rel="noreferrer">
                {data.cveId}
              </a>
              <CopyBtn text={data.cveId} />
            </div>
          </div>
        )}
        {data.severity && (
          <div className="data-field">
            <div className="data-field-label">Severity</div>
            <div className="data-field-value">
              <span className="severity-badge" style={{ background: severityColor }}>{data.severity}</span>
            </div>
          </div>
        )}
        {data.version    && <Field label="Affected Version" value={data.version} />}
        {data.releaseDate && <Field label="Published" value={data.releaseDate} />}
        {data.description && <Field label="Description" value={data.description} wide />}
      </div>

      {/* CVE list */}
      {cves.length > 1 && (
        <div style={{ marginTop: 12 }}>
          <div className="section-label">Recent CVEs ({data.totalCves} total)</div>
          {visible.map((c, i) => (
            <CveRow key={i} cve={c} />
          ))}
          {cves.length > 3 && (
            <button className="show-more-btn" onClick={() => setShowAll((s) => !s)}>
              {showAll ? "Show less" : `Show ${cves.length - 3} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function CveRow({ cve }) {
  return (
    <div className="cve-row" onClick={() => cve.url && window.open(cve.url, "_blank")}
      title={cve.url ? "Open in NVD" : undefined}>
      <div className="cve-row-header">
        <span className="cve-row-id">{cve.cveId}</span>
        <CopyBtn text={cve.cveId} />
        <span className="cve-row-meta">v{cve.version} · {cve.date}</span>
        {cve.url && <span className="cve-row-ext">↗</span>}
      </div>
      {cve.notes && <div className="cve-row-notes">{cve.notes}</div>}
    </div>
  );
}

// ── Patch Card ──
function PatchCard({ data }) {
  const [showAll, setShowAll] = useState(false);
  const patches = data.recentPatches || [];
  const visible = showAll ? patches : patches.slice(0, 3);

  return (
    <div className="software-card">
      <h3>Patch Information</h3>
      <DateBanner dateLabel={data.dateLabel} dateFilter={data.dateFilter} />
      <div className="data-grid">
        {data.software        && <Field label="Software" value={data.software} />}
        {data.version         && <Field label="Latest Patch" value={data.version} />}
        {data.releaseDate     && <Field label="Release Date" value={data.releaseDate} />}
        {data.releaseChannel  && <Field label="Channel" value={data.releaseChannel} />}
        {data.description     && <Field label="Notes" value={data.description} wide />}
        {data.sourceUrl && (
          <div className="data-field" style={{ gridColumn: "1 / -1" }}>
            <div className="data-field-label">Source</div>
            <a className="source-link" href={data.sourceUrl} target="_blank" rel="noreferrer">{data.sourceUrl}</a>
          </div>
        )}
      </div>

      {patches.length > 1 && (
        <div style={{ marginTop: 12 }}>
          <div className="section-label">Recent Patches</div>
          {visible.map((p, i) => (
            <div key={i} className="cve-row" onClick={() => p.url && window.open(p.url, "_blank")}>
              <div className="cve-row-header">
                <span className="cve-row-id">v{p.version}</span>
                {p.isCve && <span className="cve-badge">CVE</span>}
                {p.cveId && <><span className="cve-row-meta">{p.cveId}</span><CopyBtn text={p.cveId} /></>}
                <span className="cve-row-meta">{p.date}</span>
                {p.url && <span className="cve-row-ext">↗</span>}
              </div>
              {p.notes && <div className="cve-row-notes">{p.notes}</div>}
            </div>
          ))}
          {patches.length > 3 && (
            <button className="show-more-btn" onClick={() => setShowAll((s) => !s)}>
              {showAll ? "Show less" : `Show ${patches.length - 3} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Version Card ──
function VersionCard({ data }) {
  return (
    <div className="software-card">
      <h3>Version Information</h3>
      <DateBanner dateLabel={data.dateLabel} dateFilter={data.dateFilter} />
      <div className="data-grid">
        {data.software && <Field label="Software" value={data.software} />}
        {data.version && (
          <div className="data-field">
            <div className="data-field-label">Latest Version</div>
            <div className="data-field-value" style={{ fontWeight: 700, fontSize: "1.1rem" }}>{data.version}</div>
          </div>
        )}
        {data.releaseDate    && <Field label="Release Date" value={data.releaseDate} />}
        {data.releaseChannel && <Field label="Channel" value={data.releaseChannel} />}
        {data.totalEntries != null && <Field label="Releases Tracked" value={String(data.totalEntries)} />}
        {data.description    && <Field label="Notes" value={data.description} wide />}
        {data.sourceUrl && (
          <div className="data-field" style={{ gridColumn: "1 / -1" }}>
            <div className="data-field-label">Source</div>
            <a className="source-link" href={data.sourceUrl} target="_blank" rel="noreferrer">{data.sourceUrl}</a>
          </div>
        )}
      </div>

      {(data.recentVersions || []).length > 1 && (
        <div style={{ marginTop: 12 }}>
          <div className="section-label">Recent Releases</div>
          <table className="version-table">
            <thead>
              <tr><th>Version</th><th>Date</th><th>Channel</th><th>CVE</th></tr>
            </thead>
            <tbody>
              {data.recentVersions.map((v, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>{v.version}</td>
                  <td>{v.date}</td>
                  <td>{v.channel}</td>
                  <td>{v.isCve ? <span style={{ color: "#ff8800", fontWeight: 600 }}>Yes</span> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── General Card ──
function GeneralCard({ data }) {
  if (Array.isArray(data)) {
    return (
      <div className="software-card">
        <h3>Software Information</h3>
        <div className="data-grid">
          {data.slice(0, 10).map((item, i) => (
            <div key={i} className="data-field">
              <div className="data-field-label">{item.versionProductBrand || `Entry ${i + 1}`}</div>
              <div className="data-field-value">{item.versionNumber || JSON.stringify(item).slice(0, 80)}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="software-card">
      <h3>Software Information</h3>
      <div className="data-grid">
        {data.software       && <Field label="Software" value={data.software} />}
        {data.version        && <Field label="Version" value={data.version} />}
        {data.releaseDate    && <Field label="Release Date" value={data.releaseDate} />}
        {data.releaseChannel && <Field label="Channel" value={data.releaseChannel} />}
        {data.totalEntries != null && <Field label="Entries Tracked" value={String(data.totalEntries)} />}
        {data.description    && <Field label="Notes" value={data.description} wide />}
        {data.sourceUrl && (
          <div className="data-field" style={{ gridColumn: "1 / -1" }}>
            <div className="data-field-label">Source</div>
            <a className="source-link" href={data.sourceUrl} target="_blank" rel="noreferrer">{data.sourceUrl}</a>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Critical / Breaking Card (shared for all breakingType queries) ──
function CriticalCard({ data }) {
  const [showAll, setShowAll] = useState(false);
  const entries = data.entries || [];
  const visible = showAll ? entries : entries.slice(0, 5);
  const label = data.breakingLabel || "Critical Failure";
  const isCritical = label === "Critical Failure";

  return (
    <div className="software-card">
      <h3>{label} Releases</h3>
      <DateBanner dateLabel={data.dateLabel} dateFilter={data.dateFilter} />
      <div className="data-grid">
        {data.software && <Field label="Software" value={data.software} />}
        {data.totalCritical != null && <Field label="Releases Found" value={String(data.totalCritical)} />}
      </div>

      {entries.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div className="section-label">Versions tagged "{label}" ({data.totalCritical} total{data.isBroadQuery ? ", top 10 shown" : ""})</div>
          {visible.map((e, i) => (
            <div key={i} className="cve-row" onClick={() => e.url && window.open(e.url, "_blank")}
              title={e.url ? "Open release page" : undefined}>
              <div className="cve-row-header">
                {data.isBroadQuery && e.software && (
                  <span className="broad-software-badge">{e.software}</span>
                )}
                <span className="cve-row-id">v{e.version}</span>
                <span className={isCritical ? "critical-badge" : "breaking-badge"}>{label}</span>
                {e.breakingTypes.filter((t) => t !== label).map((t, j) => (
                  <span key={j} className="breaking-badge">{t}</span>
                ))}
                <span className="cve-row-meta">{e.date} · {e.channel}</span>
                {e.isCve && <span className="cve-badge">CVE</span>}
                {e.url && <span className="cve-row-ext">↗</span>}
              </div>
              {e.notes && <div className="cve-row-notes">{e.notes}</div>}
            </div>
          ))}
          {entries.length > 5 && (
            <button className="show-more-btn" onClick={() => setShowAll((s) => !s)}>
              {showAll ? "Show less" : `Show ${entries.length - 5} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Shared field ──
function Field({ label, value, wide }) {
  if (!value && value !== 0) return null;
  const display = typeof value === "object" ? JSON.stringify(value, null, 2).slice(0, 400) : String(value).slice(0, 400);
  return (
    <div className="data-field" style={wide ? { gridColumn: "1 / -1" } : {}}>
      <div className="data-field-label">{label}</div>
      <div className="data-field-value" style={{ whiteSpace: "pre-wrap" }}>{display}</div>
    </div>
  );
}
