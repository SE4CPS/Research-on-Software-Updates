import { useState } from "react";

export default function SoftwareCard({ data, queryType }) {
  if (!data) return null;
  const type = queryType || data.queryType || "general";

  const titles = {
    breaking: `${data.breakingLabel || "Breaking"} Releases`,
    patch:    "Patch Releases",
    cve:      "Security Advisories (CVE)",
    version:  "Version Information",
    general:  "Release Information",
  };
  const title = titles[type] || "Release Information";

  return (
    <div className="software-card">
      <h3>{title}</h3>
      {data.dateFilter && (
        <div className="date-banner">
          Showing results for <strong>{data.dateLabel}</strong> — {data.dateFilter}
        </div>
      )}

      <div className="data-grid" style={{ marginBottom: 12 }}>
        {data.software   && <Field label="Software"      value={data.software} />}
        {data.totalFound != null && <Field label="Results Found" value={String(data.totalFound)} />}
        {data.breakingLabel && type === "breaking" && <Field label="Type" value={data.breakingLabel} />}
      </div>

      <EntriesTable entries={data.entries || []} showCve={type === "cve"} />
    </div>
  );
}

// ── Unified 3-column table ──────────────────────────────────────────────────
function EntriesTable({ entries, showCve }) {
  const [showAll, setShowAll] = useState(false);
  if (!entries || entries.length === 0) return null;

  const visible = showAll ? entries : entries.slice(0, 10);

  return (
    <div>
      <table className="version-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Product Name</th>
            <th>Date</th>
            {showCve && <th>CVE ID</th>}
          </tr>
        </thead>
        <tbody>
          {visible.map((e, i) => (
            <tr key={i}>
              <td style={{ fontWeight: 600 }}>
                {e.versionUrl
                  ? <a href={e.versionUrl} target="_blank" rel="noreferrer" className="source-link" style={{ fontWeight: 600 }}>{e.versionNumber ?? "—"}</a>
                  : (e.versionNumber ?? "—")
                }
              </td>
              <td>{e.versionProductName ?? "—"}</td>
              <td>{e.releaseDate ?? "—"}</td>
              {showCve && <td style={{ fontFamily: "monospace", fontSize: "0.85em" }}>{e.cveId ?? "—"}</td>}
            </tr>
          ))}
        </tbody>
      </table>

      {entries.length > 10 && (
        <button className="show-more-btn" onClick={() => setShowAll((s) => !s)}>
          {showAll ? "Show less" : `Show ${entries.length - 10} more`}
        </button>
      )}
    </div>
  );
}

// ── Shared field ────────────────────────────────────────────────────────────
function Field({ label, value }) {
  if (!value && value !== 0) return null;
  return (
    <div className="data-field">
      <div className="data-field-label">{label}</div>
      <div className="data-field-value">{String(value).slice(0, 200)}</div>
    </div>
  );
}
