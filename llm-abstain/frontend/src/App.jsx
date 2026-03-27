import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';
const MAX_CHAT_ITEMS = 3;
const VENDORS_API = `${API_BASE}/api/vendors`;
const ANALYTICS_BASE = `${API_BASE}/api/analytics/summary`;
const VENDORS_DIRECT_URL = 'https://releasetrain.io/api/c/names';
const MAX_VENDORS_DISPLAY = 100;

function buildTodayTopVendors(recentEvents) {
  if (!Array.isArray(recentEvents) || recentEvents.length === 0) return [];
  const toLocalDayKey = (value) => {
    const d = new Date(value);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  };
  const todayKey = toLocalDayKey(Date.now());
  const counts = new Map();
  for (const e of recentEvents) {
    const dayKey = toLocalDayKey(e?.ts || Date.now());
    if (dayKey !== todayKey) continue;
    const vendor = String(e?.vendor || 'Unknown').trim() || 'Unknown';
    counts.set(vendor, (counts.get(vendor) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([vendor, count]) => ({ vendor, count }));
}

function cleanRedundantAnswerText(rawAnswer, structured) {
  const text = String(rawAnswer || '').trim();
  if (!text) return '';
  const hasVersion = Boolean(structured?.version);
  const hasUrl = Boolean(structured?.url);
  const hasNotes = Boolean(structured?.notes);
  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  const filtered = lines.filter((line) => {
    if (hasVersion && /^patch\s*version\s*:/i.test(line)) return false;
    if (hasVersion && /^version\s*:/i.test(line)) return false;
    if (hasUrl && /^url\s*:/i.test(line)) return false;
    if (hasNotes && /^release\s*notes?\s*:/i.test(line)) return false;
    return true;
  });
  return filtered.join('\n').trim();
}

function buildDebugView(rawResponse) {
  const sourceUrl = rawResponse?.sourceUrl || rawResponse?.additional?.versionUrl || '';
  const versionSearchTags = Array.isArray(rawResponse?.versionSearchTags)
    ? rawResponse.versionSearchTags
    : [];
  const vendor = String(rawResponse?.vendor || '').trim() || 'Unknown';
  const status = String(rawResponse?.status || '').trim() || 'unknown';
  const isVendorFound = Number(rawResponse?.isVendorFound) === 1 ? 1 : 0;
  return {
    sourceUrl,
    versionSearchTags,
    vendor,
    status,
    isVendorFound,
  };
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [chat, setChat] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [vendors, setVendors] = useState([]);
  const [vendorsLoading, setVendorsLoading] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [aboutPanelOpen, setAboutPanelOpen] = useState(true);
  const [vendorsPanelOpen, setVendorsPanelOpen] = useState(true);
  const [queryTipsPanelOpen, setQueryTipsPanelOpen] = useState(true);
  const todayTopVendors = buildTodayTopVendors(analytics?.recent);
  const todayTopVendorMax = todayTopVendors.length > 0
    ? Math.max(...todayTopVendors.map((v) => v.count), 1)
    : 1;

  const loadAnalytics = useCallback(async () => {
    setAnalyticsLoading(true);
    try {
      const params = new URLSearchParams({
        rangeDays: '1',
        flow: 'all',
        status: 'all',
        vendor: '',
      });
      const res = await fetch(`${ANALYTICS_BASE}?${params}`);
      if (!res.ok) throw new Error(`Analytics API error: ${res.status}`);
      const data = await res.json();
      setAnalytics(data?.summary ?? null);
    } catch {
      setAnalytics(null);
    } finally {
      setAnalyticsLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadList(url) {
      const res = await fetch(url);
      if (!res.ok) return null;
      const data = await res.json();
      return Array.isArray(data) ? data : (data?.names ?? data?.data ?? []);
    }
    (async () => {
      try {
        let list = await loadList(VENDORS_API);
        if ((!list || list.length === 0) && VENDORS_DIRECT_URL) {
          try {
            list = await loadList(VENDORS_DIRECT_URL) ?? [];
          } catch {
            list = [];
          }
        }
        if (!cancelled) setVendors(list || []);
      } catch {
        if (!cancelled) setVendors([]);
      } finally {
        if (!cancelled) setVendorsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      if (mounted) await loadAnalytics();
    })();
    const interval = setInterval(() => {
      if (mounted) loadAnalytics();
    }, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [loadAnalytics]);

  const handleAsk = async () => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setQuestion('');
    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API_BASE}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed })
      });
      if (!res.ok) {
        let detail = '';
        try {
          const errBody = await res.clone().json();
          if (errBody?.error) detail = ` — ${errBody.error}`;
          else if (typeof errBody?.message === 'string') detail = ` — ${errBody.message}`;
        } catch {
          try {
            const t = await res.text();
            if (t && t.length < 200) detail = ` — ${t.slice(0, 200)}`;
          } catch {
            /* ignore */
          }
        }
        throw new Error(`Backend error: ${res.status}${detail}`);
      }
      const data = await res.json();
      const structured = {
        version: data.version || data.main || '',
        vendor: data.vendor || '',
        notes: data.additional?.versionReleaseNotes || '',
        license: data.additional?.versionProductLicense || '',
        url: data.additional?.versionUrl || '',
      };
      const entry = {
        prompt: trimmed,
        response: {
          answer: cleanRedundantAnswerText(data.answer, structured),
          version: structured.version,
          vendor: structured.vendor,
          notes: structured.notes,
          license: structured.license,
          url: structured.url,
        },
        rawResponse: data
      };
      setChat((prev) => [...prev, entry].slice(-MAX_CHAT_ITEMS));
      await loadAnalytics();
    } catch (e) {
      setError(e.message || 'Unexpected error.');
      setChat((prev) => [
        ...prev,
        { prompt: trimmed, response: null, error: e.message, rawResponse: null }
      ].slice(-MAX_CHAT_ITEMS));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className={`app-root ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <div className="hero-bg">
        <img src="/Asset/background.gif" alt="" className="hero-bg-image" aria-hidden="true" />
      </div>

      <aside className={`nav-sidebar ${sidebarOpen ? 'nav-sidebar-open' : ''}`} aria-label="Navigation">
        <div className="nav-sidebar-inner">
          <div className="nav-sidebar-section">
            <details
              className="nav-about-collapsible"
              open={aboutPanelOpen}
              onToggle={(e) => setAboutPanelOpen(e.currentTarget.open)}
            >
              <summary className="nav-about-collapsible-summary">What is ReleaseHub?</summary>
              <div className="nav-about-collapsible-body">
                <p className="nav-about-text">
                  ReleaseHub is a generative AI-powered release note intelligence system that provides verified
                  version, patch, and licensing information for operating systems and software components from
                  trusted vendors. It is designed to minimize hallucinations by enforcing external knowledge
                  restriction and applying Best-of-N response verification, ensuring that all outputs are
                  grounded in validated vendor data. By enabling reliable querying of release notes, version
                  updates, and licensing details, ReleaseHub helps developers and system administrators make
                  accurate, production-ready decisions.
                </p>
              </div>
            </details>
          </div>
          <div className="nav-sidebar-section">
            <h2 className="nav-sidebar-title nav-sidebar-title-yellow">ANALYTICS DASHBOARD</h2>
            <div
              className="dashboard-panel dashboard-panel-analytics"
              aria-label="Analytics dashboard: KPIs, filters, and charts"
            >
              <div
                className={`dashboard-kpi-section${analytics && analyticsLoading ? ' dashboard-kpi-section--refreshing' : ''}`}
                aria-label="Key performance indicators"
              >
                {analytics && analyticsLoading ? (
                  <div className="dashboard-kpi-section-title">
                    <span className="dashboard-refresh-hint" aria-live="polite">
                      Updating…
                    </span>
                  </div>
                ) : null}
                {!analytics && analyticsLoading ? (
                  <div className="dashboard-kpi-grid dashboard-kpi-grid-2x2" aria-hidden>
                    {[
                      { cls: 'dashboard-kpi-card--answer', label: 'Queries answered' },
                      { cls: 'dashboard-kpi-card--abstain', label: 'Abstain responses' },
                      { cls: 'dashboard-kpi-card--patch', label: 'Patch-related queries' },
                      { cls: 'dashboard-kpi-card--os', label: 'OS version queries' },
                    ].map(({ cls, label }) => (
                      <div
                        key={label}
                        className={`dashboard-kpi-card dashboard-kpi-card--static dashboard-kpi-card--skeleton ${cls}`}
                      >
                        <span className="dashboard-kpi-card-label">{label}</span>
                        <span className="dashboard-kpi-card-value dashboard-kpi-skeleton-value">—</span>
                        <span className="dashboard-kpi-card-hint">Loading…</span>
                      </div>
                    ))}
                  </div>
                ) : !analytics ? (
                  <>
                    <div className="dashboard-kpi-grid dashboard-kpi-grid-2x2">
                      {[
                        'Queries answered',
                        'Abstain responses',
                        'Patch-related queries',
                        'OS version queries',
                      ].map((label) => (
                        <div key={label} className="dashboard-kpi-card dashboard-kpi-card--static dashboard-kpi-card--nodata">
                          <span className="dashboard-kpi-card-label">{label}</span>
                          <span className="dashboard-kpi-card-value">0</span>
                          <span className="dashboard-kpi-card-hint">No data</span>
                        </div>
                      ))}
                    </div>
                    <p className="nav-vendors-empty">Analytics unavailable. Check API and try Refresh.</p>
                  </>
                ) : (
                  <>
                    <div className="dashboard-kpi-grid dashboard-kpi-grid-2x2">
                      <div className="dashboard-kpi-card dashboard-kpi-card--answer dashboard-kpi-card--static">
                        <span className="dashboard-kpi-card-label">Queries answered</span>
                        <span className="dashboard-kpi-card-value">
                          {analytics.queriesAnswered ?? analytics.respondedCount ?? 0}
                        </span>
                        <span className="dashboard-kpi-card-hint">Answer responses</span>
                      </div>
                      <div className="dashboard-kpi-card dashboard-kpi-card--abstain dashboard-kpi-card--static">
                        <span className="dashboard-kpi-card-label">Abstain responses</span>
                        <span className="dashboard-kpi-card-value">
                          {analytics.queriesAbstain ?? analytics.abstainCountKpi ?? 0}
                        </span>
                        <span className="dashboard-kpi-card-hint">Abstain / I don&apos;t know</span>
                      </div>
                      <div className="dashboard-kpi-card dashboard-kpi-card--patch dashboard-kpi-card--static">
                        <span className="dashboard-kpi-card-label">Patch-related queries</span>
                        <span className="dashboard-kpi-card-value">{analytics.patchQueryCount ?? 0}</span>
                        <span className="dashboard-kpi-card-hint">Patch flow</span>
                      </div>
                      <div className="dashboard-kpi-card dashboard-kpi-card--os dashboard-kpi-card--static">
                        <span className="dashboard-kpi-card-label">OS version queries</span>
                        <span className="dashboard-kpi-card-value">
                          {analytics.osVersionQueryCount ?? analytics.osQueryCount ?? 0}
                        </span>
                        <span className="dashboard-kpi-card-hint">OS / versions flow</span>
                      </div>
                    </div>
                  </>
                )}
              </div>

                  {analytics ? (
                    <section className="dashboard-section-block dashboard-insights-block" aria-label="Dashboard insights">
                  <div className="dashboard-block-title">TOP 3 VENDORS</div>
                  {todayTopVendors.length > 0 && (
                    <div className="dashboard-chart-block">
                      <ul className="dashboard-hbar-list">
                        {todayTopVendors.map((v) => (
                          <li key={v.vendor}>
                            <div className="dashboard-hbar-row" title={v.vendor}>
                              <span className="dashboard-hbar-name">{v.vendor}</span>
                              <span className="dashboard-hbar-track">
                                <span
                                  className="dashboard-hbar-fill"
                                  style={{
                                    width: `${(v.count / todayTopVendorMax) * 100}%`,
                                  }}
                                />
                              </span>
                              <span className="dashboard-hbar-count">{v.count}</span>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {(analytics.topReasons || []).length > 0 && (
                    <div className="dashboard-chart-block">
                      <div className="dashboard-chart-title">Abstain reasons</div>
                      <ul className="dashboard-reason-list">
                        {analytics.topReasons.map((r) => (
                          <li key={r.reason}>
                            <span className="dashboard-reason-name">{r.reason}</span>
                            <span className="dashboard-reason-count">{r.count}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {(analytics.recent || []).length > 0 && (
                    <details className="dashboard-recent">
                      <summary className="dashboard-recent-summary">Recent events</summary>
                      <ul className="dashboard-recent-list">
                        {analytics.recent.map((e, idx) => (
                          <li key={`${e.ts}-${idx}`}>
                            <span className={`dashboard-recent-status dashboard-recent-${e.status}`}>{e.status}</span>
                            <span className="dashboard-recent-meta">
                              {e.flow} · {e.vendor}
                              {e.latencyMs != null ? ` · ${e.latencyMs}ms` : ''}
                              {e.reason ? ` · ${e.reason}` : ''}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                    </section>
                  ) : null}
            </div>
          </div>
          <div className="nav-sidebar-section">
            <details
              className="nav-vendors-collapsible"
              open={vendorsPanelOpen}
              onToggle={(e) => setVendorsPanelOpen(e.currentTarget.open)}
            >
              <summary className="nav-vendors-collapsible-summary">
                System Vendors
              </summary>
              <div className="nav-vendors-collapsible-body">
                {vendorsLoading ? (
                  <p className="nav-vendors-loading">Loading…</p>
                ) : vendors.length === 0 ? (
                  <p className="nav-vendors-empty">No vendors loaded.</p>
                ) : (
                  <ul className="nav-vendors-list" aria-label="Vendor list">
                    {vendors.slice(0, MAX_VENDORS_DISPLAY).map((name, i) => (
                      <li key={i} className="nav-vendor-item">
                        {typeof name === 'string' ? name : String(name)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </details>
          </div>
          <div className="nav-sidebar-section">
            <details
              className="nav-tips-collapsible"
              open={queryTipsPanelOpen}
              onToggle={(e) => setQueryTipsPanelOpen(e.currentTarget.open)}
            >
              <summary className="nav-tips-collapsible-summary">Query Tips</summary>
              <div className="nav-tips-collapsible-body">
                <ol className="nav-vendors-list nav-numbered-list" aria-label="Query tips">
                  <li className="nav-vendor-item">
                    <strong className="nav-tip-label">For updates:</strong> use dates as MM-DD-YYYY.
                  </li>
                  <li className="nav-vendor-item">
                    <strong className="nav-tip-label">For Patches:</strong> use wording like patch for Linux on 02-14-2026.
                  </li>
                  <li className="nav-vendor-item">
                    <strong className="nav-tip-label">For Operating Systems:</strong> include product names like Android, iOS, Windows, or linux-dist.
                  </li>
                </ol>
              </div>
            </details>
          </div>
        </div>
      </aside>

      <button
        type="button"
        className="nav-sidebar-toggle"
        onClick={() => setSidebarOpen((o) => !o)}
        aria-expanded={sidebarOpen}
        aria-label={sidebarOpen ? 'Close navigation' : 'Open navigation'}
      >
        {sidebarOpen ? '◀' : '▶'}
      </button>

      <header className="chat-header-static">
        <img src="/Asset/Logo.png" alt="ReleaseHub" className="chat-header-logo" draggable={false} />
      </header>

      <main className="chat-main">
        <section className="chat-screen" aria-label="Chat">
          {chat.map((entry, i) => (
            <div key={i} className="chat-exchange">
              <div className="chat-prompt">
                <span className="chat-prompt-label">
                  <span className="chat-prompt-icon" aria-hidden="true">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
                    </svg>
                  </span>
                  User
                </span>
                <div className="chat-prompt-bubble">
                  <p className="chat-prompt-text">{entry.prompt}</p>
                </div>
              </div>
              <div className="chat-response-free">
                <span className="chat-response-label">Release Master</span>
                {entry.error ? (
                  <p className="chat-response-error">{entry.error}</p>
                ) : entry.response ? (
                  <div className="chat-response-card">
                    <div className="chat-main-answer">
                      <span className="chat-main-answer-label">Main Answer</span>
                      <p className="chat-main-answer-value">{entry.response.version || entry.response.answer}</p>
                    </div>
                    <p className="chat-response-text">{entry.response.answer}</p>
                    <dl className="chat-response-meta">
                      {entry.response.version && (
                        <>
                          <dt>Version:</dt>
                          <dd>{entry.response.version}</dd>
                        </>
                      )}
                      {entry.response.url && (
                        <>
                          <dt className="chat-response-meta-newline">Details:</dt>
                          <dd>
                            <a href={entry.response.url} target="_blank" rel="noreferrer">
                              {entry.response.url}
                            </a>
                          </dd>
                        </>
                      )}
                      {entry.response.notes && (
                        <>
                          <dt className="chat-response-meta-newline">Release notes:</dt>
                          <dd>
                            {entry.response.notes.startsWith('http') ? (
                              <a href={entry.response.notes} target="_blank" rel="noreferrer">
                                {entry.response.notes}
                              </a>
                            ) : (
                              <span className="chat-response-notes-text">{entry.response.notes}</span>
                            )}
                          </dd>
                        </>
                      )}
                      {entry.response.license && (
                        <>
                          <dt>License</dt>
                          <dd>{entry.response.license}</dd>
                        </>
                      )}
                    </dl>
                  </div>
                ) : null}
                {showDebug && entry.rawResponse != null && (
                  <details className="chat-debug-details">
                    <summary className="chat-debug-summary">Debug details</summary>
                    {(() => {
                      const debugView = buildDebugView(entry.rawResponse);
                      return (
                        <div className="chat-debug-pre">
                          <p><strong>Source API:</strong>{' '}
                            {debugView.sourceUrl ? (
                              <a href={debugView.sourceUrl} target="_blank" rel="noreferrer">
                                {debugView.sourceUrl}
                              </a>
                            ) : (
                              <span>N/A</span>
                            )}
                          </p>
                          <p><strong>versionSearchTags:</strong> {debugView.versionSearchTags.length > 0 ? debugView.versionSearchTags.join(', ') : '[]'}</p>
                          <p><strong>Vendor:</strong> {debugView.vendor}</p>
                          <p><strong>Status:</strong> {debugView.status}</p>
                          <p><strong>isVendorFound:</strong> {debugView.isVendorFound}</p>
                        </div>
                      );
                    })()}
                  </details>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="chat-exchange">
              <div className="chat-prompt">
                <span className="chat-prompt-label">
                  <span className="chat-prompt-icon" aria-hidden="true">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
                    </svg>
                  </span>
                  User
                </span>
                <div className="chat-prompt-bubble">
                  <p className="chat-prompt-text">…</p>
                </div>
              </div>
              <div className="chat-response-free">
                <p className="chat-response-text">Thinking...</p>
              </div>
            </div>
          )}
        </section>

        <section className="chat-input-block">
          <div className="input-row input-row-full">
            <input
              className="query-input"
              type="text"
              placeholder="What is the version of OS Android on 02-14-2026?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              type="button"
              className="query-button"
              onClick={handleAsk}
              disabled={loading}
            >
              {loading ? '…' : 'Ask'}
            </button>
          </div>
          <div className="example-row">
            <label className="example-pill chat-debug-toggle">
              <input
                type="checkbox"
                checked={showDebug}
                onChange={(e) => setShowDebug(e.target.checked)}
              />
              <span>Debug response</span>
            </label>
          </div>
        </section>
      </main>
    </div>
  );
}
