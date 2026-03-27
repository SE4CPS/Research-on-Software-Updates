import React, { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';
const MAX_CHAT_ITEMS = 3;
const VENDORS_API = `${API_BASE}/api/vendors`;
const VENDORS_DIRECT_URL = 'https://releasetrain.io/api/c/names';
const MAX_VENDORS_DISPLAY = 100;

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [chat, setChat] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [vendors, setVendors] = useState([]);
  const [vendorsLoading, setVendorsLoading] = useState(true);

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
        throw new Error(`Backend error: ${res.status}`);
      }
      const data = await res.json();
      const entry = {
        prompt: trimmed,
        response: {
          answer: data.answer || '',
          version: data.version || data.main || '',
          vendor: data.vendor || '',
          notes: data.additional?.versionReleaseNotes || '',
          license: data.additional?.versionProductLicense || '',
          url: data.additional?.versionUrl || ''
        },
        rawResponse: data
      };
      setChat((prev) => [...prev, entry].slice(-MAX_CHAT_ITEMS));
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
            <h2 className="nav-sidebar-title">About ReleaseHub</h2>
            <p className="nav-about-text">
              ReleaseHub provides version intelligence for OS and software components. Ask questions about
              OS versions, release notes, and licenses. Data is sourced from ReleaseTrain to help you
              stay informed on product versions and licensing.
            </p>
          </div>
          <div className="nav-sidebar-section">
            <h2 className="nav-sidebar-title">Available Vendors</h2>
            <p className="nav-vendors-hint">Component and vendor names from ReleaseTrain (first {MAX_VENDORS_DISPLAY}):</p>
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
                  <>
                    <p className="chat-response-text">{entry.response.answer}</p>
                    <dl className="chat-response-meta">
                      {entry.response.version && (
                        <>
                          <dt>Version</dt>
                          <dd>{entry.response.version}</dd>
                        </>
                      )}
                      {entry.response.vendor && (
                        <>
                          <dt>Product</dt>
                          <dd>{entry.response.vendor}</dd>
                        </>
                      )}
                      {entry.response.url && (
                        <>
                          <dt className="chat-response-meta-newline">Details</dt>
                          <dd>
                            <a href={entry.response.url} target="_blank" rel="noreferrer">
                              {entry.response.url}
                            </a>
                          </dd>
                        </>
                      )}
                      {entry.response.notes && (
                        <>
                          <dt className="chat-response-meta-newline">Release notes</dt>
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
                  </>
                ) : null}
                {showDebug && entry.rawResponse != null && (
                  <details className="chat-debug-details">
                    <summary className="chat-debug-summary">Debug: raw response</summary>
                    <pre className="chat-debug-pre">{JSON.stringify(entry.rawResponse, null, 2)}</pre>
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
              placeholder="What is the version of OS Android on 2-02-2026?"
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
            <button
              type="button"
              className="example-pill"
              onClick={() =>
                setQuestion('What is the version of OS Android on 2-02-2026?')
              }
            >
              OS Android on 2-02-2026
            </button>
            <button
              type="button"
              className="example-pill"
              onClick={() =>
                setQuestion('What is the latest OS Android production version today?')
              }
            >
              Latest Android production
            </button>
            <button
              type="button"
              className="example-pill"
              onClick={() =>
                setQuestion('Show OS Android release notes and license for the latest version.')
              }
            >
              Android notes &amp; license
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
