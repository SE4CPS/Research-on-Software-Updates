import { useMemo, useState } from "react";
import type { EvidenceRow, PrioritizeResponse, SignalBreakdown } from "./types";

const API_BASE: string =
  ((import.meta as any)?.env?.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

async function prioritizeQuery(query: string): Promise<PrioritizeResponse> {
  const response = await fetch(`${API_BASE}/prioritize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Request failed");
  }

  return (await response.json()) as PrioritizeResponse;
}

function toolDisplayName(tool: string | null | undefined): string {
  if (tool === "release_train_api") return "ReleaseTrain API";
  if (tool === "google_news_rss") return "Google News RSS";
  return tool || "—";
}

function evidenceProvenanceLabel(by: string | undefined): { short: string; title: string; className: string } {
  switch (by) {
    case "llm":
      return {
        short: "Cited by prioritizer",
        title: "The model included evidence_refs in its structured output.",
        className: "border-emerald-500/30 bg-emerald-950/30 text-emerald-200"
      };
    case "repair_llm":
      return {
        short: "Refs repaired",
        title: "A second small LLM pass filled missing evidence_refs using only allowed ids (batched, low cost).",
        className: "border-cyan-500/30 bg-cyan-950/30 text-cyan-200"
      };
    case "deterministic":
      return {
        short: "Auto-matched",
        title: "Rules matched software name to fetched rows; verify in Evidence cited.",
        className: "border-violet-500/30 bg-violet-950/30 text-violet-200"
      };
    case "none":
      return {
        short: "No evidence ids",
        title: "No evidence_refs could be attached; row may stay unverified.",
        className: "border-slate-600 bg-slate-900/60 text-slate-400"
      };
    default:
      return {
        short: by || "—",
        title: "Evidence attachment provenance",
        className: "border-slate-600 bg-slate-900/60 text-slate-400"
      };
  }
}

function kindLabel(kind: string): string {
  if (kind === "cve") return "CVE";
  if (kind === "release_note") return "Release";
  if (kind === "news") return "News";
  return kind;
}

function truncateUrl(url: string, maxLen: number): string {
  const u = url.trim();
  if (u.length <= maxLen) return u;
  const keep = Math.floor((maxLen - 3) / 2);
  return `${u.slice(0, keep)}…${u.slice(-keep)}`;
}

function SignalMixBar({ sb }: { sb?: SignalBreakdown }) {
  if (!sb || sb.label === "no_evidence") return null;
  return (
    <div className="mt-3">
      <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        {"Cited evidence mix (weights CVE > release > news)"}
      </p>
      <div className="mt-1 flex h-2.5 overflow-hidden rounded-full bg-slate-800 ring-1 ring-slate-700/80">
        <div style={{ width: `${sb.cve_pct}%` }} className="bg-rose-500/90" title={`CVE ${sb.cve_pct}%`} />
        <div style={{ width: `${sb.release_pct}%` }} className="bg-emerald-500/90" title={`Release ${sb.release_pct}%`} />
        <div style={{ width: `${sb.news_pct}%` }} className="bg-sky-500/90" title={`News ${sb.news_pct}%`} />
      </div>
      <p className="mt-1 text-[10px] text-slate-500">
        Aligns score with <span className="text-slate-400">security vs release vs press</span> signals the prioritizer cited.
      </p>
    </div>
  );
}

function EvidenceCard({ ev }: { ev: EvidenceRow }) {
  const border =
    ev.kind === "cve"
      ? "border-rose-500/25 bg-rose-950/20"
      : ev.kind === "release_note"
        ? "border-emerald-500/25 bg-emerald-950/15"
        : ev.kind === "news"
          ? "border-sky-500/25 bg-sky-950/15"
          : "border-slate-600 bg-slate-900/50";
  return (
    <div className={`rounded-lg border px-2.5 py-2 text-xs ${border}`}>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">{ev.ref}</span>
        <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">{kindLabel(ev.kind)}</span>
        <span className="text-[10px] text-slate-500">{toolDisplayName(ev.source_tool)}</span>
        {ev.resolved === false ? (
          <span className="text-[10px] text-amber-400">unresolved</span>
        ) : null}
      </div>
      <p className="mt-1.5 text-slate-300">{ev.title}</p>
      {(ev.component || ev.version) && ev.kind !== "news" ? (
        <p className="mt-1 text-[10px] text-slate-500">
          {ev.component}
          {ev.version ? ` · ${ev.version}` : ""}
          {ev.severity ? ` · ${ev.severity}` : ""}
        </p>
      ) : null}
      {ev.kind !== "news" && ev.link ? (
        <div className="mt-2 space-y-1.5 border-t border-slate-700/60 pt-2">
          <p className="text-[10px] font-medium uppercase tracking-wide text-emerald-400/90">
            Source (ReleaseTrain API)
          </p>
          {ev.source_query ? (
            <p className="text-[10px] text-slate-400">
              Search keyword: <span className="text-slate-300">{ev.source_query}</span>
            </p>
          ) : null}
          <a
            href={ev.link}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex text-sm font-medium text-brand-400 underline-offset-2 hover:text-brand-300 hover:underline"
          >
            Open ReleaseTrain reference
          </a>
          <p
            className="break-all font-mono text-[10px] leading-relaxed text-slate-500"
            title={ev.link}
          >
            {truncateUrl(ev.link, 96)}
          </p>
        </div>
      ) : null}
      {ev.kind === "news" ? (
        <div className="mt-2 space-y-1.5 border-t border-slate-700/60 pt-2">
          <p className="text-[10px] font-medium uppercase tracking-wide text-sky-400/90">Source (Google News RSS)</p>
          {(ev.source || ev.published) && (
            <p className="text-[10px] text-slate-400">
              {ev.source ? <span className="text-slate-300">{ev.source}</span> : null}
              {ev.source && ev.published ? <span className="text-slate-600"> · </span> : null}
              {ev.published ? <span>{ev.published}</span> : null}
            </p>
          )}
          {ev.link ? (
            <>
              <a
                href={ev.link}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex text-sm font-medium text-brand-400 underline-offset-2 hover:text-brand-300 hover:underline"
              >
                Open article URL
              </a>
              <p
                className="break-all font-mono text-[10px] leading-relaxed text-slate-500"
                title={ev.link}
              >
                {truncateUrl(ev.link, 96)}
              </p>
            </>
          ) : (
            <p className="text-[10px] text-amber-400/90">No article URL in this RSS entry.</p>
          )}
        </div>
      ) : null}
    </div>
  );
}

type Message = {
  role: "user" | "assistant";
  text: string;
};

const SUGGESTIONS = [
  "Which Linux components should I update first this week?",
  "Prioritize urgent OpenSSL and kernel updates for production servers.",
  "What software should be patched first for Kubernetes worker nodes?"
];

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [result, setResult] = useState<PrioritizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const signalCounts = useMemo(() => {
    if (!result?.merged) return null;
    return {
      cves: result.merged.cves?.length ?? 0,
      releaseNotes: result.merged.release_notes?.length ?? 0,
      news: result.merged.news?.length ?? 0
    };
  }, [result]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const cleaned = query.trim();
    if (!cleaned || loading) return;

    setError(null);
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: cleaned }]);
    setQuery("");

    try {
      const data = await prioritizeQuery(cleaned);
      setResult(data);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.merged?.llm_summary || "I ranked the updates based on security, release stability, and news signals."
        }
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-4 md:p-8">
        <header className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-glow backdrop-blur">
          <p className="text-xs uppercase tracking-wider text-brand-400">LangGraph + Groq</p>
          <h1 className="mt-1 text-2xl font-semibold md:text-3xl">Software Update Prioritizer</h1>
          <p className="mt-2 text-sm text-slate-300">
            Multi step agent: an LLM plans tool use, tools fetch public data in parallel, a second LLM ranks updates and
            must cite evidence ids, the UI shows those citations next to each urgency score.
          </p>
        </header>

        {result?.merged?.agent_pipeline ? (
          <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-brand-400">This run — agent trace</h2>
            <ol className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
              {result.merged.agent_pipeline.map((step) => (
                <li
                  key={step.id}
                  className="rounded-xl border border-slate-800/80 bg-slate-950/50 p-3"
                >
                  <p className="text-[10px] uppercase text-slate-500">{step.role.replace(/_/g, " ")}</p>
                  <p className="mt-1 text-sm font-medium text-slate-200">{step.label}</p>
                  <p className="mt-1 text-xs leading-snug text-slate-500">{step.detail}</p>
                  <p className="mt-2 font-mono text-[10px] text-slate-600">{step.node}</p>
                </li>
              ))}
            </ol>
          </section>
        ) : null}

        <main className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 lg:col-span-2">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-medium">Chat</h2>
              {loading ? <span className="text-xs text-brand-400">Analyzing...</span> : null}
            </div>

            <div className="mb-4 h-[320px] space-y-3 overflow-y-auto rounded-xl bg-slate-950/70 p-3 md:h-[420px]">
              {messages.length === 0 ? (
                <p className="text-sm text-slate-400">Start by asking which software updates should be prioritized.</p>
              ) : null}
              {messages.map((m, idx) => (
                <div key={idx} className={`max-w-[92%] rounded-xl p-3 text-sm ${m.role === "user" ? "ml-auto bg-brand-600/90 text-white" : "bg-slate-800 text-slate-100"}`}>
                  {m.text}
                </div>
              ))}
            </div>

            <form onSubmit={onSubmit} className="space-y-3">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Example: Prioritize Linux and OpenSSL updates for production servers."
                className="w-full rounded-xl border border-slate-700 bg-slate-950/70 p-3 text-sm outline-none ring-brand-400 placeholder:text-slate-500 focus:ring-2"
                rows={3}
              />
              <div className="flex flex-wrap gap-2">
                {SUGGESTIONS.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setQuery(item)}
                    className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 transition hover:border-brand-400 hover:text-brand-300"
                  >
                    {item}
                  </button>
                ))}
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-xl bg-brand-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Running workflow..." : "Prioritize Updates"}
              </button>
            </form>
            {error ? <p className="mt-3 text-sm text-rose-400">{error}</p> : null}
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-lg font-medium">Ranked output + evidence</h2>
            <p className="mt-1 text-xs text-slate-500">
              Urgency score from the prioritizer LLM, bars show how cited rows split across tool sources (CVE vs release vs
              news). Missing citations are filled by a batched repair LLM, then rule based matching badges show which path
              applied.
            </p>
            {result?.merged?.evidence_provenance_summary ? (
              <p className="mt-2 text-[10px] text-slate-500">
                Evidence refs: prioritizer {result.merged.evidence_provenance_summary.llm ?? 0} · repair{" "}
                {result.merged.evidence_provenance_summary.repair_llm ?? 0} · auto{" "}
                {result.merged.evidence_provenance_summary.deterministic ?? 0} · none{" "}
                {result.merged.evidence_provenance_summary.none ?? 0}
              </p>
            ) : null}
            {!result ? (
              <p className="mt-3 text-sm text-slate-400">Results appear here after the workflow completes.</p>
            ) : (
              <div className="mt-4 space-y-4">
                {result.ranked.map((item) => (
                  <article key={`${item.rank}-${item.software}`} className="rounded-xl border border-slate-700 bg-slate-950/70 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm font-semibold">
                        #{item.rank} {item.software}
                      </p>
                      <div className="flex flex-wrap items-center gap-1">
                        {(() => {
                          const p = evidenceProvenanceLabel(item.evidence_attached_by);
                          return (
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[10px] ${p.className}`}
                              title={p.title}
                            >
                              {p.short}
                            </span>
                          );
                        })()}
                        {item.grounded === false ? (
                          <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-300">
                            Unverified
                          </span>
                        ) : null}
                        <span
                          className="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-300"
                          title="Prioritizer LLM urgency score (0–100)"
                        >
                          Urgency {item.priority_score}/100
                        </span>
                      </div>
                    </div>
                    <SignalMixBar sb={item.signal_breakdown} />
                    <p className="mt-2 text-xs text-slate-300">{item.suggested_action}</p>
                    <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-slate-400">
                      {item.reasons.map((reason, i) => (
                        <li key={i}>{reason}</li>
                      ))}
                    </ul>
                    {item.grounding_warnings && item.grounding_warnings.length > 0 ? (
                      <p className="mt-2 text-xs text-amber-400/90">
                        Checks: {item.grounding_warnings.join("; ")}
                      </p>
                    ) : null}
                    {item.evidence && item.evidence.length > 0 ? (
                      <details open className="mt-3 rounded-lg border border-slate-800 bg-slate-900/40 p-2">
                        <summary className="cursor-pointer text-xs font-medium text-slate-300">
                          Evidence cited ({item.evidence.length} rows from tools) — news rows include outlet + article link
                        </summary>
                        <div className="mt-2 space-y-2">
                          {item.evidence.map((ev) => (
                            <EvidenceCard key={ev.ref} ev={ev} />
                          ))}
                        </div>
                      </details>
                    ) : null}
                  </article>
                ))}
              </div>
            )}
          </section>
        </main>


{result ? (
          <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs uppercase text-slate-400">CVE Signals</p>
              <p className="mt-2 text-2xl font-semibold">{signalCounts?.cves ?? 0}</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs uppercase text-slate-400">Release Notes</p>
              <p className="mt-2 text-2xl font-semibold">{signalCounts?.releaseNotes ?? 0}</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs uppercase text-slate-400">News Mentions</p>
              <p className="mt-2 text-2xl font-semibold">{signalCounts?.news ?? 0}</p>
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}

export default App;
