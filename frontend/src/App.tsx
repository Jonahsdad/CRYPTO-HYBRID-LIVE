import { useEffect, useMemo, useState } from "react";
import {
  getSources, getSymbols, getSeries, getForecast,
  listWatchlists, createWatchlist
} from "./api";
import TimeSeriesChart from "./components/TimeSeriesChart";

type Source = { id: string; kind: string; name: string };
type Symbol = { source: string; symbol: string; name?: string };

export default function App() {
  const [sources, setSources] = useState<Source[]>([]);
  const [source, setSource] = useState("");
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [symbol, setSymbol] = useState("");
  const [series, setSeries] = useState<any[]>([]);
  const [forecast, setForecast] = useState<any[]>([]);
  const [bands, setBands] = useState(true);
  const [overlays, setOverlays] = useState<number[]>([5, 20]);
  const [watchlists, setWatchlists] = useState<any[]>([]);
  const [wlName, setWlName] = useState("");

  useEffect(() => { getSources().then(s => { setSources(s); if (s[0]) setSource(s[0].id); }); }, []);
  useEffect(() => { if (!source) return; getSymbols(source).then(ss => { setSymbols(ss); if (ss[0]) setSymbol(ss[0].symbol); setSeries([]); setForecast([]); }); }, [source]);
  useEffect(() => { if (!source || !symbol) return; getSeries(source, symbol).then(r => setSeries(r.points)); getForecast(source, symbol, 10).then(r => setForecast(r.forecast)); }, [source, symbol]);
  useEffect(() => { listWatchlists().then(setWatchlists).catch(() => {}); }, []);

  const title = useMemo(() => {
    const s = sources.find(x => x.id === source)?.name ?? source;
    return `${s} · ${symbol || ""}`;
  }, [sources, source, symbol]);

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, sans-serif", maxWidth: 1100, margin: "0 auto" }}>
      <header style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <h2 style={{ marginRight: "auto" }}>{title}</h2>
        <label><input type="checkbox" checked={bands} onChange={e => setBands(e.target.checked)} /> Bands</label>
      </header>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <select value={source} onChange={e => setSource(e.target.value)}>
          {sources.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        <select value={symbol} onChange={e => setSymbol(e.target.value)}>
          {symbols.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol}</option>)}
        </select>
        <span>Overlays:</span>
        {[5, 20].map(w => (
          <label key={w} style={{ marginRight: 8 }}>
            <input type="checkbox" checked={overlays.includes(w)}
              onChange={(e) => setOverlays(prev => e.target.checked ? [...prev, w] : prev.filter(x => x !== w))} />
            MA{w}
          </label>
        ))}
      </div>

      <TimeSeriesChart series={series} forecast={forecast} showBands={bands} overlays={overlays} />

      <section style={{ marginTop: 20 }}>
        <h3>Watchlists</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <input placeholder="New watchlist name" value={wlName} onChange={e => setWlName(e.target.value)} />
          <button onClick={async () => {
            if (!wlName) return;
            const wl = await createWatchlist(wlName, [symbol].filter(Boolean));
            setWatchlists([...watchlists, wl]); setWlName("");
          }}>Save Current</button>
        </div>
        <ul>{watchlists.map(wl => <li key={wl.id}>{wl.name}: {wl.symbols.join(", ")}</li>)}</ul>
      </section>
    </div>
  );
}
