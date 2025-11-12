const API = import.meta.env.VITE_API || "http://localhost:8000/v1";
const KEY = import.meta.env.VITE_API_KEY || "dev-key";

async function fetchJSON(url: string, opts: RequestInit = {}) {
  const r = await fetch(url, {
    ...opts,
    headers: { "x-api-key": KEY, ...(opts.headers || {}) }
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const getSources = () => fetchJSON(`${API}/sources`);
export const getSymbols = (source: string) => fetchJSON(`${API}/symbols/${source}`);
export const getSeries = (s: string, sym: string, start?: string, end?: string) => {
  const q = new URLSearchParams();
  if (start) q.set("start", start);
  if (end) q.set("end", end);
  return fetchJSON(`${API}/timeseries/${s}/${sym}?${q}`);
};
export const getForecast = (s: string, sym: string, h = 10) =>
  fetchJSON(`${API}/predict/${s}/${sym}?horizon=${h}`);
export const getSignals = (s: string, sym: string) =>
  fetchJSON(`${API}/signals?source=${s}&symbol=${sym}`);
export const listWatchlists = () => fetchJSON(`${API}/watchlists`);
export const createWatchlist = (name: string, symbols: string[]) =>
  fetchJSON(`${API}/watchlists`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, symbols })
  });
