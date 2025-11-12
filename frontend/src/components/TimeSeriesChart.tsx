import { useEffect, useRef } from "react";
import {
  Chart, LineController, LineElement, PointElement,
  LinearScale, TimeScale, Tooltip, Legend, CategoryScale, Filler
} from "chart.js";
Chart.register(LineController, LineElement, PointElement, LinearScale,
               TimeScale, Tooltip, Legend, CategoryScale, Filler);

type Pt = { ts: string; close?: number; value?: number };
type Fc = { ts: string; yhat: number; q10?: number; q50?: number; q90?: number };

export default function TimeSeriesChart({
  series, forecast = [], showBands = true, overlays = []
}: { series: Pt[]; forecast?: Fc[]; showBands?: boolean; overlays?: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const ctx = ref.current.getContext("2d"); if (!ctx) return;

    const yOf = (p: Pt) => p.close ?? p.value ?? null;
    const actual = series.map(p => ({ x: p.ts, y: yOf(p) }));
    const fc = forecast.map(p => ({ x: p.ts, y: p.yhat }));
    const q90 = forecast.map(p => ({ x: p.ts, y: p.q90 ?? null }));
    const q10 = forecast.map(p => ({ x: p.ts, y: p.q10 ?? null }));

    const ma = (win: number) => {
      const out: any[] = [];
      const vals = series.map(yOf);
      for (let i = 0; i < vals.length; i++) {
        const slice = vals.slice(Math.max(0, i - win + 1), i + 1).filter(v => v != null) as number[];
        out.push({ x: series[i].ts, y: slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null });
      }
      return out;
    };

    const datasets: any[] = [
      { label: "Actual", data: actual, borderWidth: 2, tension: 0.2 },
      { label: "Forecast", data: fc, borderDash: [6, 6], borderWidth: 2, tension: 0.2 }
    ];
    if (showBands) {
      datasets.push({ label: "q90", data: q90, borderWidth: 0, pointRadius: 0 });
      datasets.push({ label: "q10", data: q10, borderWidth: 0, pointRadius: 0, fill: "-1" });
    }
    overlays.forEach(w =>
      datasets.push({ label: `MA${w}`, data: ma(w), borderWidth: 1, tension: 0.1 })
    );

    const chart = new Chart(ctx, {
      type: "line",
      data: { datasets },
      options: {
        parsing: false,
        responsive: true,
        scales: { x: { type: "category", ticks: { maxRotation: 0 } }, y: { beginAtZero: false } },
        plugins: { legend: { position: "bottom" } }
      }
    });
    return () => chart.destroy();
  }, [series, forecast, showBands, overlays]);

  return <canvas ref={ref} height={300} />;
}
