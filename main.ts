// Deterministic, history-only GTA trophy rarity forecaster (Node 22, stdlib only).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const clamp = (x: number, lo = 0.01, hi = 99.99): number => Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => { const p = clamp(pct); return Math.log(p / (100 - p)); };
const logistic = (x: number): number => 100 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
const round4 = (x: number): number => Math.round(Math.max(0, Math.min(100, x)) * 10_000) / 10_000;

interface Trophy { id: number; tier?: string; points?: number; secret?: boolean; }
interface Read { at: string; players_tracked: number; trophies_earned: number; rates: Array<number | null>; }
interface Case { levels: number[]; horizon_minutes: number; trophies: Trophy[]; history: Read[]; }
interface Observation { t: number; y: number; raw: number; players: number; earned: number; }
const DAY = 86_400_000;

function elapsedDays(a: string, b: string): number {
  const x = Date.parse(a), y = Date.parse(b);
  return Number.isFinite(x) && Number.isFinite(y) ? Math.max(0, (y - x) / DAY) : 0;
}
function weightedMedian(xs: Array<{ value: number; weight: number }>): number {
  const a = xs.filter((x) => Number.isFinite(x.value) && x.weight > 0).sort((x, y) => x.value - y.value);
  const total = a.reduce((s, x) => s + x.weight, 0);
  let n = 0;
  for (const x of a) { n += x.weight; if (n >= total / 2) return x.value; }
  return a.length ? a[a.length - 1].value : 0;
}

function prior(t: Trophy): { speed: number; secret: number; floor: number; ceiling: number } {
  const tier = (t.tier ?? '').toLowerCase();
  const points = Number.isFinite(t.points) ? t.points! : 20;
  const speed = tier.includes('bronze') ? 1.2 : tier.includes('platinum') || tier.includes('gold') ? 0.66 : tier.includes('silver') ? 0.9 : points <= 15 ? 1.15 : points >= 75 ? 0.68 : 0.86;
  return { speed, secret: t.secret ? 0.72 : 1, floor: t.secret ? 0.00012 : 0.00038, ceiling: t.secret ? 0.000025 : 0.00005 };
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((v, i) => v !== LEVELS[i])) throw new Error(`Expected levels ${JSON.stringify(LEVELS)}`);
  if (!c.history.length) throw new Error('Case has no history');
  const horizon = Math.max(0, c.horizon_minutes / 1440);
  const elapsed = c.history.map((r, i) => i ? elapsedDays(c.history[0].at, r.at) : 0);
  const latest = elapsed[elapsed.length - 1];
  const series: Observation[][] = c.trophies.map((_, i) => c.history.flatMap((r, j) => {
    const raw = r.rates[i];
    return raw == null || !Number.isFinite(raw) ? [] : [{ t: elapsed[j], y: logit(raw), raw: clamp(raw), players: Math.max(1, r.players_tracked || 1), earned: Math.max(0, r.trophies_earned || 0) }];
  }));

  // Median logit trend across trophies is the case's launch-curve shape.
  const popSteps: Array<{ value: number; weight: number }> = [];
  for (let j = 1; j < c.history.length; j++) {
    const dt = elapsed[j] - elapsed[j - 1];
    if (dt <= 0) continue;
    const changes: number[] = [];
    for (const s of series) {
      const a = s.find((o) => o.t === elapsed[j - 1]), b = s.find((o) => o.t === elapsed[j]);
      if (a && b) changes.push((b.y - a.y) / dt);
    }
    if (changes.length) {
      changes.sort((a, b) => a - b);
      popSteps.push({ value: changes[Math.floor(changes.length / 2)], weight: Math.exp(-(latest - elapsed[j]) / 35) * Math.sqrt(changes.length) });
    }
  }
  const populationSlope = weightedMedian(popSteps);
  const growths: Array<{ value: number; weight: number }> = [];
  for (let j = 1; j < c.history.length; j++) {
    const dt = elapsed[j] - elapsed[j - 1], a = c.history[j - 1], b = c.history[j];
    if (dt > 0 && a.players_tracked > 0 && b.players_tracked > 0)
      growths.push({ value: Math.max(-0.02, Math.min(0.1, Math.log(b.players_tracked / a.players_tracked) / dt)), weight: Math.exp(-(latest - elapsed[j]) / 35) });
  }
  const playerGrowth = weightedMedian(growths);
  const last = c.history[c.history.length - 1];

  return c.trophies.map((t, i) => {
    const obs = series[i], raw = last.rates[i];
    const current = raw != null && Number.isFinite(raw) ? clamp(raw) : obs[obs.length - 1]?.raw ?? 0.01;
    let slope = 0, residual = 0.5, reliability = 0;
    if (obs.length >= 2) {
      const weights = obs.map((o) => Math.exp(-(latest - o.t) / 35));
      const sw = weights.reduce((a, b) => a + b, 0);
      const mx = obs.reduce((s, o, j) => s + weights[j] * o.t, 0) / sw;
      const my = obs.reduce((s, o, j) => s + weights[j] * o.y, 0) / sw;
      const sxx = obs.reduce((s, o, j) => s + weights[j] * (o.t - mx) ** 2, 0);
      const sxy = obs.reduce((s, o, j) => s + weights[j] * (o.t - mx) * (o.y - my), 0);
      const rawSlope = sxx > 1e-8 ? sxy / sxx : 0;
      const increments: Array<{ value: number; weight: number }> = [];
      for (let j = 1; j < obs.length; j++) {
        const dt = obs[j].t - obs[j - 1].t;
        if (dt > 0) increments.push({ value: (obs[j].y - obs[j - 1].y - rawSlope * dt) / Math.sqrt(dt), weight: Math.exp(-(latest - obs[j].t) / 35) });
      }
      if (increments.length) {
        const iw = increments.reduce((s, x) => s + x.weight, 0);
        residual = Math.sqrt(increments.reduce((s, x) => s + x.weight * x.value ** 2, 0) / iw);
      }
      const span = Math.max(0, obs[obs.length - 1].t - obs[0].t);
      reliability = Math.min(0.88, (obs.length - 1) / (obs.length + 2) * span / (span + 7)) / (1 + (residual / 0.8) ** 2);
      slope = rawSlope * reliability;
    }

    // Correct observed changes for the lag between new tracked players and earned trophies.
    const lagSamples: Array<{ value: number; weight: number }> = [];
    for (let j = 1; j < obs.length; j++) {
      const growth = Math.log(obs[j].players / obs[j - 1].players);
      const dt = obs[j].t - obs[j - 1].t;
      if (growth > 0.002 && dt > 0) lagSamples.push({ value: Math.max(0, Math.min(1, -(obs[j].y - obs[j - 1].y) / growth)), weight: Math.exp(-(latest - obs[j].t) / 35) });
    }
    const lag = lagSamples.length ? Math.max(0.12, Math.min(0.65, 0.2 + 0.25 * weightedMedian(lagSamples))) : 0.2;
    // The historical slope contains dilution already. Remove that component
    // before extrapolating the underlying curve, then apply expected future
    // tracker growth once below.
    const underlyingSlope = slope + reliability * lag * Math.max(0, playerGrowth);
    const dilution = -lag * Math.max(0, playerGrowth);
    const p = prior(t);
    const priorSpeed = p.speed * p.secret;
    const effectiveSpeed = 1 + (priorSpeed - 1) * (1 - reliability);
    const population = populationSlope * priorSpeed;
    const blended = underlyingSlope * effectiveSpeed + (population - underlyingSlope * effectiveSpeed) * (1 - reliability) * 0.2;
    const floor = Math.max(0, 1.1 - current) * p.floor;
    const ceiling = Math.max(0, current - 85) * p.ceiling;
    const drift = blended + dilution * (1 - reliability) + floor + ceiling;
    const median = logit(current) + drift * horizon;
    const sigma = Math.max(0.12, Math.min(0.8, 0.08 + residual * 0.55 + (obs.length < 3 ? 0.12 : 0)));
    // The fitted increment variance includes level-fit uncertainty; discount
    // that component before turning it into a future process spread.
    const spread = 0.72 * Math.sqrt(sigma * sigma * horizon + 0.0144 * horizon + 0.04);
    const q = Z.map((z) => round4(logistic(median + z * spread)));
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
  const id = file.slice(0, -3);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: forecast(c) })}\n`);
}
