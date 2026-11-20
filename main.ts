import { readdir, readFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// Run directly with Node 22. No packages, training files, or random state.
type Trophy = { id: number; tier?: string; secret?: boolean };
type Snapshot = {
  slot?: number; at: string; players_tracked: number; rates: (number | null)[];
};
type ForecastCase = {
  schema: string; id: unknown; at: string; horizon_minutes: number;
  target?: string; trophies: Trophy[]; history: Snapshot[];
};
type Point = { t: number; y: number; weight: number; noise: number };
type Line = {
  slope: number; intercept: number; residualVariance: number;
  slopeVariance: number; span: number; count: number; cadence: number;
};
type Curve = { last: number; fit: Line; samplingVariance: number };

const DAY = 86_400_000;
const EPSILON = 1e-8;
const NORMAL_QUANTILES = [-1.64485362695147, -0.674489750196082, 0,
  0.674489750196082, 1.64485362695147];
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const logit = (p: number) => Math.log(clamp(p, EPSILON, 1 - EPSILON) /
  (1 - clamp(p, EPSILON, 1 - EPSILON)));
const logistic = (z: number) => z >= 0 ? 1 / (1 + Math.exp(-z)) :
  Math.exp(z) / (1 + Math.exp(z));
const median = (values: number[]) => {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
};

// Weighted robust regression in days, centered on the most recent observation.
function fitLine(points: Point[]): Line {
  if (points.length < 2) {
    return { slope: 0, intercept: points[0]?.y ?? 0, residualVariance: 0,
      slopeVariance: 0.25, span: 0, count: points.length, cadence: 1 };
  }
  const weights = points.map(p => p.weight);
  let slope = 0;
  let intercept = 0;
  let spread = 0;
  let sumWeight = 0;
  for (let pass = 0; pass < 3; pass++) {
    sumWeight = weights.reduce((sum, w) => sum + w, 0);
    const meanT = points.reduce((sum, p, i) => sum + weights[i] * p.t, 0) / sumWeight;
    const meanY = points.reduce((sum, p, i) => sum + weights[i] * p.y, 0) / sumWeight;
    spread = points.reduce((sum, p, i) => sum + weights[i] * (p.t - meanT) ** 2, 0);
    slope = spread > 1e-12 ? points.reduce((sum, p, i) =>
      sum + weights[i] * (p.t - meanT) * (p.y - meanY), 0) / spread : 0;
    intercept = meanY - slope * meanT;
    const errors = points.map(p => Math.abs(p.y - intercept - slope * p.t));
    const scale = Math.max(0.015, median(errors) * 1.4826);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = points[i].weight * Math.min(1, 1.5 * scale / Math.max(errors[i], 1e-12));
    }
  }
  const residualVariance = points.reduce((sum, p) =>
    sum + p.weight * (p.y - intercept - slope * p.t) ** 2, 0) /
    points.reduce((sum, p) => sum + p.weight, 0) * points.length / Math.max(1, points.length - 2);
  const sampling = points.reduce((sum, p) => sum + p.noise, 0) / points.length;
  const span = points[points.length - 1].t - points[0].t;
  const intervals = points.slice(1).map((p, i) => p.t - points[i].t).filter(x => x > 0);
  return { slope, intercept, residualVariance,
    slopeVariance: Math.max(residualVariance, sampling, 0.0001) /
      Math.max(0.001, spread / Math.max(sumWeight, 1e-12) * points.length),
    span, count: points.length, cadence: Math.max(1 / 1440, median(intervals) || 1) };
}

function driftPrior(trophy: Trophy): number {
  const tier = String(trophy.tier ?? '').toLowerCase();
  const base = ({ bronze: 0.18, silver: 0.12, gold: 0.075, platinum: 0.035 } as
    Record<string, number>)[tier] ?? 0.12;
  return base * (trophy.secret ? 0.82 : 1);
}

function validate(input: ForecastCase): void {
  if (input?.schema !== 'makefun/gta6-trophies-case/v1') throw new Error('Unsupported case schema');
  if (!Array.isArray(input.trophies) || !Array.isArray(input.history)) {
    throw new Error('Expected trophies and history arrays');
  }
  if (!Number.isFinite(input.horizon_minutes) || input.horizon_minutes < 0 ||
      !Number.isFinite(Date.parse(input.at))) throw new Error('Invalid forecast time or horizon');
  const ids = new Set<number>();
  for (const trophy of input.trophies) {
    if (!Number.isSafeInteger(trophy.id) || ids.has(trophy.id)) throw new Error('Invalid or duplicate trophy id');
    ids.add(trophy.id);
  }
  for (const row of input.history) {
    if (!Number.isFinite(Date.parse(row.at)) || !Number.isFinite(row.players_tracked) ||
        row.players_tracked <= 0 || !Array.isArray(row.rates)) throw new Error('Invalid history row');
  }
}

function forecast(input: ForecastCase) {
  validate(input);
  const origin = Date.parse(input.at);
  const horizon = input.horizon_minutes / 1440;
  // Ignore future observations and collapse duplicate timestamps deterministically.
  const rowsByTime = new Map<number, Snapshot>();
  for (const row of input.history) {
    const stamp = Date.parse(row.at);
    if (stamp <= origin) rowsByTime.set(stamp, row);
  }
  const rows = [...rowsByTime.entries()].sort((a, b) => a[0] - b[0])
    .map(([stamp, row]) => ({ ...row, t: (stamp - origin) / DAY }));
  const span = rows.length > 1 ? rows[rows.length - 1].t - rows[0].t : 0;
  const decayWindow = Math.max(1, span * 0.65);
  const lastTime = rows[rows.length - 1]?.t ?? 0;
  const weightAt = (t: number) => Math.exp((t - lastTime) / decayWindow);
  const populationSizes = rows.map(row => ({ t: row.t - lastTime,
    y: Math.log(row.players_tracked), weight: weightAt(row.t), noise: 0 }));
  const growthFit = fitLine(populationSizes);
  const recentGrowthFit = fitLine(populationSizes.slice(-Math.max(3, Math.ceil(rows.length / 2))));
  const growth = clamp(0.4 * growthFit.slope + 0.6 * recentGrowthFit.slope, -0.5, 2);

  function curveFor(rateAt: (row: typeof rows[number]) => number | null): Curve {
    const observations: { t: number; p: number; n: number; z: number; noise: number }[] = [];
    for (const row of rows) {
      const rate = rateAt(row);
      if (typeof rate !== 'number' || !Number.isFinite(rate) || rate < 0 || rate > 100) continue;
      // Jeffreys smoothing keeps observed 0%/100% finite on the logit scale.
      const p = (row.players_tracked * rate / 100 + 0.5) / (row.players_tracked + 1);
      observations.push({ t: row.t, p, n: row.players_tracked, z: logit(p),
        noise: 1 / Math.max(0.25, row.players_tracked * p * (1 - p)) });
    }
    if (!observations.length) return { last: lastTime,
      fit: fitLine([]), samplingVariance: 0.5 };
    const latest = observations[observations.length - 1];
    let dilutionOffset = 0;
    const adjusted = observations.map((o, i) => {
      if (i > 0) {
        const previous = observations[i - 1];
        // d(logit p)/dt = intrinsic drift - d(log N)/dt / (1-p).
        dilutionOffset += Math.log(o.n / previous.n) *
          0.5 * (1 / Math.max(0.025, 1 - previous.p) + 1 / Math.max(0.025, 1 - o.p));
      }
      return { t: o.t - latest.t, y: o.z + dilutionOffset,
        weight: weightAt(o.t), noise: o.noise };
    });
    const fit = fitLine(adjusted);
    // Fit the drift, but anchor forecasts to the latest actual observed rarity.
    fit.intercept = latest.z;
    return { last: latest.t, fit, samplingVariance: latest.noise };
  }

  const population = curveFor(row => {
    const observed = input.trophies.map((_, i) => row.rates[i])
      .filter((r): r is number => typeof r === 'number' && Number.isFinite(r) && r >= 0 && r <= 100);
    return observed.length ? median(observed) : null;
  });
  const typicalPrior = median(input.trophies.map(driftPrior)) || 0.12;
  const populationStrength = population.fit.count / (population.fit.count + 2) *
    population.fit.span / (population.fit.span + 0.5);
  const populationDrift = clamp(populationStrength * population.fit.slope +
    (1 - populationStrength) * typicalPrior, -2, 4);
  const trendMemory = Math.max(5, span * 2);
  const growthMemory = Math.max(7, span * 2);

  // Exact logistic and dilution substeps avoid unstable logits near 100%.
  function advance(z: number, drift: number, dilution: number, dt: number): number {
    const halfDiluted = clamp(logistic(z) * Math.exp(-dilution * dt / 2), EPSILON, 1 - EPSILON);
    const acquired = logistic(logit(halfDiluted) + drift * dt);
    return logit(clamp(acquired * Math.exp(-dilution * dt / 2), EPSILON, 1 - EPSILON));
  }

  return { trophies: input.trophies.map((trophy, index) => {
    const curve = curveFor(row => row.rates[index] ?? null);
    const fit = curve.fit;
    const prior = 0.7 * populationDrift + 0.3 * driftPrior(trophy);
    const information = fit.count / (fit.count + 1.5) * fit.span / (fit.span + 0.25) /
      (1 + fit.slopeVariance / (0.3 ** 2));
    const intrinsicDrift = clamp(information * fit.slope + (1 - information) * prior, -2, 4);
    const elapsed = Math.max(0, horizon - curve.last);
    const steps = Math.max(1, Math.min(2048, Math.ceil(elapsed * 48)));
    const dt = elapsed / steps;
    let z = fit.count ? fit.intercept : logit(0.01);
    let center = population.fit.count ? population.fit.intercept : logit(0.1);
    // Align the population reference with a trophy whose last valid rate is older.
    if (curve.last < population.last) {
      center -= population.fit.slope * (population.last - curve.last);
    }
    for (let step = 0; step < steps; step++) {
      const age = (step + 0.5) * dt;
      const futureAge = Math.max(0, curve.last + age);
      const dilution = growth * Math.exp(-futureAge / growthMemory);
      const fadingDrift = prior + (intrinsicDrift - prior) * Math.exp(-age / trendMemory);
      const gap = center - z;
      const reversion = 0.035 * gap / (1 + Math.abs(gap));
      z = advance(z, fadingDrift + reversion, dilution, dt);
      center = advance(center, populationDrift, dilution, dt);
    }
    const effectiveHorizon = trendMemory * (-Math.expm1(-elapsed / trendMemory));
    const residual = Math.sqrt(fit.residualVariance);
    const driftUncertainty = Math.sqrt(fit.slopeVariance + (1 - information) * 0.08 ** 2);
    const processVariance = (0.04 ** 2 + 0.2 * fit.residualVariance) * elapsed;
    const uncertainty = Math.sqrt(
      curve.samplingVariance + fit.residualVariance +
      (driftUncertainty * effectiveHorizon) ** 2 + processVariance +
      (residual * Math.sqrt(elapsed / Math.max(0.25, fit.cadence)) * 0.3) ** 2 +
      (0.08 * Math.log1p(elapsed) * (1 + Math.abs(growth))) ** 2
    );
    const q = NORMAL_QUANTILES.map(normal => {
      const percent = 100 * logistic(z + normal * uncertainty);
      return Math.round(clamp(percent, 0, 100) * 1e6) / 1e6;
    });
    return { id: trophy.id, q };
  }) };
}

async function main(): Promise<void> {
  const inputDirectory = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDirectory = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  const entries = (await readdir(inputDirectory, { withFileTypes: true }))
    .filter(entry => entry.isFile() && entry.name.endsWith('.in'))
    .sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  await mkdir(outputDirectory, { recursive: true });
  for (const entry of entries) {
    try {
      const input = JSON.parse(await readFile(join(inputDirectory, entry.name), 'utf8')) as ForecastCase;
      const result = forecast(input);
      const serialized = JSON.stringify(result) + '\n';
      if (Buffer.byteLength(serialized, 'utf8') > 65536) throw new Error('Output exceeds 65536 bytes');
      for (const row of result.trophies) {
        if (row.q.length !== 5 || row.q.some((value, i) => !Number.isFinite(value) ||
            value < 0 || value > 100 || (i > 0 && value < row.q[i - 1]))) {
          throw new Error('Invalid quantile output');
        }
      }
      await writeFile(join(outputDirectory, `${entry.name.slice(0, -3)}.out`), serialized, 'utf8');
    } catch (error) {
      process.stderr.write(`${entry.name}: ${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    }
  }
}

main().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
