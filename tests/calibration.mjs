import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const root = resolve(import.meta.dirname, '..');
const work = mkdtempSync(join(tmpdir(), 'trophy-calibration-'));
const input = join(work, 'input'), output = join(work, 'output');
mkdirSync(input); mkdirSync(output);
let seed = 0x71a5;
function random() { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; }
function normal() { return Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random()); }
const logistic = (x) => 100 / (1 + Math.exp(-x));
const logit = (p) => Math.log(p / (100 - p));
const rates = [1, 5, 15, 35, 60, 82, 95];
const levels = [0.05, 0.25, 0.5, 0.75, 0.95];
const counts = Array(5).fill(0), total = 400 * rates.length, medianErrors = [];

try {
  for (let k = 0; k < 400; k++) {
    const trophies = rates.map((r, i) => ({ id: i + 1, tier: i < 2 ? 'bronze' : i > 4 ? 'gold' : 'silver', points: i < 2 ? 15 : 50, secret: i === 3 }));
    const history = [];
    let earned = 0;
    for (let day = 0; day <= 28; day += 7) {
      const players = Math.round(10000 * Math.exp(0.012 * day));
      earned += 1200;
      history.push({ at: new Date(Date.UTC(2025, 0, 1 + day)).toISOString(), players_tracked: players, trophies_earned: earned,
        rates: rates.map((r, i) => {
          const drift = (i < 2 ? 0.012 : i > 4 ? 0.004 : 0.008) - 0.0024 + (r < 2 ? 0.001 : 0);
          return Math.max(0.01, Math.min(99.99, logistic(logit(r) + drift * day + 0.12 * Math.sqrt(day) * normal())));
        }) });
    }
    const horizon = 14 + Math.floor(random() * 60);
    const truth = rates.map((r, i) => {
      const drift = (i < 2 ? 0.012 : i > 4 ? 0.004 : 0.008) - 0.0024 + (r < 2 ? 0.001 : 0);
      return logistic(logit(history.at(-1).rates[i]) + drift * horizon + 0.12 * Math.sqrt(horizon) * normal());
    });
    const c = { levels, horizon_minutes: horizon * 1440, trophies, history };
    writeFileSync(join(input, 'case.in'), JSON.stringify(c));
    const run = spawnSync(process.execPath, ['--experimental-strip-types', join(root, 'main.ts')], { env: { ...process.env, MAKEFUN_INPUT_DIR: input, MAKEFUN_OUTPUT_DIR: output }, encoding: 'utf8' });
    if (run.status !== 0) throw new Error(run.stderr);
    const forecast = JSON.parse(readFileSync(join(output, 'case.out'), 'utf8')).trophies;
    forecast.forEach((row, i) => medianErrors.push(logit(row.q[2]) - logit(truth[i])));
    forecast.forEach((row, i) => row.q.forEach((q, j) => { if (truth[i] <= q) counts[j]++; }));
  }
  const coverage = counts.map((n) => n / total);
  medianErrors.sort((a, b) => a - b);
  console.log(JSON.stringify({ cases: 400, quantiles: levels, empiricalCoverage: coverage, medianLogitError: medianErrors[Math.floor(medianErrors.length / 2)] }));
  for (let i = 0; i < levels.length; i++) if (Math.abs(coverage[i] - levels[i]) > 0.11) process.exitCode = 1;
} finally {
  rmSync(work, { recursive: true, force: true });
}
