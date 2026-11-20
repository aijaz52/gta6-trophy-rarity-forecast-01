import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DAY = 86_400_000;
const epoch = Date.parse('2026-10-01T00:00:00Z');
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const logistic = z => 1 / (1 + Math.exp(-z));
const logit = p => Math.log(p / (1 - p));
const stamp = day => new Date(epoch + day * DAY).toISOString();
const tiers = ['bronze', 'silver', 'gold', 'platinum'];
const trophies = Array.from({ length: 16 }, (_, i) => ({
  id: 101 + i * 3,
  name: `Synthetic trophy ${i + 1}`,
  description: 'A simulated achievement with a known underlying population rate.',
  tier: tiers[i % tiers.length],
  points: [15, 30, 90, 180][i % 4],
  secret: i % 3 === 0,
}));

// Ground truth uses a continuous population ODE, integrated independently by RK4.
// New players start without the trophy, producing dp/dt = a*p*(1-p) - g*p.
function simulate({ initial, acquisition, growth, duration, interval = 0.25 }) {
  let state = [...initial];
  let players = 120_000;
  const samples = [];
  const dt = 1 / 960;
  const steps = Math.round(duration / dt);
  const stride = Math.round(interval / dt);
  const derivative = (time, values) => values.map((p, i) =>
    acquisition(time, i, values) * p * (1 - p) - growth(time) * p);
  for (let step = 0; step <= steps; step++) {
    const time = step * dt;
    if (step % stride === 0 || step === steps) {
      samples.push({ time, players, rates: state.map(p => 100 * p) });
    }
    if (step === steps) break;
    const k1 = derivative(time, state);
    const k2 = derivative(time + dt / 2, state.map((p, i) => p + dt * k1[i] / 2));
    const k3 = derivative(time + dt / 2, state.map((p, i) => p + dt * k2[i] / 2));
    const k4 = derivative(time + dt, state.map((p, i) => p + dt * k3[i]));
    state = state.map((p, i) => clamp(p + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6,
      1e-9, 1 - 1e-9));
    players *= Math.exp(growth(time + dt / 2) * dt);
  }
  return samples;
}

function buildCase(id, samples, observedDays, horizonDays, noise = 0) {
  const observed = samples.filter(sample => sample.time <= observedDays + 1e-8);
  const truth = samples.find(sample => Math.abs(sample.time - observedDays - horizonDays) < 1e-8);
  assert.ok(truth, `${id}: missing simulated future`);
  const input = {
    schema: 'makefun/gta6-trophies-case/v1', id,
    at: stamp(observedDays), horizon_minutes: horizonDays * 1440,
    target: stamp(observedDays + horizonDays), levels: [0.05, 0.25, 0.5, 0.75, 0.95],
    game: 'Synthetic GTA VI', trophies,
    history: observed.map((sample, slot) => ({
      slot, at: stamp(sample.time), players_tracked: Math.round(sample.players),
      trophies_earned: sample.rates.reduce((total, rate) =>
        total + Math.round(sample.players * rate / 100), 0),
      rates: sample.rates.map((rate, i) => {
        const perturbation = noise * (0.7 * Math.sin(slot * 1.9 + i * 0.8) +
          0.3 * Math.cos(slot * 0.73 - i * 1.1));
        return Number((100 * logistic(logit(rate / 100) + perturbation)).toFixed(6));
      }),
    })),
    notes: 'Deterministic synthetic coverage test; truth is withheld from the forecaster.',
  };
  return { input, truth: truth.rates };
}

const initial = trophies.map((_, i) => 0.025 + i * 0.031);
const logisticCase = buildCase('logistic', simulate({
  initial, acquisition: (_, i) => 0.11 + i * 0.011, growth: () => 0,
  duration: 7,
}), 6, 1, 0.012);

const dilutionCase = buildCase('dilution', simulate({
  initial: initial.map(p => p + 0.14),
  acquisition: (_, i) => 0.055 + i * 0.0025,
  growth: () => 0.16,
  duration: 7.5,
}), 6, 1.5, 0.015);

const reversionSamples = simulate({
  initial: initial.map(p => p * 0.7),
  acquisition: (time, i, values) => {
    const ordered = [...values].sort((a, b) => a - b);
    const center = (ordered[7] + ordered[8]) / 2;
    return 0.14 + (0.19 * Math.sin(i * 1.6)) * Math.exp(-time / 6) +
      0.025 * (logit(center) - logit(values[i]));
  },
  growth: time => 0.04 * Math.exp(-time / 12),
  duration: 9,
});
const reversionCase = buildCase('reversion', reversionSamples, 6, 3, 0.10);

const cases = [logisticCase, dilutionCase, reversionCase];
const workspace = await mkdtemp(join(tmpdir(), 'gta6-synthetic-'));
const inputDirectory = join(workspace, 'input');
const outputDirectory = join(workspace, 'output');
await mkdir(inputDirectory);

function execute(destination) {
  const run = spawnSync(process.execPath, [join(root, 'main.ts')], {
    cwd: root,
    env: { ...process.env, MAKEFUN_INPUT_DIR: inputDirectory, MAKEFUN_OUTPUT_DIR: destination },
    encoding: 'utf8', timeout: 30_000,
  });
  assert.ifError(run.error);
  assert.equal(run.status, 0, `Forecaster failed:\n${run.stderr}`);
}

function checkShape(output, input, bytes) {
  assert.ok(bytes <= 65536, 'Output exceeds byte limit');
  assert.deepEqual(Object.keys(output), ['trophies']);
  assert.equal(output.trophies.length, input.trophies.length);
  assert.deepEqual(output.trophies.map(row => row.id), input.trophies.map(row => row.id));
  for (const row of output.trophies) {
    assert.deepEqual(Object.keys(row).sort(), ['id', 'q']);
    assert.equal(row.q.length, 5);
    row.q.forEach((value, i) => {
      assert.equal(typeof value, 'number');
      assert.ok(Number.isFinite(value) && value >= 0 && value <= 100);
      if (i > 0) assert.ok(value >= row.q[i - 1], 'Unordered quantiles');
    });
  }
}

try {
  for (const test of cases) {
    await writeFile(join(inputDirectory, `${test.input.id}.in`), JSON.stringify(test.input));
  }
  await writeFile(join(inputDirectory, 'ignored.txt'), 'Not a forecast case');
  execute(outputDirectory);
  assert.deepEqual((await readdir(outputDirectory)).sort(), cases.map(c => `${c.input.id}.out`).sort());
  const outputs = new Map();
  let coveredTotal = 0;
  for (const test of cases) {
    const bytes = await readFile(join(outputDirectory, `${test.input.id}.out`));
    const output = JSON.parse(bytes.toString('utf8'));
    checkShape(output, test.input, bytes.length);
    outputs.set(test.input.id, output);
    let covered = 0;
    let absoluteError = 0;
    let width = 0;
    for (let i = 0; i < test.truth.length; i++) {
      const q = output.trophies[i].q;
      const actual = test.truth[i];
      if (actual >= q[0] && actual <= q[4]) covered++;
      absoluteError += Math.abs(q[2] - actual);
      width += q[4] - q[0];
    }
    const count = test.truth.length;
    assert.ok(covered / count >= 0.80, `${test.input.id}: inadequate 90% interval coverage (${covered}/${count})`);
    assert.ok(absoluteError / count < 4, `${test.input.id}: median MAE exceeds 4 percentage points`);
    assert.ok(width / count < 30, `${test.input.id}: intervals are uninformatively wide`);
    coveredTotal += covered;
    console.log(`${test.input.id}: coverage ${covered}/${count}, median MAE ${(absoluteError / count).toFixed(3)} pp, mean 90% width ${(width / count).toFixed(3)} pp; valid ${bytes.length}-byte output`);
  }
  assert.ok(coveredTotal / (cases.length * trophies.length) >= 0.85, 'Aggregate coverage below 85%');

  // Rising tracked-player counts should allow rarity to fall despite ongoing unlocks.
  const declining = outputs.get('dilution').trophies.filter((row, i) =>
    row.q[2] < dilutionCase.input.history.at(-1).rates[i]).length;
  assert.ok(declining >= 14, 'Population dilution did not lower predicted rates');
  const rising = outputs.get('logistic').trophies.filter((row, i) =>
    row.q[2] > logisticCase.input.history.at(-1).rates[i]).length;
  assert.ok(rising >= 14, 'Logit trend did not capture increasing achievement rates');

  // Re-run the same three cases to verify deterministic serialization.
  const repeatDirectory = join(workspace, 'repeat');
  execute(repeatDirectory);
  for (const test of cases) {
    assert.deepEqual(await readFile(join(outputDirectory, `${test.input.id}.out`)),
      await readFile(join(repeatDirectory, `${test.input.id}.out`)), 'Nondeterministic output');
  }

  // Counterfactuals on the same cases isolate uncertainty horizon and residual noise.
  const shortHorizon = structuredClone(reversionCase.input);
  shortHorizon.horizon_minutes = 360;
  shortHorizon.target = stamp(6.25);
  await writeFile(join(inputDirectory, 'reversion.in'), JSON.stringify(shortHorizon));
  const cleanReversion = buildCase('reversion', reversionSamples, 6, 3, 0).input;
  await writeFile(join(inputDirectory, 'logistic.in'), JSON.stringify(cleanReversion));
  const counterDirectory = join(workspace, 'counterfactuals');
  execute(counterDirectory);
  const shortBytes = await readFile(join(counterDirectory, 'reversion.out'));
  const cleanBytes = await readFile(join(counterDirectory, 'logistic.out'));
  const shortOutput = JSON.parse(shortBytes.toString('utf8'));
  const cleanOutput = JSON.parse(cleanBytes.toString('utf8'));
  checkShape(shortOutput, shortHorizon, shortBytes.length);
  checkShape(cleanOutput, cleanReversion, cleanBytes.length);
  const logitWidth = output => output.trophies.reduce((total, row) =>
    total + logit(clamp(row.q[4] / 100, 1e-8, 1 - 1e-8)) -
      logit(clamp(row.q[0] / 100, 1e-8, 1 - 1e-8)), 0);
  assert.ok(logitWidth(outputs.get('reversion')) > logitWidth(shortOutput), 'Horizon must widen uncertainty');
  assert.ok(logitWidth(outputs.get('reversion')) > logitWidth(cleanOutput), 'Residual noise must widen uncertainty');
  console.log('PASS: three known dynamics, coverage, direction, determinism, horizon/noise widening, and output contract.');
} finally {
  await rm(workspace, { recursive: true, force: true });
}
