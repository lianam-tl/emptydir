#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = process.argv[2];
const outputPath = process.argv[3];
if (!htmlPath || !outputPath) {
  throw new Error("usage: export_seed.js HTML_PATH OUTPUT_PATH");
}

const html = fs.readFileSync(htmlPath, "utf8");
const sourceDirectory = path.dirname(htmlPath);
const elementBySelector = new Map();
const documentMock = {
  querySelector(selector) {
    if (!elementBySelector.has(selector)) {
      elementBySelector.set(selector, {
        value: "",
        innerHTML: "",
        addEventListener() {},
        classList: { add() {}, remove() {} },
      });
    }
    return elementBySelector.get(selector);
  },
  querySelectorAll() {
    return [];
  },
  getElementById() {
    return {};
  },
};
const context = vm.createContext({
  console,
  document: documentMock,
  Chart: function Chart() {},
});

for (const assetName of [
  "260722_entity_v02_half_sample_scores.js",
  "260722_entity_v02_db_updates.js",
  "260722_entity_v02_shot_counts.js",
  "260722_sme_h13_h14_dtype_scores.js",
]) {
  vm.runInContext(fs.readFileSync(path.join(sourceDirectory, assetName), "utf8"), context);
}

const inlineScripts = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
const dashboardScript = inlineScripts.map((match) => match[1]).find((script) => script.includes("const ENTITY_V02 ="));
if (!dashboardScript) {
  throw new Error("could not find dashboard script");
}

const exportScript = `
const exportedHalfRows = [
  ...ENTITY_V02_HALF_SAMPLE_SCORES.filter(row => !ENTITY_V02_DISPLAYABLE_DB_UPDATE_NAMES.has(row.name)),
  ...ENTITY_V02_DB_UPDATE_HALF_SAMPLE_SCORES.filter(row => ENTITY_V02_DISPLAYABLE_DB_UPDATE_NAMES.has(row.name))
];
const exportedHalfSamplesByName = Object.fromEntries(exportedHalfRows.map(row => [row.name, row.samples]));
globalThis.exportedRows = ENTITY_V02.map(row => ({
  name: row.name,
  run_id: row.run,
  path: row.path,
  dataset_revision: row.datasetRevision || ENTITY_V02_REVISION,
  backfilled: Boolean(row.backfilled),
  scored: row.scored,
  sample_count: row.scoreSampleCount || (row.datasetRevision ? 18 : 20),
  parse_failures: row.parseFailures || 0,
  failed: row.failed,
  naming: row.naming,
  score: row.score,
  delta: row.delta,
  full_naming: row.fullNaming,
  full_score: row.fullScore,
  full_delta: row.fullDelta,
  half_naming: row.halfNaming,
  half_score: row.halfScore,
  half_delta: row.halfDelta,
  average_shots: row.averageShotSegments,
  average_shot_duration: row.averageShotDuration,
  parsed_media_count: row.shotSegmentCounts.parsed_media_count,
  total_shot_segments: row.shotSegmentCounts.total_shot_segments,
  half_samples: exportedHalfSamplesByName[row.name] || {}
}));
`;
vm.runInContext(`${dashboardScript}\n${exportScript}`, context);

const payload = {
  generated_at: new Date().toISOString(),
  rows: context.exportedRows,
};
fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
console.log(`exported ${payload.rows.length} rows to ${outputPath}`);
