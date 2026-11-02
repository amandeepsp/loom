#!/usr/bin/env node

import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const skillRoot = process.env.DATA_ANALYTICS_SKILL_ROOT ??
  "/home/amandeeps/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills";
const scripts = `${skillRoot}/build-report/scripts`;

const { buildPortableArtifact } = await import(`${scripts}/build_portable_artifact.mjs`);
const { extractPortableChartSvgs } = await import(`${scripts}/extract_portable_chart_svgs.mjs`);
const { verifyPortableArtifact } = await import(`${scripts}/verify_portable_artifact.mjs`);

const [inputArg = "build/showcase/artifact.json", outputArg = "build/showcase/loom-showcase.html"] =
  process.argv.slice(2);
const inputPath = resolve(inputArg);
const outputPath = resolve(outputArg);
const artifact = JSON.parse(readFileSync(inputPath, "utf8"));

// Chromium headless-shell's --dump-dom mode does not always dispatch window.load
// before virtual time starts. Booting immediately is safe because the loader is
// emitted at the end of <body>, after both compressed templates exist.
function bootWithoutLoadEvent(html) {
  const original = [
    '  if (document.readyState === "complete") scheduleBoot();',
    '  else window.addEventListener("load", scheduleBoot, { once: true });',
  ].join("\n");
  if (!html.includes(original)) throw new Error("Portable loader boot sequence was not found.");
  const compatibilityStyle = `<style data-loom-dashboard-compat>
.analytics-top-bar { width: 100% !important; margin-right: 0 !important; margin-left: 0 !important; }
.analytics-top-bar .page-title-edit-target { width: 100% !important; max-width: 100% !important; margin-right: 0 !important; margin-left: 0 !important; }
.recharts-default-legend.chart-legend { display: flex !important; width: 100% !important; max-width: 100% !important; }
</style>`;
  return html
    .replace(original, "  void boot();")
    .replace("</head>", `${compatibilityStyle}\n</head>`);
}

let html = bootWithoutLoadEvent(buildPortableArtifact(artifact));
const extractionPath = `${outputPath}.extract.html`;
writeFileSync(extractionPath, html);
const staticCharts = await extractPortableChartSvgs({
  actionTimeoutMs: 5_000,
  htmlPath: extractionPath,
  readyTimeoutMs: 30_000,
});
rmSync(extractionPath, { force: true });
html = bootWithoutLoadEvent(buildPortableArtifact(artifact, { staticCharts }));
writeFileSync(outputPath, html);

const verification = await verifyPortableArtifact({
  actionTimeoutMs: 5_000,
  artifactPath: inputPath,
  htmlPath: outputPath,
  readyTimeoutMs: 30_000,
  timeoutMs: 45_000,
});
process.stdout.write(`${JSON.stringify({ ...verification, staticChartCount: Object.keys(staticCharts).length })}\n`);
