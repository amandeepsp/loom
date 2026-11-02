-- DuckDB source contract for the generated loom showcase dashboard.
-- `tools.showcase_metrics` emits one JSON file per reviewed dashboard dataset.

SELECT * FROM read_json_auto('build/showcase/datasets/hardware_summary.json');
SELECT * FROM read_json_auto('build/showcase/datasets/throughput_comparison.json');
SELECT * FROM read_json_auto('build/showcase/datasets/efficiency_ladder.json');
SELECT * FROM read_json_auto('build/showcase/datasets/resource_utilization.json');
SELECT * FROM read_json_auto('build/showcase/datasets/roofline.json');
SELECT * FROM read_json_auto('build/showcase/datasets/gemm_overlap.json');
SELECT * FROM read_json_auto('build/showcase/datasets/software_summary.json');
SELECT * FROM read_json_auto('build/showcase/datasets/accuracy_comparison.json');
SELECT * FROM read_json_auto('build/showcase/datasets/operator_mix.json');
SELECT * FROM read_json_auto('build/showcase/datasets/layer_metrics.json');
SELECT * FROM read_json_auto('build/showcase/datasets/model_sections.json');
SELECT * FROM read_json_auto('build/showcase/datasets/model_cycle_breakdown.json');
SELECT * FROM read_json_auto('build/showcase/datasets/instrumentation_status.json');
