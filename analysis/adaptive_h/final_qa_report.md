# Final Adaptive-h Benchmark QA

- Status: **PASS**
- Errors: 0
- Warnings: 1

| Check | Result | Severity | Detail |
| --- | --- | --- | --- |
| manifest_complete | PASS | error | status=complete |
| no_s0 | PASS | error | uses_s0=False |
| iid_protocol | PASS | error | calibration=iid/r=1, test=iid/r=1 |
| job_count | PASS | error | actual=600, expected=600 |
| paper_macrorep_threshold | PASS | error | unique_macroreps=50, require_final=True |
| paired_arms_complete | PASS | error | jobs_with_incomplete_arms=0 |
| aggregate_metrics_finite | PASS | error | ['coverage', 'coverage_interval', 'width', 'interval_score', 'q_hat'] |
| per_point_file_count | PASS | error | actual=1800, expected=1800 |
| per_point_schema | PASS | error | all required columns present |
| per_point_values_finite | PASS | error | all finite |
| paired_test_data_identical | PASS | error | all paired arms share test data |
| grid_boundary_rate | FAIL | warning | max_interval_clip=0.3010, max_y_outside=0.0010, arm_files_over_threshold=2/1800, mean_interval_clip=0.000214, mean_y_outside=0.000010, threshold=0.0200 |
| summary_exists | PASS | error | experiments/adaptive_h/output_final_adaptive_h/summary.csv |
| paired_deltas_exists | PASS | error | experiments/adaptive_h/output_final_adaptive_h/paired_deltas.csv |
| scale_diagnostics_summary_exists | PASS | error | experiments/adaptive_h/output_final_adaptive_h/scale_diagnostics_summary.csv |
| coverage_sanity | PASS | warning | maximum marginal-coverage deviation=2.52 MCSE |
| scale_diagnostics_summary_complete | PASS | error | rows=12, expected_rows=12, value_columns=9 |
| score_homogeneity_per_arm_exists | PASS | error | experiments/adaptive_h/output_final_adaptive_h/score_homogeneity_per_arm.csv |
| score_homogeneity_summary_exists | PASS | error | experiments/adaptive_h/output_final_adaptive_h/score_homogeneity_summary.csv |
| score_homogeneity_paired_exists | PASS | error | experiments/adaptive_h/output_final_adaptive_h/score_homogeneity_paired_deltas.csv |
| score_homogeneity_per_arm_schema | PASS | error | all required columns present |
| score_homogeneity_key_match | PASS | error | actual_unique=1800, expected_unique=1800 |
| score_homogeneity_group_count | PASS | error | rows=1800, expected_rows=1800, expected_groups=10 |
| score_homogeneity_per_arm_values | PASS | error | rows=1800, metrics=['max_pairwise_score_ks', 'mean_pairwise_score_ks', 'bin_mean_score_range'] |
| score_homogeneity_metric_ranges | PASS | error | KS in [0,1] and bin-mean range nonnegative |
| score_homogeneity_summary_complete | PASS | error | rows=36, expected_rows=36, n_macro=50 |
| score_homogeneity_paired_complete | PASS | error | rows=1800, expected_rows=1800, comparisons=['oracle_minus_fixed', 'plugin_sd_nw_minus_fixed', 'plugin_sd_nw_minus_oracle'] |
| score_homogeneity_outputs_fresh | PASS | error | score outputs are no older than the latest per-point file |
| figure_qa_passes | PASS | error | path=experiments/adaptive_h/output_final_adaptive_h/figures/figure_qa.json, status=pass |
| figure_outputs_fresh | PASS | error | figure QA is no older than all summary and score inputs |
| final_asset_set_complete | PASS | error | verified_artifacts=19 |
