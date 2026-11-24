# Data artifacts

`raw/` holds unchanged source images/annotations and candidate provenance; a folder name does not mean its labels are approved. `processed/` holds derived data, manifests and local model runs. Both are excluded from Git except directory markers. A fresh clone contains no dataset or model. See [machine setup](../docs/team_setup.md) for the separate runtime bundle.

Before acquisition record source, revision, terms/date, original labels/splits, download time and hashes. Never commit private uploads or raw images. Save acquisition instructions and non-sensitive audit summaries so authorized students can reproduce the process.

Existing preparation tools produce review/split manifests with paths, labels, checksums and vehicle-group candidates. Preserve original annotations and record every exclusion. No duplicate group may cross train/validation/test. Fit transforms on training only and share inference preprocessing. Existing experimental splits are development evidence, not independently verified street data. The additional Week 5 candidates still await human review.
