# Data artifacts

raw/ holds approved unchanged images/annotations; processed/ holds cleaned derivatives, class mapping and manifests. Both are currently empty and excluded from Git except directory markers.

Before acquisition record source, revision, terms/date, original labels/splits, download time and hashes. Never commit private uploads or raw images. Save acquisition instructions and non-sensitive audit summaries so authorized students can reproduce the process.

Future manifest fields: sample_id, source_id, relative_path, original_label, class_id, original_split, assigned_split, sha256 and duplicate_group. Preserve original annotations and record every exclusion. No duplicate group may cross train/validation/test. Fit transforms on training only and share inference preprocessing. Final manifests/counts do not exist yet.
