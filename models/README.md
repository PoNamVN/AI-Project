# Model artifacts

Large artifacts are ignored by Git. Existing CNN/MobileNetV2 experiments and the current V3.1 runtime live under `data/processed/`, not in this folder. V3.1 uses `model.json` and `classifier.npz` with pinned pretrained encoder/detector caches. Follow [machine setup](../docs/team_setup.md) for the separate runtime bundle and checksum verification.

Keep the ordered class mapping, preprocessing metadata, configuration, source/split hashes and evaluation context with each model version. Pretrained weights alone are not a trained CarVision classifier. Packaging the existing model does not establish new accuracy or approve its reference labels.
