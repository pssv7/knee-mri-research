# Fixed protocol ensemble submission — 0.914

Submission **56196386** is COMPLETE with public score **0.914**, confirmed by
the user and authenticated Kaggle API on September 13, 2026 around 15:07 UTC.
It is the new incumbent, +0.003 over submission 56135074 at 0.911.

The user explicitly requested submission to the leaderboard. One entry was
submitted September 13 at 01:31:11.647 UTC. Kaggle reported four entries
remaining that day; no duplicate entry was made.

- Notebook: https://www.kaggle.com/code/pssv27/knee-16-24-protocol-fixed-blend
- Version 1, kernel ID 134134153; private, offline, Tesla T4, nine-hour cap.
- Model dataset: `pssv27/knee-ensemble-protocol-private`, version 1.
- Manifest: `outputs/ensemble_16_24_protocol/ensemble.json`.
- Recipe: five selected models each at 16 slices, 24 slices and protocol
  selection; each checkpoint has probability weight 1/15. No EMA/SWA,
  rank averaging, soup, per-finding weights or flip TTA.

## Verification

The notebook's cell sources and private/offline/image/input metadata matched
the local staging after pull. Fifteen model hashes and equal weights matched
the passed benchmark. The visible test yielded three valid rows, exact expected
columns and IDs, finite bounded probabilities and zero fallbacks. It completed
in 74.804 seconds including setup. The CSV SHA-256 was
`8fbea01856373182cfe575234c15036ff53c7cee510de07f88b8ce62d9ece73f`.

Staged notebook SHA-256:
`117819fc92146475e185e5b79c47f3adc520e89b2c518a13ed75a27e0bbfa2ae`.
Manifest SHA-256:
`dbb09326c7876930ddd6ae65a0dd2f678179c8260d8d02d08c4ac6e9a4cf0469`.

The earlier full OOF comparison improved 0.876676 to 0.885228 on all 4349
studies. The private GPU benchmark passed 100/100 raw studies with zero errors
and projected 1.925 hours for 1300 studies. These validation results supported
the submission; neither is the hidden-test runtime. The scoring API supplied
the public score but no scoring-run runtime or private score.

Receipts and outputs: `outputs/kaggle_protocol_submission_receipt/`,
`outputs/kaggle_protocol_submission_results/` and
`outputs/kaggle_protocol_submission/competition_receipt.json`.
Preserve this exact package as the new incumbent and retain the 0.911 package
for comparisons. Next planned experiment: independently screened ConvNeXt-V2;
restage with the fixed diagnostic adapter before launch. No new job or entry
was launched during this score verification.
