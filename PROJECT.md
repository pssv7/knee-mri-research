# Knee MRI Project — Retained State

The 2026 RSNA competition is complete. Kaggle execution packages, failed runs,
duplicate submissions, intermediate ensembles, superseded plans, generated
metrics, and model checkpoints have been removed from version control. The
remaining repository is a compact research and engineering reference.

## Retained application and research code

- `src/` contains the reusable data, label, model, validation, and application
  modules.
- The remaining notebooks retain the core local workflow, excluding obsolete
  submission and Colab-export notebooks.
- `tools/` retains local auditing, label-quality, OAI target construction,
  research, and application utilities.
- `tests/` retains tests for the reusable modules and retained tools.

## Retained data and evidence

- Ignored `data/` contains local metadata and development inputs. It is not
  published with the source repository.
- Ignored `outputs/experiment18_oai_pretrain_v1/` retains OAI source tables, selected
  image manifests, and constructed targets. Its Colab notebooks were removed.
- Ignored `outputs/experiment22_image_hard_cases_v1/` retains reconciled labels and
  selection metadata. Rendered review images and its Colab notebook were removed.
- Local label-research directories contain the final Claude/Opus review,
  report-label evaluation, extraction repair, and latest candidate labels.
- Ignored `outputs/final_archive_2026/` contains compact evidence for the final
  competition submissions and Experiment 19 training result.

## Final competition records

- Experiment 9 submission SHA-256:
  `e9d6e787da63168190fad340342484616307b346aaf73f86c2a7b6385a36bda3`;
  recorded public score **0.950**.
- Experiment 8 submission SHA-256:
  `eda3318ebbc17d64b4d6f1c53992f1996fb62a2f4fa5489bff4dd54fc9ced498`;
  recorded public score **0.938**.
- The recovered fixed RSNA10 diagnostic comparison is under
  `outputs/experiment7_8_rsna10_validation_v1/recovered_complete/`:
  Experiment 7 macro **0.965497**, Experiment 8 macro **0.960296**. These are
  selection-overlapping diagnostic values, not independent held-out estimates.

No Kaggle watcher, controller, training job, or submission process remains
active in this repository.
