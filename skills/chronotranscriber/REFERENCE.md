# ChronoTranscriber reference

Disclosed reference for [SKILL.md](SKILL.md). The stderr message names the fix for most
errors; this page holds the fixes it does not name.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| an item `blocked`, `items_blocked` above 0, exit 1 | another run leases the item; a killed run's lease expires after 10 minutes. An item waiting for a batch is blocked too (exit 0): `batch check --input <output folder>` |
| pages stored as `[transcription error]`, or `batch check --wait` exits 1 | rerun the command, or `repair --input <rendered file, database or output folder>`; `--all-failures` also redoes pages marked no text or not possible, and its `--json` reports `usage` |
| a page skipped as `skipped_no_text` holds text | `repair --input <output> --skipped` transcribes the skipped pages without the pre-pass |
| `batch check` reports an unresolved submission | settle it with `--adopt BATCH_ID` or `--abandon SUBMISSION_ID` (README, Batch Processing) in a direct call, never in a step a wrapper may rerun |
