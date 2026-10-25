# Public historical regression fixtures

These small files were extracted **offline** from previously completed local
experiments. They are regression inputs, not fresh model runs or independently
reviewed medical evidence. The original records under `evaluations/results/`
remain unchanged and are excluded from the public Git candidate set.

`provenance.json` records each original record's repository-relative path and
SHA256, the projection's SHA256, and the exact extraction scope. The public
projections remove local absolute paths, hostnames, environment metadata, Git
working-tree inventories, and full request/response journals. All retained task,
article, classification, timing and label values are copied without correction.

- `writer/`: four saved Writer histories and their public-style results. Only
  fields used by the offline regression tests are retained.
- `claim_verifier/v1_full.json`: the original 24-row V1 observations needed for
  the bounded comparison guard test. The original source and dataset hashes are
  retained; the current Verifier must still fail that historical source guard.
- `claim_verifier/holdout_paired.json`: eight V1 and eight V2 rows needed to
  reproduce the explanation-review queue, preserving unreviewed labels/reasons.
- `historical_source/*.txt`: two original source files, kept byte-for-byte for
  historical SHA guards and stored as text so they are not active modules.
- `historical_source/qwen3_4b_manifest.json`: original model manifest metadata,
  not model weights, used only by an explicitly mocked offline source-guard test.

The helper `evaluations.public_fixtures` validates projection hashes before use.
Real-model runners still accept explicit historical inputs and preserve their
normal source/model guards. The regressions using these fixtures need neither
the author's local full results directory, an Ollama service, nor model weights.
