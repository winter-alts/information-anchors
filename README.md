# Information Anchors: Understanding and Using Forecast-Relevant Information in Time-Series Foundation Models

## Package contents
- `configs/` contains the registered experiment configurations.
- `experiments/` contains the model adapters, estimators, probes, interventions, and downstream methods.
- `scripts/` contains the experiment launchers and analysis utilities.
- `tests/` contains focused tests for the included code.
- `third_party/` contains third-party notices, licenses, and the TS-RAG patch.
- `requirements-atlas.txt` and `requirements-anchor-rag.txt` list environment dependencies.
- `SHA256SUMS.txt` lists SHA-256 digests for every included file except the checksum manifest itself.

The manuscript source and PDF, paper tables, generated figures, datasets, predictions, model weights, and runtime caches are not bundled.

## Experiments in this repository

### Information-anchor atlas, probes, and interventions

`experiments/information_anchor/` implements frozen-model activation capture,
null-calibrated MI estimation, target-aligned probes, selection stability,
history controls, and donor-replacement interventions. The 42 registered
model–dataset configurations are in `configs/`. Each config's
`data.local_path` is the placeholder `LOCAL_PATH`; replace it with the path to
that dataset's CSV before running the configuration. The V6 config generator
uses `INFORMATION_ANCHOR_DATA_ROOT` when set, and otherwise writes relative CSV
paths.

Example atlas run:

```bash
python -m experiments.information_anchor.run_mi \
  --config configs/chronos2_etth1_v6.json \
  --device cuda:0
```

The V6 launchers in `scripts/information_anchor/` coordinate the registered
probe and intervention runs. They expect completed atlas outputs under the
local `results/` directory. The selected analysis scripts and focused tests
are under `scripts/` and `tests/`.

### Seven-dataset Anchor-RAG experiment

`scripts/information_anchor/run_official_tsrag_mi.py` is the experiment runner;
`build_official_rag_mi_artifact.py` prepares the history-only MI sidecar, and
`summarize_anchor_rag_pure_mi_system.py` reproduces the paired test inference
and ablation summaries from per-origin losses.

The frozen TS-RAG predictor receives the MI-selected top-10 candidates from the
fixed top-20 retrieval pool. MI distances are used by candidate selection and
the outer forecast correction. The query future is not used to select or weight
candidates. The `pure_mi_locked_protocol.csv` file records the dataset-specific
query ranges and the locked correction gain and disagreement scale. Those
parameters were selected on three validation blocks by minimizing the worst
block MSE ratio to TS-RAG, then held fixed for test evaluation.

The runner expects these external inputs, which are not included here:

1. the seven retrieved-candidate CSVs and the official TS-RAG retrieval
   databases;
2. history-only MI sidecars produced by the included builder and registered
   atlas/protocol code;
3. the public Chronos-Bolt weights and the TS-RAG retrieval checkpoint.

Set up the pinned TS-RAG source overlay with `bash scripts/setup_tsrag_source.sh`.
Then run `python scripts/information_anchor/run_official_tsrag_mi.py --help` for
the runner interface. The checked-in protocol table is the source of the
dataset-specific `--query-end`, `--forecast-fusion-strength` (correction gain),
and `--forecast-fusion-confidence-scale` values. Use
`--system-ablation-suite` to evaluate the locked ablation arms on the same test
windows. The summarizer accepts those run directories through `--results-root`
and writes inference tables through `--output-dir`.

Per-origin loss arrays, prediction dumps, caches, downloaded datasets, and
model weights are intentionally excluded. Aggregated paper tables and the code
needed to regenerate them from a full run are included.

## Environment

The reference experiments used NVIDIA A100 80-GB GPUs. The final downstream
Anchor-RAG jobs recorded Python 3.12.3, PyTorch 2.5.1+cu124, and Transformers
4.57.3. The atlas experiments use model-specific dependencies listed in
`requirements-atlas.txt`; install a CUDA-matched PyTorch build separately.
`requirements-anchor-rag.txt` records the non-PyTorch packages for the
downstream runner. The exact model revisions and protocol settings are recorded
in the registered JSON configurations.

## Verify checksums

Check bundled-file hashes from the repository root with:

```bash
sha256sum -c SHA256SUMS.txt
```
