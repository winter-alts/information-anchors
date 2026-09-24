# External TS-RAG source

The Anchor-RAG runner imports TS-RAG source at
`repository_packages/external_rag/TS-RAG/TS-RAG`. To recreate that source tree,
run from this repository root:

```bash
bash scripts/setup_tsrag_source.sh
```

The script checks out the pinned upstream revision and applies the local patch.
It does not fetch datasets, retrieval databases, checkpoints, or model weights.
See `THIRD_PARTY_NOTICES.md` for the upstream license.
