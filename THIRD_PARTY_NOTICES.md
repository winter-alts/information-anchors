# Third-party code and assets

- The TS-RAG overlay is based on the upstream repository at
  <https://github.com/UConn-DSIS/TS-RAG>, pinned to commit
  `73ac807789d2e61b8a3dfc8514e3fc947fe185cc`. The upstream MIT license is in
  `third_party/licenses/TS-RAG-MIT-LICENSE.txt`; `third_party/patches/` contains
  the experiment-specific changes.
- The paper uses the ICLR 2027 conference style files included in `paper/`.
- Pretrained model weights, benchmark datasets, and retrieval databases are
  external assets. Their terms are set by their respective providers and are
  not changed by this repository.

No Align-RAG source tree, git history, checkpoints, downloaded data, or hidden
model caches are bundled.
