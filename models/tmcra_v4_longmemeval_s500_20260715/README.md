# TMCRA V4 benchmark weights

These are the TMCRA graph scorer and reranker weights used by the public
LongMemEval-S500 reproduction pipeline.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `node_scorer.pt` | graph node scoring | `d2318aafb07f9a15a1d95e6eac1b3e09afa07c570dd9d15d3f75544c9950f201` |
| `path_scorer.pt` | graph path scoring | `ecdc2bca51b7646a8a56c7db9cea4aeec581e64c3b61e6e06f7e9fa2a93b3bf5` |
| `tmcra_v3_reranker.pt` | learned retrieval reranking | `09a285b484ca857b24b53ad19d5998302f16d224ca9e4fd73a4eb5a52f022942` |

The weights are released under Apache-2.0. Load them only through the
repository runtime, which requires `torch.load(..., weights_only=True)` and
validates the expected checkpoint structure. Do not disable these checks for
untrusted files.

The public checkpoints contain only their inference contracts and model state.
Training state, training-machine paths, and run metadata are intentionally
excluded. Their tensors are unchanged from the benchmark-bound checkpoints.

The files are tracked with Git LFS. To pull only this release:

```bash
git lfs pull --include="models/tmcra_v4_longmemeval_s500_20260715/*.pt"
```
