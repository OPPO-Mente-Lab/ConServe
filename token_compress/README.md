# Training-time visual token compression

The patched training model imports this package for OCR box handling and
student-side SQ-FPS token selection. `TOKEN_COMPRESS=1` enables it;
`TOKEN_COMPRESS_MODE=prune` selects physical token pruning. The package loads
the clustering implementation from this repository's `compression/` directory
by default. `TOKEN_COMPRESS_CLUSTER_CODE_DIR` can point to an equivalent local
implementation when needed.

The training launcher in `training/train_fps_prune.sh` places this repository
on `PYTHONPATH` before starting a compatible external `verl` checkout.
