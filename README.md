# ConServe

ConServe is a research framework for efficient Text Image Machine Translation (TIMT). It reduces the visual-token cost of multimodal translation while preserving translation-relevant visual context.

The repository contains four main components:

- **SQ-FPS**: context-preserving visual-token compression using spatial-quota farthest-point sampling.
- **M3 inference**: multi-region translation with up to three target text regions per image.
- **OPSD/OPD training**: teacher-student training for preserving translation preferences under compressed visual inputs.
- **Agent evaluation**: automated evaluation of translation faithfulness and fluency.

The code is intended for reproducing the ConServe experiments and for research on efficient multimodal text-image translation.

## Repository layout

```text
.
├── compression/      # SQ-FPS and visual-token compression
├── inference/        # M3 inference pipelines
├── training/         # SFT data preparation and OPSD/OPD training
├── token_compress/   # Training-side token compression integration
├── agent_eval/       # Agent-based translation evaluation
├── configs/          # Example runtime configurations
├── docs/             # Data format reference
├── scripts/          # Auxiliary test/batch scripts
├── requirements.txt
├── LICENSE
├── NOTICE
└── README.md
```

## Installation

Use Python 3.10 or newer. A CUDA installation compatible with the installed PyTorch and vLLM is required for model inference. Install the Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The version floors in `requirements.txt` do not guarantee Qwen3.5 support. Use a mutually compatible Transformers and vLLM environment as described by the [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B); `transformers 4.57.0`, for example, cannot import its model class. For training, install verl in the same environment (see [Training](#training)). The paper used H100 GPUs: 8 for SFT data generation, 16 across 2 nodes for SFT, and 12 student plus 4 teacher GPUs for OPSD.

See [`.env.example`](.env.example) for additional settings. Scripts do not load `.env` automatically.

## Quick start

This example exercises the three-box inference path with a synthetic image and matching OCR. It requires a CUDA GPU, a Qwen3.5-compatible Transformers/vLLM environment, and about 20 GB of disk space for the [base Qwen3.5-9B weights](https://huggingface.co/Qwen/Qwen3.5-9B). It is a pipeline smoke test; paper results require the M3 SFT checkpoint described under [Training](#training).

From the repository root, after [installation](#installation):

```bash
export MODEL_DIR=/path/to/Qwen3.5-9B
hf download Qwen/Qwen3.5-9B --local-dir "$MODEL_DIR"
export DEMO_DIR=/tmp/conserve-demo
python scripts/make_quickstart_input.py --out-dir "$DEMO_DIR"
python inference/SFT_M3_optimized_hf_spatial_quota_fps.py \
  --image-dir "$DEMO_DIR/images" --ocr-dir "$DEMO_DIR/ocr" \
  --output-dir "$DEMO_DIR/output" --model-path "$MODEL_DIR" \
  --target-lang 中文 --m 3 --keep-ratio 0.30
```

The generated image contains three English text regions. Check `$DEMO_DIR/output/demo__vlm.json` for the three paragraph results. For your own images, generate OCR JSON with [the PaddleOCR adapter](docs/data_formats.md#12-generate-ocr-for-your-images), or supply files in the [documented format](docs/data_formats.md#11-ocr-json-input).

## Inference

The main entry point is:

```text
inference/SFT_M3_optimized_hf_spatial_quota_fps.py
```

It first renders the M3 one-image/three-box prompt, encodes the image with the vision encoder in [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), preserves OCR foreground tokens, and selects background representative tokens with spatial-quota FPS. The selected visual embeddings are then passed to the language decoder through the multimodal embedding interface.

Use the [quick start](#quick-start) command with your own `--ocr-dir`, `--image-dir`, `--output-dir` and local `--model-path`. OCR files and images must share a filename stem (`demo_ocr.json` pairs with `demo.png`). Set `CONSERVE_COMPRESSION_DIR` only if `compression/` is not beside `inference/`. For ablations, `--cluster-method fps` disables spatial quotas; `--prototype centroid` pools representatives instead of retaining real tokens.

The lower-level implementation is in `compression/`:

- `ogstb_cluster_merge.py`: spatial quota, diversity weighting and FPS logic.
- `ogstb_b1_merge.py`: visual embedding extraction and foreground masking.
- `run_M3_multi_box_b1.py` / `run_M3_multi_box_cluster.py`: integration adapters.

## Training

**M3 SFT checkpoint.** Use the included [data builder, filter, exporter and SFT launcher](docs/training.md) to fine-tune [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) on your own multi-region training data. The resulting local checkpoint is `STUDENT_MODEL` for OPSD; `TEACHER_MODEL` defaults to the same path. Model weights and training data are not distributed with this code. The upstream base model can exercise the inference code path, but paper results use the M3 SFT checkpoint.

**Install and patch verl.** Use [verl](https://github.com/verl-project/verl) commit `3ba8181` in the same environment as this project. Install its FSDP/vLLM backend dependencies following that checkout's instructions. The two included patches modify Qwen3.5 visual-token handling and verl's FSDP forward pass:

```bash
export CONSERVE_ROOT="$(pwd)"  # run from this repository's root
export VERL_ROOT=/path/to/verl
git clone https://github.com/verl-project/verl.git "$VERL_ROOT"
git -C "$VERL_ROOT" checkout 3ba8181
pip install -e "$VERL_ROOT"
cp "$CONSERVE_ROOT/training/qwen3_5.py" "$VERL_ROOT/verl/models/transformers/qwen3_5.py"
cp "$CONSERVE_ROOT/training/transformer_impl.py" "$VERL_ROOT/verl/workers/engine/fsdp/transformer_impl.py"
```

The launcher uses the included `training/opsd_vl_global_caption.sh` to run verl's FSDP student and vLLM teacher with SQ-FPS and forward-KL distillation. It adds this repository to `PYTHONPATH` automatically.

**Prepare data.** Supply your own Agent results, VLM translations, image list, OCR JSON and images. The [data guide](docs/data_formats.md#3-training-data) shows how to create `AGENT_RESULTS_JSONL` from the evaluation client and how its IDs match the image list and VLM outputs:

```bash
export AGENT_RESULTS_JSONL=/path/to/results.jsonl
export VLM_OUTPUT_DIR=/path/to/vlm_outputs
export IMAGE_LIST_JSONL=/path/to/image_list.jsonl
export OCR_DIR=/path/to/ocr
export OUTPUT_DIR=/path/to/opsd_data
bash "$CONSERVE_ROOT/training/build_opsd_data.sh"
```

The builder writes `train.parquet` and `val.parquet` in `OUTPUT_DIR`. If the training runtime mounts the data under another path, set both `MNT_ROOT` and `NOTEBOOK_ROOT` (see [`.env.example`](.env.example)).

**Train.** The default launcher allocation is 12 student GPUs and 4 teacher GPUs on one node. Other GPU layouts can be configured through the variables in [`.env.example`](.env.example).

```bash
export STUDENT_MODEL=/path/to/m3_sft_checkpoint
export TRAIN_PARQUET="$OUTPUT_DIR/train.parquet"
export VAL_PARQUET="$OUTPUT_DIR/val.parquet"
bash "$CONSERVE_ROOT/training/train_fps_prune.sh"
```

Use `CONSERVE_DRY_RUN=1 bash "$CONSERVE_ROOT/training/train_fps_prune.sh"` to inspect the verl command without starting training. See [`token_compress/README.md`](token_compress/README.md) for compression settings.

The M3 SFT commands and data conversion are in the [training guide](docs/training.md).

## Evaluation

`agent_eval/skills/` contains the visual evaluator's base prompts; `agent_eval/visual_eval_agent/agent_skills/` contains its detailed skills. The text evaluator loads its prompts from `agent_eval/text_eval_agent/prompts/` and detailed skills from `agent_eval/text_eval_agent/agent_skills/`. The Python client and scoring/parser modules are in `agent_eval/text_eval_agent/`. A provider-compatible runtime configuration is provided at:

```text
configs/text_critic_runtime.example.json
```

Before running, edit a private copy of that config with the approved provider endpoint and export credentials in the shell:

```bash
export MODEL_SERVICE_AK='your-access-key'
export MODEL_SERVICE_SK='your-secret-key'
```

Build a manifest containing image IDs, source text and model translations; the [manifest format](docs/data_formats.md#21-manifest-input) shows the complete schema and an example. Then run:

```bash
PYTHONPATH="$PWD/agent_eval" python -m text_eval_agent.text_critic_batch \
  --manifest /path/to/manifest.json \
  --runtime-config /path/to/runtime.json \
  --output-jsonl /path/to/results.jsonl \
  --output-json /path/to/results.pretty.json \
  --max-workers 2
```

The public code supports recomputing Faithfulness, Fluency and the overall score. The reference EATA evaluator used the API-hosted [Gemini 3.1 Pro Preview](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview) (`gemini-3.1-pro-preview`). Model weights and API access are not included; use an authorized service endpoint and credentials. The batch client accepts a compatible signed-API endpoint, but a different evaluation model may produce different scores. If omission filtering is used, document the exact rule (for example, remove `omission` and `under_translation` while retaining other errors) in the experiment record.

For fixed-OCR test batches, `scripts/run_full_test_fixed_ocr_agent_batch.sh` also requires `FULL_TEST_RESULT_ROOT`, `FULL_TEST_DATA_ROOT` and `FULL_TEST_OUTPUT_ROOT`. Supply a private `TEXT_CRITIC_RUNTIME_CONFIG` and service credentials for live evaluation; set `TEXT_CRITIC_DRY_RUN=1` to validate input construction without API calls.

## Data formats

The pipelines exchange three kinds of files, summarized below; see [`docs/data_formats.md`](docs/data_formats.md) for complete schemas and examples:

- **OCR JSON** (inference input): per-image PaddleOCR-style files in `--ocr-dir`, matched to images by filename stem. OCR items are grouped into paragraphs by `paragraphNo`; each item carries a `polygon`, text and an optional `paragraph_polygon`.
- **Evaluation manifest** (Agent eval input): a JSON array, or an object with a `tasks` array. Each task has a `task_id` and either `source_text` / `translated_text` or a `text_pairs` list.
- **Training records**: the SFT builder emits `manifest_train.jsonl` / `manifest_eval.jsonl`; the OPSD data script emits positive/negative teacher JSON and combines verl-compatible `train.parquet` / `val.parquet` for `TRAIN_PARQUET` / `VAL_PARQUET`.

Inference output is `{stem}__vlm.json`, containing one entry per paragraph with `should_translate`, `original_text` and `translated_text`.

## Data and model availability

The MCiT dataset can be downloaded from the [InImageTrans](https://github.com/fzuo1230/InImageTrans) project. Supply its images and matching OCR JSON at run time via `--image-dir` / `--ocr-dir` for inference. For OPSD data preparation, supply the image list and OCR directory described under [Training](#training), then pass the generated parquet paths as `TRAIN_PARQUET` / `VAL_PARQUET`.

For inference, `--model-path` accepts a local Qwen3.5-9B checkpoint. Use the upstream base checkpoint for the quick start, or produce an M3 SFT checkpoint with the [training guide](docs/training.md) for research runs.

For reproducibility, keep an experiment record for every run — the dataset version and hash, model revision, decoding and compression parameters, and random seeds.

## License

Project-owned source is licensed under the [Apache License 2.0](LICENSE). The two model patches retain the upstream Apache-2.0 notices; the OPSD launcher credits the upstream example. See [`NOTICE`](NOTICE).
