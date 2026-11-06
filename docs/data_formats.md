# Data Formats

This document describes the file formats exchanged by the ConServe pipelines. All text files are UTF-8.

## 1. Inference

The MCiT dataset used in the paper can be obtained from the [InImageTrans](https://github.com/fzuo1230/InImageTrans) project. The inference scripts consume OCR JSON supplied by the user. `inference/step1_ocr_paragraph_vlm_ocr+vlm+bbx_ocr-vllm-generate.py` reads that OCR and the corresponding images to produce paragraph translations; it does not run OCR.

### 1.1 OCR JSON (input)

Files are placed in the directory passed to `--ocr-dir` and are matched to images in `--image-dir` by filename stem: `{stem}_ocr.json` (or `{stem}.json`) pairs with `{stem}.jpg` / `.jpeg` / `.png` / `.webp`, including one level of subdirectories.

The expected input schema follows PaddleOCR-style paragraph output:

```json
{
  "data": {
    "width": 1080,
    "height": 1920,
    "items": [
      {
        "text": "Settings",
        "text_suggest": "Settings",
        "polygon": [
          {"x": 120, "y": 80},
          {"x": 260, "y": 80},
          {"x": 260, "y": 130},
          {"x": 120, "y": 130}
        ],
        "paragraph_polygon": [
          {"x": 116, "y": 76},
          {"x": 264, "y": 76},
          {"x": 264, "y": 134},
          {"x": 116, "y": 134}
        ],
        "paragraphNo": 0,
        "confidence": 0.98,
        "angle": 0,
        "text_color": [0, 0, 0],
        "background_color": [255, 255, 255]
      }
    ]
  }
}
```

The top-level `data` wrapper is optional; `width`, `height` and `items` may also sit at the root.

Item fields:

| Field | Required | Description |
| --- | --- | --- |
| `paragraphNo` | yes | Paragraph id; items sharing an id form one translation box. |
| `polygon` | yes | Quadrilateral points of the text line, each `{"x", "y"}`. |
| `paragraph_polygon` | no | Paragraph-level polygon; if absent, the paragraph bbox is the union of item polygons. |
| `text` / `text_suggest` | one required | Recognized text (`text_suggest` takes precedence). |
| `confidence` | no | Recognition confidence in `[0, 1]`. |
| `angle` | no | Text angle in degrees. |
| `text_color`, `background_color` | no | RGB triplets used by the rendering helpers. |

### 1.2 Generate OCR for your images

The repository's inference commands read OCR; they do not perform OCR themselves. For raw images, install the inference engine and `paddleocr` using the [PaddleOCR installation guide](https://www.paddleocr.ai/main/en/version3.x/installation.html). Then generate ConServe's input JSON with the included adapter (PaddleOCR is needed only for this preparation step):

```bash
python scripts/generate_ocr.py --image-dir /path/to/images --ocr-dir /path/to/ocr --lang en
```

For `images/img0001.jpg`, the script writes `ocr/img0001_ocr.json` with the image dimensions, recognized text, confidence and four-point polygons. It assigns each recognized **text line** a separate `paragraphNo`; if a task needs several lines in one translation region, edit those lines to share a paragraph number and, if useful, add a `paragraph_polygon`. Inspect the OCR and paragraph grouping before running inference. You may use any other OCR tool that produces the [schema above](#11-ocr-json-input).

The base pipeline `inference/SFT_M3_optimized.py` additionally accepts a JSONL image list via `--image-list-jsonl` to select and order images and supply per-image target languages. Each line accepts `image_path` (or `path` / `filename`) and `tgt_lang` (or `target_language`), using either an ISO code (`zh`, `en`, ...) or a Chinese language name (`中文`, `英语`, ...).

### 1.3 VLM result JSON (output)

For each image the pipeline writes `{stem}__vlm.json`. Top-level fields:

```json
{
  "stem": "img0001",
  "image_path": "/path/to/img0001.jpg",
  "image_size": [1080, 1920],
  "target_lang": "中文",
  "tgt_lang_code": "zh",
  "m_per_batch": 3,
  "num_paragraphs": 12,
  "num_vlm_batches": 4,
  "paragraphs": [
    {
      "paragraph_no": 0,
      "paragraph_bbox": [116, 76, 264, 134],
      "padded_bbox": [110, 70, 270, 140],
      "ocr_text": "Settings",
      "should_translate": false,
      "original_text": "Settings",
      "translated_text": "",
      "skip_reason": null
    }
  ]
}
```

## 2. Agent evaluation

### 2.1 Manifest (input)

The manifest passed to `--manifest` is either a JSON array of tasks, or an object with a `tasks` array (a UTF-8 BOM is accepted):

```json
{
  "tasks": [
    {
      "task_id": "img0001",
      "source_language": "en",
      "target_language": "zh",
      "text_pairs": [
        {
          "pair_id": "img0001:p0",
          "source_text": "Settings",
          "translated_text": "设置"
        },
        {
          "pair_id": "img0001:p1",
          "source_text": "Follow",
          "translated_text": "关注"
        }
      ]
    }
  ]
}
```

A task may instead carry a single pair at the top level:

```json
{
  "task_id": "img0002",
  "source_language": "en",
  "target_language": "zh",
  "source_text": "Cancel",
  "translated_text": "取消"
}
```

`task_id` is required; each pair requires `source_text` and `translated_text`. `source_language` / `target_language` may be set per task or per pair.

### 2.2 Results (output)

`--output-jsonl` receives one JSON object per task (one per line), including `task_id`, `status`, and `scores` (for example `scores.text_score_5` on the 1-5 scale, plus Faithfulness and Fluency outputs). `--output-json` receives the same records as a pretty-printed JSON array, in input order.

## 3. Training data

### 3.1 SFT manifests

`training/build_sft_m3_datapairs.py` scans inference `__vlm.json` outputs and writes `manifest_train.jsonl` / `manifest_eval.jsonl`. Each record is one generated M3 training sample; an image can produce multiple samples. The [SFT training instructions](training.md) cover conversion to LLaMA-Factory format.

The filtering stage uses `training/hallucination_filter.py` by default. Use `--hallucination-script` to substitute another evaluator, or `--skip-filter` with an existing `filtered_images.json` in `--out-dir`.

```json
{
  "id": "train__img0001__k1_s0000",
  "split": "train",
  "stem": "img0001",
  "images": ["samples/train/train__img0001__k1_s0000/framed_multi.jpg"],
  "system": "M3 system prompt for one numbered box",
  "user": "Target language: Chinese. Translate box #2.",
  "assistant": "[{\"should_translate\":true,\"original_text\":\"Follow\",\"translated_text\":\"关注\"}]",
  "boxes": [
    {
      "paragraph_no": 2,
      "padded_bbox": [110, 70, 270, 140],
      "should_translate": true,
      "original_text": "Follow",
      "translated_text": "关注"
    }
  ]
}
```

### 3.2 OPSD positive/negative teacher records

`training/opsd_build_agent_pos_neg.py` converts evaluation results into two teacher JSON files. Images with `scores.text_score_5 == 5` become positive teachers; lower-scoring images become negative teachers whose error paragraphs carry evidence and a suggested correction:

```json
{
  "model": "agent_positive_teacher",
  "num_images": 1,
  "images": [
    {
      "stem": "img0001",
      "text_score_5": 5.0,
      "teacher_mode": "positive",
      "paragraphs": [
        {
          "paragraph_no": 2,
          "should_translate": true,
          "original_text": "Follow",
          "translated_text": "关注",
          "ocr_text": "Follow"
        }
      ]
    }
  ]
}
```

Negative paragraphs use `teacher_mode: "negative"` and add an `errors` list, each error containing the target fragment, evidence and a suggestion.

### 3.3 Training parquet

`training/build_opsd_data.sh` converts Agent results and matching VLM outputs into positive/negative annotations, then uses `training/opsd_vl_global_ab_teacher.py` to create and combine verl-compatible `train.parquet` and `val.parquet`. It also requires OCR JSON and an image list JSONL with `stem` and `image_path` on each line; `target_lang` and `source` are optional:

```json
{"stem":"img0001","image_path":"/path/to/img0001.jpg","target_lang":"中文"}
```

To create `AGENT_RESULTS_JSONL`, first write an [evaluation manifest](#21-manifest-input) from the VLM output and matching OCR. Use `task_id` equal to the image stem (`img0001`), and `pair_id` ending in `:pN`, where `N` is the matching OCR `paragraphNo` and VLM `paragraph_no` (for example, `img0001:p0` for paragraph 0). Evaluate it with the client described in the [README Evaluation section](../README.md#evaluation):

```bash
PYTHONPATH="$PWD/agent_eval" python -m text_eval_agent.text_critic_batch \
  --manifest /path/to/manifest.json \
  --runtime-config /path/to/private_runtime.json \
  --output-jsonl /path/to/results.jsonl \
  --output-json /path/to/results.pretty.json
export AGENT_RESULTS_JSONL=/path/to/results.jsonl
```

Each JSONL line must be a successful evaluation record with a matching `task_id`, a numeric `scores.text_score_5`, and `text_result.text_pairs`. The OPSD builder uses score 5 for positive teachers and lower scores with `text_result.unit_results` errors for negative teachers; failed or unscored records are skipped. The minimal shape of a **positive** record is:

```json
{"task_id":"img0001","status":"completed","scores":{"text_score_5":5},"text_result":{"text_pairs":[{"pair_id":"img0001:p0","source_text":"Settings","translated_text":"设置"}]}}
```

Place `img0001__vlm.json` in `VLM_OUTPUT_DIR`, `img0001_ocr.json` in `OCR_DIR`, and include the same stem and real image path in `IMAGE_LIST_JSONL` as shown above. Set those paths and `OUTPUT_DIR`, then run `training/build_opsd_data.sh` as shown in the [README Training section](../README.md#training). The script writes `train.parquet` and `val.parquet` under `OUTPUT_DIR`.

The image and OCR files, Agent results and VLM outputs are supplied separately; they are not included in this repository. By default, parquet image paths are absolute local paths. If training reads them under a different mount point, set `MNT_ROOT` and `NOTEBOOK_ROOT` together to map the local output directory to its runtime location. The parquet columns follow the verl data schema for the pinned checkout (commit `3ba8181`).
