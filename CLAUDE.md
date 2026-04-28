# histomicstk CLAUDE.md

Python image analysis library and Slicer CLI Docker tasks for the Digital Slide Archive. CLIs run inside the `dsarchive/histomicstk` Docker image spawned by the Celery worker.

## Key commands

```bash
# Lint / test
tox -e lint
tox -e py310     # or py311, py312, py313, py314

# Rebuild Docker image with local CLI changes (fast — layers on existing image)
docker build -f Dockerfile.pixel-classifier -t dsarchive/histomicstk:latest .

# Re-register updated image with running DSA (no pull from Docker Hub)
PASS="<admin-password>"
TOKEN=$(curl -s -u "admin:$PASS" 'http://localhost:8080/api/v1/user/authentication' | python3 -c "import sys,json; print(json.load(sys.stdin)['authToken']['token'])")
curl -s -X PUT 'http://localhost:8080/api/v1/slicer_cli_web/docker_image' \
  -H "Girder-Token: $TOKEN" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'name=dsarchive%2Fhistomicstk%3Alatest&pull=false'
```

## CLI structure

Each CLI lives in `histomicstk/cli/<ToolName>/`:
- `<ToolName>.xml` — Slicer CLI parameter descriptor (defines the REST interface)
- `<ToolName>.py` — Python entry point; uses `CLIArgumentParser` + `girder_client`
- `__init__.py` — empty package marker

`slicer_cli_list.json` is the registry — add new CLI names here.

## Deployed CLIs

| CLI | Purpose |
|---|---|
| `TridentEmbeddings` | Generate patch/slide embeddings via TRIDENT |
| `BuildSlideClassifier` | Train slide-level classifier on TRIDENT embeddings |
| `ApplySlideClassifier` | Apply slide classifier to a folder of slides |
| `BuildPixelClassifier` | Train pixel-level classifier from polygon/brush annotations; posts prediction overlay |
| `ApplyPixelClassifier` | Apply pixel classifier to a target slide; posts prediction overlay |
| `SuperpixelSegmentation` | SLIC superpixel segmentation |
| `PositivePixelCount` | Count positive pixels by stain intensity |
| `NucleiDetection` / `NucleiClassification` | Nuclear detection and classification |
| `ColorDeconvolution` / `SeparateStains*` | Stain separation |

## Pixel classifier shared utilities (`pixel_classifier_utils.py`)

Shared by `BuildPixelClassifier` and `ApplyPixelClassifier`:
- `read_annotation_as_training_data()` — reads pixelmap or polygon/shape elements; auto-detects classes from element groups when `classes=[]`
- `post_pixelmap_annotation()` — encodes a label map as a pixelmap annotation and POSTs to Girder
- `get_patch_embeddings()` — TRIDENT h5 primary path for embedding lookup
- `get_patch_crops_from_ts()` — extracts patch crops from a large_image TileSource
- `build_classifier()` — builds an sklearn Pipeline for the chosen model type
- `upload_model_to_girder()` — serializes and uploads model bundle + training report

## Dockerfile.pixel-classifier

Thin layer on `dsarchive/histomicstk:latest` that copies only the new CLI files. Use this for fast local rebuilds instead of the full Dockerfile (which reinstalls PyTorch and TRIDENT).
