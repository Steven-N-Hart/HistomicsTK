import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_classifier_utils import (  # noqa: E402
    get_patch_crops_from_ts,
    post_pixelmap_annotation,
)

from histomicstk.cli.utils import CLIArgumentParser


def _resolve_api_url(url):
    if not url:
        return url
    if url.startswith(('http://', 'https://')):
        return url
    return 'http://girder:8080/' + url.lstrip('/')


def _download_model(gc, model_item_id, job_dir):
    """Download model.pkl from the model Girder item."""
    files = list(gc.listFile(model_item_id))
    model_file = next((f for f in files if f['name'] == 'model.pkl'), None)
    if model_file is None:
        raise RuntimeError(
            f'No model.pkl found in Girder item {model_item_id}. '
            'Re-run BuildPixelClassifier to regenerate the model.'
        )
    local_path = os.path.join(job_dir, 'model.pkl')
    gc.downloadFile(model_file['_id'], local_path)
    return local_path


def _infer_on_slide(pipeline, ts, model_dict, magnification, patch_size,
                    model_type):
    """Run patch-by-patch inference and return (label_map, step)."""
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    meta = ts.getMetadata()
    native_mag = meta.get('magnification') or magnification
    slide_w = meta.get('sizeX', 0)
    slide_h = meta.get('sizeY', 0)

    scale = native_mag / magnification if native_mag else 1.0
    step = int(round(patch_size * scale))
    cols = max(1, int(np.ceil(slide_w / step)))
    rows = max(1, int(np.ceil(slide_h / step)))

    print(f'Inference: {cols}×{rows} patches at {magnification}x '
          f'(native {native_mag}x, step={step}px)...')

    centers = []
    for row in range(rows):
        for col in range(cols):
            cx = int(col * step + step // 2)
            cy = int(row * step + step // 2)
            centers.append((cx, cy))

    batch_size = 256
    all_preds = []

    if model_type in ('trident_linear', 'trident_mlp'):
        # For apply, we don't have an h5 file for the target slide by default.
        # Fall back to ResNet encoding so the model still works.
        print('  Note: TRIDENT embeddings not available for target slide — '
              'using ResNet-18 for inference encoding.')
        model_type = 'resnet_linear'

    if model_type == 'resnet_linear':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'  Device: {device}')
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = torch.nn.Identity()
        backbone.eval()
        backbone.to(device)
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        for i in range(0, len(centers), batch_size):
            batch_centers = centers[i:i + batch_size]
            crops = get_patch_crops_from_ts(ts, batch_centers,
                                            magnification, patch_size)
            with torch.no_grad():
                imgs = torch.stack(
                    [transform(c[:, :, :3]) for c in crops]
                ).to(device)
                X_batch = backbone(imgs).cpu().numpy().astype(np.float32)
            preds = pipeline.predict(X_batch)
            all_preds.extend(preds.tolist())
            if (i // batch_size) % 20 == 0:
                print(f'  {i}/{len(centers)} patches processed')

    else:  # random_forest
        from skimage.feature import local_binary_pattern

        for i in range(0, len(centers), batch_size):
            batch_centers = centers[i:i + batch_size]
            crops = get_patch_crops_from_ts(ts, batch_centers,
                                            magnification, patch_size)
            rows_feats = []
            for crop in crops:
                rgb = crop[:, :, :3].astype(np.float32) / 255.0
                r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
                cmax = np.maximum(r, np.maximum(g, b))
                cmin = np.minimum(r, np.minimum(g, b))
                s = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0)
                color_feats = np.array([
                    r.mean(), r.std(), g.mean(), g.std(),
                    b.mean(), b.std(), s.mean(), s.std(),
                ])
                gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
                gray_u8 = (gray * 255).astype(np.uint8)
                lbp = local_binary_pattern(gray_u8, P=8, R=1, method='uniform')
                hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
                rows_feats.append(np.concatenate([color_feats, hist]))
            X_batch = np.array(rows_feats, dtype=np.float32)
            preds = pipeline.predict(X_batch)
            all_preds.extend(preds.tolist())
            if (i // batch_size) % 20 == 0:
                print(f'  {i}/{len(centers)} patches processed')

    label_map = np.array(all_preds, dtype=np.uint8).reshape(rows, cols)
    return label_map, step


def main(args):
    import large_image

    api_url = _resolve_api_url(getattr(args, 'girderApiUrl', None))
    token = getattr(args, 'girderToken', None)
    if not api_url or not token:
        raise RuntimeError(
            'girderApiUrl/girderToken were not injected. '
            'Slicer CLI Web should populate these automatically.'
        )

    if not args.item_id:
        raise RuntimeError('item_id is required.')
    if not args.model_item_id:
        raise RuntimeError('model_item_id is required.')
    if not args.job_dir:
        raise RuntimeError('job_dir is required.')

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    os.makedirs(args.job_dir, exist_ok=True)

    # Download and load model
    print(f'Downloading model from item {args.model_item_id}...')
    model_path = _download_model(gc, args.model_item_id, args.job_dir)

    with open(model_path, 'rb') as fh:
        model_dict = pickle.load(fh)

    pipeline = model_dict['model']
    model_type = model_dict['embedder']
    patch_size = model_dict['patch_size']
    magnification = model_dict['magnification']
    classes = model_dict.get('categories') or [
        {'name': n} for n in model_dict.get('classes', [])
    ]
    session_name = model_dict.get('session_name', 'PixelClassifier')

    print(f'Model loaded: {model_type}, {len(classes)} classes, '
          f'patch_size={patch_size}, magnification={magnification}x')

    # Open target slide
    print(f'Opening slide {args.item_id}...')
    item_files = list(gc.listFile(args.item_id, limit=1))
    if not item_files:
        raise RuntimeError(f'Item {args.item_id} has no files.')

    slide_local = os.path.join(args.job_dir, item_files[0]['name'])
    file_resp = gc.get(f'file/{item_files[0]["_id"]}/download',
                       jsonResp=False, stream=True)
    with open(slide_local, 'wb') as fh:
        for chunk in file_resp.iter_content(chunk_size=8192):
            fh.write(chunk)

    ts = large_image.open(slide_local)

    # Magnification mismatch warning
    meta = ts.getMetadata()
    native_mag = meta.get('magnification')
    if native_mag and abs(native_mag - magnification) / max(native_mag, magnification) > 0.5:
        print(
            f'WARNING: Slide native magnification ({native_mag}x) differs '
            f'significantly from model training magnification ({magnification}x). '
            'Predictions may be unreliable.'
        )

    # Run inference
    categories_for_ann = []
    for cls in classes:
        categories_for_ann.append({
            'label': cls.get('name', ''),
            'fillColor': cls.get('fillColor', 'rgba(128,128,128,0.4)'),
            'strokeColor': cls.get('strokeColor', 'rgba(128,128,128,1)'),
        })

    label_map, step = _infer_on_slide(
        pipeline, ts, model_dict, magnification, patch_size, model_type
    )
    print(f'Inference complete. Label map: {label_map.shape}')

    # Post overlay annotation
    ann_name = f'PixelClassifier Applied — {session_name}'
    print(f'Posting overlay: {ann_name!r}...')
    overlay_id = post_pixelmap_annotation(
        gc, args.item_id, label_map, categories_for_ann,
        ann_name, step, magnification, origin_xy=(0, 0)
    )
    print(f'Overlay annotation ID: {overlay_id}')
    print('ApplyPixelClassifier complete.')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
