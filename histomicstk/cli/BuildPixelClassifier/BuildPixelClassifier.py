import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_classifier_utils import (  # noqa: E402
    build_classifier,
    compute_oob_or_cv_accuracy,
    get_patch_crops_from_ts,
    get_patch_embeddings,
    post_pixelmap_annotation,
    read_annotation_as_training_data,
    upload_model_to_girder,
)

from histomicstk.cli.utils import CLIArgumentParser


def _resolve_api_url(url):
    if not url:
        return url
    if url.startswith(('http://', 'https://')):
        return url
    return 'http://girder:8080/' + url.lstrip('/')


def _add_background_samples(patch_coords_per_class, positive_centers,
                            ts, magnification, patch_size):
    """Sample random slide patches as background (class 0) for single-class training."""
    meta = ts.getMetadata()
    native_mag = meta.get('magnification') or magnification
    slide_w = meta.get('sizeX', 1)
    slide_h = meta.get('sizeY', 1)

    scale = native_mag / magnification if native_mag else 1.0
    step = int(round(patch_size * scale))

    # Build grid of all valid patch centers
    all_grid = []
    for row in range(0, slide_h, step):
        for col in range(0, slide_w, step):
            cx = col + step // 2
            cy = row + step // 2
            if (cx, cy) not in positive_centers:
                all_grid.append((cx, cy))

    # Sample 3× the positive count, up to the available grid size
    n_positive = sum(len(v) for v in patch_coords_per_class.values())
    n_bg = min(n_positive * 3, len(all_grid))
    rng = np.random.default_rng(42)
    chosen_idx = rng.choice(len(all_grid), size=n_bg, replace=False)
    bg_centers = [all_grid[i] for i in chosen_idx]

    print(f'  Sampled {n_bg} background patches (3× the {n_positive} positive patches).')
    result = {0: bg_centers}
    result.update(patch_coords_per_class)
    return result


def _build_training_data(patch_coords_per_class, item_meta, ts,
                         magnification, patch_size, model_type):
    """Extract embeddings for all annotated patches and build X, y arrays."""
    all_centers = []
    all_labels = []

    for cat_idx, centers in sorted(patch_coords_per_class.items()):
        all_centers.extend(centers)
        all_labels.extend([cat_idx] * len(centers))

    if len(all_centers) == 0:
        raise RuntimeError('No labeled patches found in the training annotation.')

    print(f'Extracting embeddings for {len(all_centers)} labeled patches '
          f'using model_type={model_type!r}...')

    if model_type in ('trident_linear', 'trident_mlp'):
        X = get_patch_embeddings(None, item_meta, all_centers,
                                 magnification, patch_size, model_type)
    else:
        crops = get_patch_crops_from_ts(ts, all_centers, magnification, patch_size)
        if model_type == 'resnet_linear':
            import torch
            import torchvision.models as tvm
            import torchvision.transforms as T

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'  Using device: {device}')
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

            batch_size = 64
            feats = []
            with torch.no_grad():
                for i in range(0, len(crops), batch_size):
                    batch = torch.stack(
                        [transform(c[:, :, :3]) for c in crops[i:i + batch_size]]
                    ).to(device)
                    feats.append(backbone(batch).cpu().numpy())
            X = np.concatenate(feats, axis=0).astype(np.float32)
        else:  # random_forest
            from skimage.feature import local_binary_pattern

            rows = []
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
                rows.append(np.concatenate([color_feats, hist]))
            X = np.array(rows, dtype=np.float32)

    y = np.array(all_labels, dtype=np.int32)
    return X, y, all_centers


def _run_inference(pipeline, ts, item_meta, magnification, patch_size,
                   model_type):
    """Apply trained model to the full slide, returning a 2-D label map."""
    meta = ts.getMetadata()
    native_mag = meta.get('magnification') or magnification
    slide_w = meta.get('sizeX', 0)
    slide_h = meta.get('sizeY', 0)

    # Scale patch_size to full-resolution pixels
    scale = native_mag / magnification if native_mag else 1.0
    step = int(round(patch_size * scale))

    cols = max(1, int(np.ceil(slide_w / step)))
    rows = max(1, int(np.ceil(slide_h / step)))

    print(f'Running inference: {cols}x{rows} patches at {magnification}x '
          f'(native {native_mag}x, step={step}px)...')

    # Build list of all patch centers (full-resolution coords)
    centers = []
    for row in range(rows):
        for col in range(cols):
            cx = int(col * step + step // 2)
            cy = int(row * step + step // 2)
            centers.append((cx, cy))

    batch_size = 256
    all_preds = []

    if model_type in ('trident_linear', 'trident_mlp'):
        # Use TRIDENT h5 embeddings for inference patches too
        for i in range(0, len(centers), batch_size):
            batch_centers = centers[i:i + batch_size]
            X_batch = get_patch_embeddings(None, item_meta, batch_centers,
                                           magnification, patch_size, model_type)
            preds = pipeline.predict(X_batch)
            all_preds.extend(preds.tolist())
            if (i // batch_size) % 20 == 0:
                print(f'  {i}/{len(centers)} patches processed')
    else:
        import torch
        import torchvision.transforms as T
        import torchvision.models as tvm

        if model_type == 'resnet_linear':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        else:
            backbone = None
            transform = None
            device = None

        for i in range(0, len(centers), batch_size):
            batch_centers = centers[i:i + batch_size]
            crops = get_patch_crops_from_ts(ts, batch_centers, magnification,
                                            patch_size)
            if model_type == 'resnet_linear':
                with torch.no_grad():
                    imgs = torch.stack(
                        [transform(c[:, :, :3]) for c in crops]
                    ).to(device)
                    X_batch = backbone(imgs).cpu().numpy().astype(np.float32)
            else:  # random_forest
                from skimage.feature import local_binary_pattern
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


def _update_session_metadata(gc, item_id, iteration, job_id,
                             model_item_id, overlay_annotation_id,
                             n_train_patches):
    """Append iteration record to _pixelClassifierSession in item metadata."""
    from datetime import datetime, timezone

    item = gc.get(f'item/{item_id}')
    session = ((item.get('meta') or {}).get('_pixelClassifierSession') or {})

    iterations = list(session.get('iterations') or [])

    # Find and update skeleton entry (stamped by frontend) or append new
    existing = next((it for it in iterations if it.get('num') == iteration), None)
    now = datetime.now(timezone.utc).isoformat()
    if existing:
        existing['job_id'] = job_id
        existing['model_item_id'] = model_item_id
        existing['overlay_annotation_id'] = overlay_annotation_id
        existing['trained_at'] = now
        existing['n_train_patches'] = n_train_patches
    else:
        iterations.append({
            'num': iteration,
            'job_id': job_id,
            'model_item_id': model_item_id,
            'overlay_annotation_id': overlay_annotation_id,
            'trained_at': now,
            'n_train_patches': n_train_patches,
        })

    session['iterations'] = iterations
    session['current_model_item_id'] = model_item_id
    session['updated_at'] = now

    gc.put(f'item/{item_id}/metadata',
           json={'_pixelClassifierSession': session})
    print(f'Updated _pixelClassifierSession on item {item_id} (iteration {iteration}).')


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
    if not args.annotation_id:
        raise RuntimeError('annotation_id is required.')
    if not args.job_dir:
        raise RuntimeError('job_dir is required.')

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    try:
        classes = json.loads(args.classes_json) if args.classes_json else []
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f'classes_json is not valid JSON: {exc}') from exc

    # classes may be empty when using an existing annotation — auto-detected later
    class_names = [c.get('name', f'Class {i + 1}') for i, c in enumerate(classes)]
    print(f'Session: {args.session_name!r}')
    print(f'Classes: {class_names}')
    print(f'Model type: {args.model_type}')
    print(f'Magnification: {args.magnification}x, patch size: {args.patch_size}px')

    os.makedirs(args.job_dir, exist_ok=True)

    # Fetch item metadata for TRIDENT h5 path
    item = gc.get(f'item/{args.item_id}')
    item_meta = item.get('meta') or {}

    # Read training annotation (supports pixelmap and polygon/shape elements)
    print(f'Reading annotation {args.annotation_id}...')
    patch_coords_per_class, ann_categories = read_annotation_as_training_data(
        gc, args.annotation_id, classes
    )
    # Sync classes from ann_categories in case they were auto-detected
    if not classes:
        classes = [{'name': c['label'],
                    'fillColor': c.get('fillColor', ''),
                    'strokeColor': c.get('strokeColor', '')}
                   for c in ann_categories]
        class_names = [c['name'] for c in classes]
        print(f'Classes (auto-detected): {class_names}')

    if not patch_coords_per_class:
        raise RuntimeError(
            'No labeled regions found in the training annotation. '
            'Add brush strokes or polygon annotations before training.'
        )

    # Open slide
    print('Opening slide...')
    item_files = list(gc.listFile(args.item_id, limit=1))
    if not item_files:
        raise RuntimeError(f'Item {args.item_id} has no files.')

    try:
        file_info = gc.get(f'file/{item_files[0]["_id"]}/download',
                           jsonResp=False, stream=True)
        slide_local = os.path.join(args.job_dir, item_files[0]['name'])
        with open(slide_local, 'wb') as fh:
            for chunk in file_info.iter_content(chunk_size=8192):
                fh.write(chunk)
        ts = large_image.open(slide_local)
    except Exception as exc:
        raise RuntimeError(f'Could not open slide: {exc}') from exc

    # If only one positive class, auto-sample background patches (class 0)
    if len(patch_coords_per_class) == 1:
        print('Single-class mode: sampling background patches automatically...')
        positive_centers = set(
            c for centers in patch_coords_per_class.values() for c in centers
        )
        patch_coords_per_class = _add_background_samples(
            patch_coords_per_class, positive_centers, ts,
            args.magnification, args.patch_size
        )
        # Prepend implicit Background to categories so index 0 is Background
        ann_categories = [
            {'label': 'Background',
             'fillColor': 'rgba(200,200,200,0.2)',
             'strokeColor': 'rgba(200,200,200,0.5)'}
        ] + ann_categories
        classes = [{'name': 'Background',
                    'fillColor': 'rgba(200,200,200,0.2)',
                    'strokeColor': 'rgba(200,200,200,0.5)'}] + classes
        class_names = ['Background'] + class_names

    # Build training data
    X, y, all_centers = _build_training_data(
        patch_coords_per_class, item_meta, ts,
        args.magnification, args.patch_size, args.model_type
    )

    # Per-class counts
    unique, counts = np.unique(y, return_counts=True)
    class_counts = {class_names[int(k)]: int(v)
                    for k, v in zip(unique, counts)
                    if int(k) < len(class_names)}
    n_train_patches = int(len(y))
    print(f'Training patches per class: {class_counts}')

    # Check minimum samples
    min_samples = min(counts)
    if min_samples < 5:
        raise RuntimeError(
            f'At least 5 patches per class required; '
            f'minimum found is {min_samples}. Add more annotations.'
        )

    # Train
    print('Training classifier...')
    pipeline = build_classifier(args.model_type, args.n_estimators,
                                args.random_seed)
    pipeline.fit(X, y)
    print('Training complete.')

    accuracy = compute_oob_or_cv_accuracy(pipeline, X, y, args.model_type)
    print(f'Accuracy estimate: {accuracy:.4f}' if accuracy else 'Accuracy: N/A')

    # Run inference
    print('Running full-slide inference...')
    # categories_for_annotation is already built from ann_categories (includes
    # implicit Background prepended in single-class mode)
    categories_for_annotation = ann_categories

    label_map, step = _run_inference(
        pipeline, ts, item_meta,
        args.magnification, args.patch_size, args.model_type
    )
    print(f'Inference complete. Label map shape: {label_map.shape}')

    # Post prediction overlay annotation
    ann_name = f'PixelClassifier Iteration {args.iteration} — {args.session_name}'
    print(f'Posting overlay annotation: {ann_name!r}...')
    overlay_id = post_pixelmap_annotation(
        gc, args.item_id, label_map, categories_for_annotation,
        ann_name, step, args.magnification, origin_xy=(0, 0)
    )
    print(f'Overlay annotation ID: {overlay_id}')

    # Upload model
    model_dict = {
        'model': pipeline,
        'embedder': args.model_type,
        'classes': class_names,
        'categories': classes,
        'patch_size': args.patch_size,
        'magnification': args.magnification,
        'session_name': args.session_name,
    }
    report_dict = {
        'oob_accuracy': accuracy,
        'class_counts': class_counts,
        'n_train_patches': n_train_patches,
        'model_type': args.model_type,
        'iteration': args.iteration,
    }

    model_item_id = upload_model_to_girder(
        gc, args.output_folder_id or None,
        args.session_name, model_dict, args.job_dir, report_dict
    )

    # Stamp metadata on the slide item
    _update_session_metadata(
        gc, args.item_id, args.iteration,
        job_id=None,  # job_id not available inside the CLI itself
        model_item_id=model_item_id,
        overlay_annotation_id=overlay_id,
        n_train_patches=n_train_patches,
    )

    print('BuildPixelClassifier complete.')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
