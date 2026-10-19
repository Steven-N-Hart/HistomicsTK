import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_classifier_utils import (  # noqa: E402
    build_classifier,
    build_roi_mask,
    compute_oob_or_cv_accuracy,
    filter_coords_by_roi,
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


def _find_annotation_by_name(gc, item_id, annotation_name):
    """Return the first annotation on item_id whose name matches annotation_name."""
    anns = gc.get('annotation', parameters={'itemId': item_id, 'limit': 500})
    for ann in anns:
        if (ann.get('annotation') or {}).get('name', '') == annotation_name:
            return ann['_id']
    return None


class _RemoteSlideMetadata:
    """Lightweight proxy that satisfies ts.getMetadata() without downloading the file."""
    def __init__(self, meta):
        self._meta = meta

    def getMetadata(self):
        return self._meta


def _open_slide(gc, item_id, job_dir, model_type):
    """Return (local_path_or_None, ts_or_proxy).

    For TRIDENT models embeddings are pre-computed — fetch dimensions from the
    Girder large_image metadata API instead of downloading the whole file.
    For other models download the file and open a real TileSource.
    """
    if model_type in ('trident_linear', 'trident_mlp'):
        try:
            meta = gc.get(f'large_image/item/{item_id}/metadata')
            return None, _RemoteSlideMetadata(meta)
        except Exception as exc:
            print(f'    Warning: could not fetch large_image metadata for {item_id}: {exc}')
            return None, None

    import large_image
    item_files = list(gc.listFile(item_id, limit=1))
    if not item_files:
        return None, None
    fname = item_files[0]['name']
    local_path = os.path.join(job_dir, f'{item_id}_{fname}')
    if not os.path.exists(local_path):
        gc.downloadFile(item_files[0]['_id'], local_path)
    ts = large_image.open(local_path)
    return local_path, ts


def _collect_patches_from_slide(gc, item_id, item_meta, ts, annotation_name,
                                 roi_annotation_name, classes,
                                 magnification, patch_size, model_type):
    """Return (patch_coords_per_class, categories) for one slide.

    Downloads the named annotation, reads positive patch centers, optionally
    filters by ROI, and returns the coordinate dict ready for embedding extraction.
    """
    ann_id = _find_annotation_by_name(gc, item_id, annotation_name)
    if ann_id is None:
        print(f'    No annotation named {annotation_name!r} on item {item_id} — skipping.')
        return {}, []

    coords, categories = read_annotation_as_training_data(gc, ann_id, classes)
    if not coords:
        print(f'    Annotation {annotation_name!r} on item {item_id} has no elements — skipping.')
        return {}, categories

    # Apply ROI mask if specified
    roi_geom = None
    if roi_annotation_name:
        roi_ann_id = _find_annotation_by_name(gc, item_id, roi_annotation_name)
        if roi_ann_id:
            roi_geom = build_roi_mask(gc, roi_ann_id)

    if roi_geom is not None:
        for cat_idx in list(coords.keys()):
            filtered = filter_coords_by_roi(coords[cat_idx], roi_geom)
            if filtered:
                coords[cat_idx] = filtered
            else:
                del coords[cat_idx]

    n_pos = sum(len(v) for v in coords.values())
    print(f'    {n_pos} positive patch centers from {annotation_name!r}.')
    return coords, categories


def _extract_embeddings(patch_coords_per_class, item_meta, ts,
                        magnification, patch_size, model_type):
    """Extract embedding matrix and label vector for a coordinate dict."""
    all_centers = []
    all_labels = []
    for cat_idx, centers in sorted(patch_coords_per_class.items()):
        all_centers.extend(centers)
        all_labels.extend([cat_idx] * len(centers))

    if not all_centers:
        return np.empty((0, 1), dtype=np.float32), np.empty(0, dtype=np.int32)

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
            backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = torch.nn.Identity()
            backbone.eval()
            backbone.to(device)
            transform = T.Compose([
                T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
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
                feats_row = np.array([
                    r.mean(), r.std(), g.mean(), g.std(),
                    b.mean(), b.std(), s.mean(), s.std(),
                ])
                gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
                lbp = local_binary_pattern(
                    (gray * 255).astype(np.uint8), P=8, R=1, method='uniform')
                hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
                rows.append(np.concatenate([feats_row, hist]))
            X = np.array(rows, dtype=np.float32)

    return X, np.array(all_labels, dtype=np.int32)


def _add_background_for_slide(coords, ts, magnification, patch_size, roi_geom=None,
                              seed_offset=0):
    """Add class-0 background samples for a single-class slide."""
    meta = ts.getMetadata()
    native_mag = meta.get('magnification') or magnification
    slide_w = meta.get('sizeX', 1)
    slide_h = meta.get('sizeY', 1)
    scale = native_mag / magnification if native_mag else 1.0
    step = int(round(patch_size * scale))

    pos_set = {c for centers in coords.values() for c in centers}
    grid = []
    for row in range(0, slide_h, step):
        for col in range(0, slide_w, step):
            cx, cy = col + step // 2, row + step // 2
            if (cx, cy) not in pos_set:
                grid.append((cx, cy))

    if roi_geom is not None:
        grid = filter_coords_by_roi(grid, roi_geom)

    n_pos = sum(len(v) for v in coords.values())
    n_bg = min(n_pos * 3, len(grid))
    if n_bg == 0:
        return coords
    rng = np.random.default_rng(42 + seed_offset)
    chosen = rng.choice(len(grid), size=n_bg, replace=False)
    result = {0: [grid[i] for i in chosen]}
    result.update(coords)
    return result


def _run_inference_on_slide(pipeline, ts, item_meta, magnification, patch_size,
                            model_type, categories_for_ann, session_name, iteration,
                            gc, item_id):
    """Run full-slide inference and post overlay annotation. Returns annotation ID."""
    meta = ts.getMetadata()
    native_mag = meta.get('magnification') or magnification
    slide_w = meta.get('sizeX', 0)
    slide_h = meta.get('sizeY', 0)
    scale = native_mag / magnification if native_mag else 1.0
    step = int(round(patch_size * scale))
    cols = max(1, int(np.ceil(slide_w / step)))
    rows = max(1, int(np.ceil(slide_h / step)))

    centers = [(col * step + step // 2, row * step + step // 2)
               for row in range(rows) for col in range(cols)]

    batch_size = 256
    all_preds = []

    if model_type in ('trident_linear', 'trident_mlp'):
        for i in range(0, len(centers), batch_size):
            X_batch = get_patch_embeddings(None, item_meta, centers[i:i + batch_size],
                                           magnification, patch_size, model_type)
            all_preds.extend(pipeline.predict(X_batch).tolist())
    else:
        import torch
        import torchvision.models as tvm
        import torchvision.transforms as T

        if model_type == 'resnet_linear':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = torch.nn.Identity()
            backbone.eval()
            backbone.to(device)
            transform = T.Compose([
                T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            backbone = transform = device = None

        for i in range(0, len(centers), batch_size):
            batch = centers[i:i + batch_size]
            crops = get_patch_crops_from_ts(ts, batch, magnification, patch_size)
            if model_type == 'resnet_linear':
                with torch.no_grad():
                    imgs = torch.stack(
                        [transform(c[:, :, :3]) for c in crops]).to(device)
                    X_batch = backbone(imgs).cpu().numpy().astype(np.float32)
            else:
                from skimage.feature import local_binary_pattern
                rows_f = []
                for crop in crops:
                    rgb = crop[:, :, :3].astype(np.float32) / 255.0
                    r, g, b2 = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
                    cmax = np.maximum(r, np.maximum(g, b2))
                    cmin = np.minimum(r, np.minimum(g, b2))
                    s = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0)
                    fv = np.array([r.mean(), r.std(), g.mean(), g.std(),
                                   b2.mean(), b2.std(), s.mean(), s.std()])
                    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b2)
                    lbp = local_binary_pattern(
                        (gray * 255).astype(np.uint8), P=8, R=1, method='uniform')
                    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
                    rows_f.append(np.concatenate([fv, hist]))
                X_batch = np.array(rows_f, dtype=np.float32)
            all_preds.extend(pipeline.predict(X_batch).tolist())
            if (i // batch_size) % 20 == 0:
                print(f'  {i}/{len(centers)} patches processed')

    label_map = np.array(all_preds, dtype=np.uint8).reshape(rows, cols)
    ann_name = f'PixelClassifier Iteration {iteration} — {session_name}'
    overlay_id = post_pixelmap_annotation(
        gc, item_id, label_map, categories_for_ann,
        ann_name, step, magnification, origin_xy=(0, 0)
    )
    return overlay_id


def _update_folder_session_metadata(gc, folder_id, iteration, job_id,
                                    model_item_id, split_counts, metrics):
    """Append iteration record to _pixelClassifierSession in folder metadata."""
    from datetime import datetime, timezone
    folder = gc.get(f'folder/{folder_id}')
    session = ((folder.get('meta') or {}).get('_pixelClassifierSession') or {})
    iterations = list(session.get('iterations') or [])
    existing = next((it for it in iterations if it.get('num') == iteration), None)
    now = datetime.now(timezone.utc).isoformat()
    record = {
        'num': iteration,
        'job_id': job_id,
        'model_item_id': model_item_id,
        'trained_at': now,
        'split_counts': split_counts,
        'metrics': metrics,
    }
    if existing:
        existing.update(record)
    else:
        iterations.append(record)
    session['iterations'] = iterations
    session['current_model_item_id'] = model_item_id
    session['updated_at'] = now
    gc.put(f'folder/{folder_id}/metadata',
           json={'_pixelClassifierSession': session})
    print(f'Updated _pixelClassifierSession on folder {folder_id} (iteration {iteration}).')


def main(args):
    api_url = _resolve_api_url(getattr(args, 'girderApiUrl', None))
    token = getattr(args, 'girderToken', None)
    if not api_url or not token:
        raise RuntimeError('girderApiUrl/girderToken not injected.')
    if not args.folder_id:
        raise RuntimeError('folder_id is required.')
    if not args.positive_annotation_name:
        raise RuntimeError('positive_annotation_name is required.')
    if not args.job_dir:
        raise RuntimeError('job_dir is required.')

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    os.makedirs(args.job_dir, exist_ok=True)

    # Partition items into splits
    all_items = list(gc.listItem(args.folder_id, limit=10000))
    print(f'Folder has {len(all_items)} item(s).')

    split_items = {'train': [], 'test': [], 'val': []}
    for item in all_items:
        split = (item.get('meta') or {}).get('_pixelClassifierSplit')
        if split in split_items:
            split_items[split].append(item)

    for split, items in split_items.items():
        print(f'  {split}: {len(items)} slide(s)')

    if not split_items['train']:
        raise RuntimeError(
            'No slides assigned to Train split. '
            'Assign at least one slide to Train in the Pixel Classifier panel.'
        )

    # Collect patches from all slides per split
    classes = []
    ann_categories = []
    all_bg_prepended = False

    def _gather_split(items, split_label):
        nonlocal classes, ann_categories, all_bg_prepended
        X_parts, y_parts = [], []
        seed_offset = 0
        for item in items:
            item_id = item['_id']
            item_meta = item.get('meta') or {}
            print(f'  Processing {split_label} slide: {item["name"]}...')
            _, ts = _open_slide(gc, item_id, args.job_dir, args.model_type)
            if ts is None:
                print(f'    Could not open slide — skipping.')
                continue

            coords, cats = _collect_patches_from_slide(
                gc, item_id, item_meta, ts,
                args.positive_annotation_name,
                getattr(args, 'roi_annotation_name', ''),
                classes,
                args.magnification, args.patch_size, args.model_type
            )
            if not coords:
                continue

            # Auto-detect classes from first successful annotation
            if not classes and cats:
                classes[:] = [{'name': c.get('label', f'Class {i + 1}'),
                                'fillColor': c.get('fillColor', ''),
                                'strokeColor': c.get('strokeColor', '')}
                               for i, c in enumerate(cats)]
                ann_categories[:] = cats
                print(f'  Classes (auto-detected): {[c["name"] for c in classes]}')

            # Background sampling for single-class mode
            if len(coords) == 1 and split_label == 'train':
                roi_geom = None
                roi_name = getattr(args, 'roi_annotation_name', '')
                if roi_name:
                    roi_ann_id = _find_annotation_by_name(gc, item_id, roi_name)
                    if roi_ann_id:
                        roi_geom = build_roi_mask(gc, roi_ann_id)
                coords = _add_background_for_slide(
                    coords, ts, args.magnification, args.patch_size,
                    roi_geom=roi_geom, seed_offset=seed_offset
                )
                if not all_bg_prepended:
                    ann_categories[:] = [
                        {'label': 'Background',
                         'fillColor': 'rgba(200,200,200,0.2)',
                         'strokeColor': 'rgba(200,200,200,0.5)'}
                    ] + list(ann_categories)
                    classes[:] = [
                        {'name': 'Background',
                         'fillColor': 'rgba(200,200,200,0.2)',
                         'strokeColor': 'rgba(200,200,200,0.5)'}
                    ] + list(classes)
                    all_bg_prepended = True
                seed_offset += 1

            X, y = _extract_embeddings(
                coords, item_meta, ts,
                args.magnification, args.patch_size, args.model_type
            )
            if X.shape[0] > 0:
                X_parts.append(X)
                y_parts.append(y)

        if not X_parts:
            return np.empty((0, 1), dtype=np.float32), np.empty(0, dtype=np.int32)
        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

    print('\n=== Collecting Train patches ===')
    X_train, y_train = _gather_split(split_items['train'], 'train')
    print('\n=== Collecting Test patches ===')
    X_test, y_test = _gather_split(split_items['test'], 'test')
    print('\n=== Collecting Val patches ===')
    X_val, y_val = _gather_split(split_items['val'], 'val')

    if X_train.shape[0] == 0:
        raise RuntimeError(
            f'No training patches collected. Check that annotation '
            f'{args.positive_annotation_name!r} exists on Train slides.'
        )

    unique, counts = np.unique(y_train, return_counts=True)
    print(f'\nTrain class distribution: '
          f'{dict(zip(unique.tolist(), counts.tolist()))}')
    min_samples = int(counts.min())
    if min_samples < 5:
        raise RuntimeError(
            f'At least 5 train patches per class required; '
            f'minimum found is {min_samples}.'
        )

    # Train
    print('\n=== Training ===')
    pipeline = build_classifier(args.model_type, args.n_estimators, args.random_seed)
    pipeline.fit(X_train, y_train)
    print('Training complete.')

    train_acc = compute_oob_or_cv_accuracy(pipeline, X_train, y_train, args.model_type)
    print(f'Train accuracy: {train_acc:.4f}' if train_acc else 'Train accuracy: N/A')

    def _eval_split(X, y, name):
        if X.shape[0] == 0:
            return None
        from sklearn.metrics import accuracy_score
        acc = float(accuracy_score(y, pipeline.predict(X)))
        print(f'{name} accuracy: {acc:.4f}')
        return acc

    test_acc = _eval_split(X_test, y_test, 'Test')
    val_acc = _eval_split(X_val, y_val, 'Val')

    split_counts = {
        'train': int(X_train.shape[0]),
        'test': int(X_test.shape[0]),
        'val': int(X_val.shape[0]),
    }
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'val_accuracy': val_acc,
    }

    # Upload model
    class_names = [c.get('name', f'Class {i}') for i, c in enumerate(classes)]
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
        'split_counts': split_counts,
        'metrics': metrics,
        'model_type': args.model_type,
        'iteration': args.iteration,
        'positive_annotation_name': args.positive_annotation_name,
    }
    model_item_id = upload_model_to_girder(
        gc, args.output_folder_id or None,
        args.session_name, model_dict, args.job_dir, report_dict
    )

    # Generate overlay on all train slides as sanity check
    print('\n=== Generating overlays on Train slides ===')
    for item in split_items['train']:
        item_id = item['_id']
        item_meta = item.get('meta') or {}
        print(f'  Overlay on {item["name"]}...')
        _, ts = _download_slide(gc, item_id, args.job_dir)
        if ts is None:
            continue
        _run_inference_on_slide(
            pipeline, ts, item_meta,
            args.magnification, args.patch_size, args.model_type,
            ann_categories, args.session_name, args.iteration,
            gc, item_id
        )

    # Update folder session metadata
    _update_folder_session_metadata(
        gc, args.folder_id, args.iteration,
        job_id=None,
        model_item_id=model_item_id,
        split_counts=split_counts,
        metrics=metrics,
    )

    print('\nBuildPixelClassifier complete.')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
