"""Shared utilities for BuildPixelClassifier and ApplyPixelClassifier CLIs."""
import base64
import zlib
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    ('rgba(255,80,80,0.4)', 'rgba(255,0,0,1)'),
    ('rgba(80,200,80,0.4)', 'rgba(0,160,0,1)'),
    ('rgba(80,80,255,0.4)', 'rgba(0,0,200,1)'),
    ('rgba(255,180,0,0.4)', 'rgba(200,120,0,1)'),
    ('rgba(180,80,220,0.4)', 'rgba(140,0,180,1)'),
    ('rgba(0,200,220,0.4)', 'rgba(0,160,180,1)'),
]


def _auto_detect_classes(elements):
    """Return a classes list derived from annotation element groups/categories."""
    # For pixelmap: use categories array
    pixelmap_el = next((el for el in elements if el.get('type') == 'pixelmap'), None)
    if pixelmap_el:
        cats = pixelmap_el.get('categories') or []
        return [{'name': c.get('label', f'Class {i + 1}'),
                 'fillColor': c.get('fillColor', _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)][0]),
                 'strokeColor': c.get('strokeColor', _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)][1])}
                for i, c in enumerate(cats)]

    # For shape elements: collect unique groups in order of first appearance
    seen = {}
    for el in elements:
        g = (el.get('group') or '').strip()
        if g and g not in seen:
            seen[g] = len(seen)

    if not seen:
        # No groups — treat everything as one class named after the annotation
        return [{'name': 'Positive',
                 'fillColor': _DEFAULT_COLORS[0][0],
                 'strokeColor': _DEFAULT_COLORS[0][1]}]

    return [{'name': name,
             'fillColor': _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)][0],
             'strokeColor': _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)][1]}
            for name, idx in sorted(seen.items(), key=lambda x: x[1])]


def read_annotation_as_training_data(gc, annotation_id, classes, ts_meta=None):
    """Download an annotation and return per-class patch center coordinates.

    Handles both pixelmap elements (from PixelClassifierPanel brush) and
    shape elements (polygon, polyline, circle, rectangle, ellipse) from any
    existing Girder annotation.

    When ``classes`` is empty, class names and colors are auto-detected from
    the annotation's element groups (for shapes) or categories (for pixelmaps).

    For shape elements the centroid of each shape is used as the patch center.
    All shapes are mapped to class 1 regardless of their group when only one
    class is defined; with multiple classes, shape ``group`` is matched to
    class names.

    Parameters
    ----------
    gc : girder_client.GirderClient
    annotation_id : str
    classes : list[dict]
        Class definitions from the session (``[{name, fillColor, ...}, ...]``).
        Index 0 in this list corresponds to category index 1 in the annotation.
    ts_meta : dict or None
        TileSource metadata (used for bounds checking). Optional.

    Returns
    -------
    patch_coords_per_class : dict[int, list[tuple[int, int]]]
        Maps 1-based category index to (slide_x, slide_y) centers.
    categories : list[dict]
        Category definitions suitable for posting a pixelmap annotation.
    """
    annotation = gc.get(f'annotation/{annotation_id}')
    elements = (annotation.get('annotation') or {}).get('elements') or []

    # Auto-detect classes from the annotation when none are provided
    if not classes:
        classes = _auto_detect_classes(elements)
        print(f'Auto-detected classes: {[c["name"] for c in classes]}')

    # Build category list
    categories = [
        {
            'label': c.get('name', f'Class {i + 1}'),
            'fillColor': c.get('fillColor', 'rgba(128,128,128,0.4)'),
            'strokeColor': c.get('strokeColor', 'rgba(128,128,128,1)'),
        }
        for i, c in enumerate(classes)
    ]

    # Map class name → 1-based index
    class_name_to_idx = {c.get('name', ''): i + 1 for i, c in enumerate(classes)}

    patch_coords_per_class = {}

    # Check for pixelmap elements first
    pixelmap_el = next((el for el in elements if el.get('type') == 'pixelmap'), None)
    if pixelmap_el is not None:
        return _read_pixelmap_element(pixelmap_el, patch_coords_per_class), categories

    # Fall back to shape elements
    shape_types = {'polygon', 'polyline', 'circle', 'rectangle', 'ellipse',
                   'point', 'arrow'}
    shape_elements = [el for el in elements if el.get('type') in shape_types]

    if not shape_elements:
        raise RuntimeError(
            f'Annotation {annotation_id} contains no pixelmap or shape elements. '
            'Check that the annotation has drawn regions.'
        )

    single_class = len(classes) == 1

    for el in shape_elements:
        el_type = el.get('type', '')
        group = el.get('group', '')

        # Determine class index
        if single_class:
            cat_idx = 1
        else:
            cat_idx = class_name_to_idx.get(group)
            if cat_idx is None:
                # Try case-insensitive match
                cat_idx = next(
                    (idx for name, idx in class_name_to_idx.items()
                     if name.lower() == group.lower()),
                    None
                )
            if cat_idx is None:
                print(f'  Skipping element with group {group!r}: '
                      f'no matching class (expected one of {list(class_name_to_idx)}).')
                continue

        cx, cy = _element_centroid(el, el_type)
        if cx is None:
            continue

        patch_coords_per_class.setdefault(cat_idx, []).append((int(cx), int(cy)))

    print(f'Loaded {sum(len(v) for v in patch_coords_per_class.values())} '
          f'shape centroids from annotation {annotation_id}.')
    return patch_coords_per_class, categories


def _element_centroid(el, el_type):
    """Return (x, y) centroid of a shape element, or (None, None) on error."""
    try:
        if el_type in ('polygon', 'polyline'):
            pts = el.get('points') or []
            if not pts:
                return None, None
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        if el_type == 'circle':
            c = el.get('center') or {}
            return c.get('x'), c.get('y')

        if el_type in ('rectangle', 'ellipse'):
            c = el.get('center') or {}
            return c.get('x'), c.get('y')

        if el_type in ('point', 'arrow'):
            c = el.get('center') or {}
            return c.get('x'), c.get('y')

    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return None, None


def _read_pixelmap_element(pixelmap_el, patch_coords_per_class):
    """Extract (class_idx → [(x,y)]) from a pixelmap element."""
    boundaries = pixelmap_el.get('boundaries') or {}
    origin_x = boundaries.get('x', 0)
    origin_y = boundaries.get('y', 0)
    bnd_width = boundaries.get('width', 0)
    bnd_height = boundaries.get('height', 0)

    raw_values = pixelmap_el.get('values')
    if isinstance(raw_values, str) and raw_values:
        compressed = base64.b64decode(raw_values)
        flat = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8)
    elif isinstance(raw_values, list):
        flat = np.array(raw_values, dtype=np.uint8)
    else:
        return patch_coords_per_class  # empty / uninitialised pixelmap

    if bnd_width > 0 and bnd_height > 0:
        label_map = flat.reshape(bnd_height, bnd_width)
    else:
        side = int(np.ceil(np.sqrt(len(flat))))
        label_map = flat[:side * side].reshape(side, side)

    for row in range(label_map.shape[0]):
        for col in range(label_map.shape[1]):
            cat = int(label_map[row, col])
            if cat == 0:
                continue
            slide_x = int(origin_x + col)
            slide_y = int(origin_y + row)
            patch_coords_per_class.setdefault(cat, []).append((slide_x, slide_y))

    return patch_coords_per_class


def read_pixelmap_annotation(gc, annotation_id):
    """Legacy wrapper — calls read_annotation_as_training_data with no classes."""
    annotation = gc.get(f'annotation/{annotation_id}')
    elements = (annotation.get('annotation') or {}).get('elements') or []
    pixelmap_el = next((el for el in elements if el.get('type') == 'pixelmap'), None)
    if pixelmap_el is None:
        raise RuntimeError(
            f'Annotation {annotation_id} contains no pixelmap element.'
        )
    coords = _read_pixelmap_element(pixelmap_el, {})
    categories = pixelmap_el.get('categories') or []
    boundaries = pixelmap_el.get('boundaries') or {}
    return coords, categories, boundaries


def post_pixelmap_annotation(
        gc, item_id, label_array, categories,
        annotation_name, patch_size, magnification,
        origin_xy=(0, 0)):
    """Encode a 2-D label map as a pixelmap annotation and POST it to Girder.

    Parameters
    ----------
    gc : girder_client.GirderClient
    item_id : str
    label_array : np.ndarray, shape (H, W), dtype uint8
        Per-patch class indices (0 = background/unlabeled).
    categories : list[dict]
        [{label, fillColor, strokeColor}, …]
    annotation_name : str
    patch_size : int
        Patch side length in pixels (used to compute slide-space boundaries).
    magnification : float
        Magnification at which patch_size applies (informational metadata).
    origin_xy : tuple[int, int]
        (x, y) top-left corner of the label map in full-resolution slide pixels.

    Returns
    -------
    annotation_id : str
    """
    h, w = label_array.shape
    flat = label_array.astype(np.uint8).flatten()
    compressed = zlib.compress(flat.tobytes(), level=6)
    encoded = base64.b64encode(compressed).decode('ascii')

    # boundaries in slide pixel space
    origin_x, origin_y = origin_xy
    slide_width = w * patch_size
    slide_height = h * patch_size

    element = {
        'type': 'pixelmap',
        'values': encoded,
        'categories': categories,
        'boundaries': {
            'x': origin_x,
            'y': origin_y,
            'width': slide_width,
            'height': slide_height,
        },
    }

    payload = {
        'annotation': {
            'name': annotation_name,
            'description': (
                f'Pixel classifier prediction — patch_size={patch_size}px '
                f'@ {magnification}x'
            ),
            'elements': [element],
        }
    }

    result = gc.post(f'annotation?itemId={item_id}', json=payload)
    return result['_id']


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_patch_embeddings(gc, item_meta, patch_centers, magnification,
                         patch_size, model_type):
    """Extract embeddings for a list of patch center coordinates.

    Primary path (model_type in {'trident_linear', 'trident_mlp'}):
        Load TRIDENT .h5 file from ``item_meta['trident']['features_h5']`` and
        look up the nearest pre-computed embedding for each patch center.

    Fallback path (model_type == 'resnet_linear'):
        Extract patch crops from the slide via large_image and encode with a
        pretrained ResNet-18 on GPU.

    Handcrafted path (model_type == 'random_forest'):
        Extract patch crops and compute color statistics + LBP features on CPU.

    Parameters
    ----------
    gc : girder_client.GirderClient
    item_meta : dict
        The Girder item's ``meta`` dict (for TRIDENT h5 path).
    patch_centers : list[tuple[int, int]]
        (slide_x, slide_y) coordinates of the center of each patch.
    magnification : float
    patch_size : int
    model_type : str

    Returns
    -------
    embeddings : np.ndarray, shape (N, D), dtype float32
    """
    if model_type in ('trident_linear', 'trident_mlp'):
        return _embeddings_from_trident_h5(item_meta, patch_centers)
    else:
        raise ValueError(
            f'model_type {model_type!r} requires a TileSource for feature extraction. '
            'Call get_patch_embeddings() only for TRIDENT-based model types. '
            'For resnet_linear/random_forest, extract crops via get_patch_crops_from_ts() '
            'and encode them inline.'
        )


def _embeddings_from_trident_h5(item_meta, patch_centers):
    """Load pre-computed TRIDENT embeddings and look up nearest patch."""
    import h5py

    trident_meta = (item_meta or {}).get('trident') or {}
    h5_path = trident_meta.get('features_h5')
    if not h5_path:
        raise RuntimeError(
            'No meta.trident.features_h5 found on item. '
            'Run TRIDENT embeddings first or choose a different model_type.'
        )

    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]           # (N_patches, D)
        coords = f['coords'][:]               # (N_patches, 2) — [x, y]

    # For each requested patch center, find the nearest stored coord
    coords_arr = coords.astype(np.float32)
    centers_arr = np.array(patch_centers, dtype=np.float32)  # (M, 2)

    # Vectorized nearest-neighbor via broadcasting
    diff = centers_arr[:, None, :] - coords_arr[None, :, :]  # (M, N, 2)
    dists = np.sum(diff ** 2, axis=-1)                        # (M, N)
    nearest_idx = np.argmin(dists, axis=-1)                   # (M,)

    return features[nearest_idx].astype(np.float32)


def get_patch_crops_from_ts(ts, patch_centers, magnification, patch_size):
    """Extract patch crops from a large_image TileSource.

    Parameters
    ----------
    ts : large_image TileSource
    patch_centers : list[tuple[int, int]]
        (x, y) in full-resolution slide pixels.
    magnification : float
    patch_size : int

    Returns
    -------
    crops : list[np.ndarray]  shape (patch_size, patch_size, C), dtype uint8
    """
    import large_image

    meta = ts.getMetadata()
    native_mag = meta.get('magnification') or magnification
    scale_factor = native_mag / magnification if native_mag else 1.0

    # Half-width in native pixels
    half = int(round(patch_size * scale_factor / 2))

    crops = []
    for cx, cy in patch_centers:
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        region_width = half * 2
        region_height = half * 2

        tile, _ = ts.getRegion(
            region=dict(left=x0, top=y0, width=region_width, height=region_height,
                        units='base_pixels'),
            scale=dict(magnification=magnification),
            format=large_image.constants.TILE_FORMAT_NUMPY,
        )
        if tile.shape[0] != patch_size or tile.shape[1] != patch_size:
            from PIL import Image
            import numpy as np
            pil = Image.fromarray(tile[:, :, :3])
            pil = pil.resize((patch_size, patch_size), Image.BILINEAR)
            tile = np.array(pil)

        crops.append(tile)

    return crops


# ---------------------------------------------------------------------------
# sklearn model builder
# ---------------------------------------------------------------------------

def build_classifier(model_type, n_estimators=100, random_seed=42):
    """Return an unfitted scikit-learn Pipeline for the requested model_type."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_type in ('trident_linear', 'resnet_linear'):
        clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_seed, n_jobs=-1
        )
    elif model_type == 'trident_mlp':
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=random_seed,
        )
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f'Unknown model_type: {model_type!r}')

    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def compute_oob_or_cv_accuracy(pipeline, X, y, model_type):
    """Return an OOB or cross-validation accuracy estimate."""
    from sklearn.model_selection import cross_val_score

    clf = pipeline.named_steps['clf']
    if model_type == 'random_forest' and hasattr(clf, 'oob_score_'):
        return float(clf.oob_score_)

    if len(X) < 10:
        return None

    n_splits = min(5, len(set(y)))
    if n_splits < 2:
        return None

    scores = cross_val_score(pipeline, X, y, cv=n_splits, scoring='accuracy', n_jobs=-1)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Girder result upload
# ---------------------------------------------------------------------------

def upload_model_to_girder(gc, output_folder_id, session_name, model_dict,
                           job_dir, report_dict):
    """Serialize model bundle, write files, upload to Girder.

    Returns
    -------
    model_item_id : str
    """
    import os
    import pickle

    if not output_folder_id:
        me = gc.get('user/me')
        private = gc.loadOrCreateFolder('Private', me['_id'], 'user')
        output_folder_id = private['_id']

    models_folder = gc.loadOrCreateFolder('Pixel Classifier Models', output_folder_id, 'folder')
    session_folder = gc.loadOrCreateFolder(
        session_name or 'session', models_folder['_id'], 'folder')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run_folder = gc.loadOrCreateFolder(f'run_{timestamp}', session_folder['_id'], 'folder')
    item = gc.createItem(parentFolderId=run_folder['_id'], name=f'model_{timestamp}')

    model_path = os.path.join(job_dir, 'model.pkl')
    with open(model_path, 'wb') as fh:
        pickle.dump(model_dict, fh)
    gc.uploadFileToItem(item['_id'], model_path)

    report_path = os.path.join(job_dir, 'training_report.json')
    with open(report_path, 'w') as fh:
        import json as _json
        _json.dump(report_dict, fh, indent=2)
    gc.uploadFileToItem(item['_id'], report_path)

    gc.addMetadataToItem(item['_id'], {
        'pixelClassifierModel': {
            'session_name': session_name,
            'classes': model_dict.get('classes', []),
            'model_type': model_dict.get('embedder'),
            'patch_size': model_dict.get('patch_size'),
            'magnification': model_dict.get('magnification'),
            'oob_accuracy': report_dict.get('oob_accuracy'),
            'completed_at': datetime.now(timezone.utc).isoformat(),
        }
    })

    print(f'Uploaded model to Girder item {item["_id"]}')
    return item['_id']
