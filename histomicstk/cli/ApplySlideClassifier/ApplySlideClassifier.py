import json
import os
import pickle

import numpy as np

from histomicstk.cli.utils import CLIArgumentParser


def _resolve_api_url(url):
    if not url:
        return url
    if url.startswith(('http://', 'https://')):
        return url
    return 'http://girder:8080/' + url.lstrip('/')


def _load_features_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        feats = f['features'][:]
    return feats


def _slide_feature_path(features_h5, slide_encoder):
    if not features_h5 or not slide_encoder:
        return None
    feat_dir = os.path.dirname(features_h5)
    coords_dir = os.path.dirname(feat_dir)
    stem = os.path.splitext(os.path.basename(features_h5))[0]
    candidate = os.path.join(coords_dir, f'slide_features_{slide_encoder}', f'{stem}.h5')
    return candidate if os.path.exists(candidate) else None


def main(args):
    api_url = _resolve_api_url(getattr(args, 'girderApiUrl', None))
    token = getattr(args, 'girderToken', None)
    if not api_url or not token:
        raise RuntimeError(
            'girderApiUrl/girderToken were not injected. '
            'Slicer CLI Web should populate these automatically.'
        )
    if not args.folder_id:
        raise RuntimeError('folder_id is required.')
    if not args.model_path:
        raise RuntimeError('model_path is required.')

    if not os.path.exists(args.model_path):
        raise RuntimeError(f'model_path not found: {args.model_path}')

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    with open(args.model_path, 'rb') as fh:
        bundle = pickle.load(fh)

    model = bundle['model']
    classes = bundle['classes']
    task_type = bundle.get('task_type', 'multilabel')
    feature_kind = bundle.get('feature_kind', 'mean_pool_patch')
    slide_encoder = bundle.get('slide_encoder', '')

    print(f'Loaded model: classes={classes}, task_type={task_type}, feature_kind={feature_kind}')

    if args.item_ids:
        id_set = set(i.strip() for i in args.item_ids.split(',') if i.strip())
        items = [gc.getItem(i) for i in id_set]
        print(f'Predicting on {len(items)} selected item(s).')
    else:
        items = list(gc.listItem(args.folder_id, limit=10000))
        print(f'Folder has {len(items)} item(s).')

    results = []
    skipped = 0
    for item in items:
        meta = item.get('meta') or {}
        trident_meta = meta.get('trident') or {}
        features_h5 = trident_meta.get('features_h5')
        if not features_h5:
            skipped += 1
            continue

        if feature_kind == 'slide':
            feat_path = _slide_feature_path(features_h5, slide_encoder)
            if not feat_path:
                print(f'  SKIP {item["name"]}: no slide-level features for "{slide_encoder}"')
                skipped += 1
                continue
        else:
            feat_path = features_h5
            if not os.path.exists(feat_path):
                print(f'  SKIP {item["name"]}: features_h5 path missing on disk')
                skipped += 1
                continue

        try:
            feats = _load_features_h5(feat_path)
        except Exception as e:
            print(f'  SKIP {item["name"]}: failed to load features: {e}')
            skipped += 1
            continue

        if feats.ndim == 2:
            vec = feats.mean(axis=0)
        elif feats.ndim == 1:
            vec = feats
        else:
            print(f'  SKIP {item["name"]}: unexpected feature shape {feats.shape}')
            skipped += 1
            continue

        X = vec.reshape(1, -1).astype(np.float32)

        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            pred_idx = model.predict(X)[0]
            proba = np.zeros(len(classes))
            if hasattr(pred_idx, '__len__'):
                for j, v in enumerate(pred_idx):
                    if v:
                        proba[j] = 1.0
            else:
                proba[int(pred_idx)] = 1.0

        prediction = {c: float(proba[i]) for i, c in enumerate(classes)}
        results.append({'item_id': item['_id'], 'name': item['name'], 'prediction': prediction})

        gc.addMetadataToItem(item['_id'], {'_slideClassifierPrediction': prediction})

    print(f'Predicted {len(results)} slides, skipped {skipped}.')

    if args.job_dir:
        os.makedirs(args.job_dir, exist_ok=True)
        out_path = os.path.join(args.job_dir, 'predictions.json')
        with open(out_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f'Wrote {out_path}')

    print('Apply complete.')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
