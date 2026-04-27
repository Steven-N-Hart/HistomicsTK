import csv
import json
import os
import pickle
from datetime import datetime, timezone

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


def _gather_dataset(gc, folder_id, classes, feature_kind, slide_encoder):
    items = list(gc.listItem(folder_id, limit=10000))
    print(f'Folder {folder_id} has {len(items)} item(s).')

    rows = []
    for item in items:
        meta = item.get('meta') or {}
        split = meta.get('_aiSplit')
        if split not in ('train', 'val', 'test'):
            continue

        # Missing labels → all-negative (slide is a negative example for every class)
        labels_dict = meta.get('_slideClassifierLabels') or {}

        trident_meta = meta.get('trident') or {}
        features_h5 = trident_meta.get('features_h5')
        if not features_h5:
            print(f'  SKIP {item["name"]}: no meta.trident.features_h5')
            continue

        if feature_kind == 'slide':
            feat_path = _slide_feature_path(features_h5, slide_encoder)
            if not feat_path:
                print(f'  SKIP {item["name"]}: no slide-level features for encoder "{slide_encoder}"')
                continue
        else:
            feat_path = features_h5
            if not os.path.exists(feat_path):
                print(f'  SKIP {item["name"]}: features_h5 path missing on disk')
                continue

        rows.append({
            'item_id': item['_id'],
            'name': item['name'],
            'split': split,
            'feat_path': feat_path,
            'labels_dict': labels_dict,
        })

    print(f'Collected {len(rows)} usable item(s).')

    X, Y, names, splits, item_ids = [], [], [], [], []
    for row in rows:
        feats = _load_features_h5(row['feat_path'])
        if feats.ndim == 2:
            vec = feats.mean(axis=0)
        elif feats.ndim == 1:
            vec = feats
        else:
            print(f'  SKIP {row["name"]}: unexpected feature shape {feats.shape}')
            continue

        label_vec = [1 if row['labels_dict'].get(c) else 0 for c in classes]
        X.append(vec)
        Y.append(label_vec)
        names.append(row['name'])
        splits.append(row['split'])
        item_ids.append(row['item_id'])

    if not X:
        raise RuntimeError(
            'No usable items found. Ensure slides have _aiSplit set, '
            '_slideClassifierLabel assigned, and TRIDENT embeddings computed.'
        )

    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.int8), names, splits, item_ids


def _build_model(model_type, task_type, regularization, max_iter, random_seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_type == 'mlp':
        base = MLPClassifier(hidden_layer_sizes=(256,), max_iter=max_iter, random_state=random_seed)
    else:
        base = LogisticRegression(C=regularization, max_iter=max_iter, random_state=random_seed)

    if task_type == 'multilabel':
        clf = OneVsRestClassifier(base)
    else:
        clf = base

    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def _per_target_metrics(y_true, y_score, y_pred, class_names):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    out = {}
    for i, name in enumerate(class_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        col_score = y_score[:, i] if y_score is not None else None
        try:
            auc = (roc_auc_score(col_true, col_score)
                   if col_score is not None and len(set(col_true)) > 1 else None)
        except Exception:
            auc = None
        out[name] = {
            'support': int(col_true.sum()),
            'accuracy': float(accuracy_score(col_true, col_pred)),
            'precision': float(precision_score(col_true, col_pred, zero_division=0)),
            'recall': float(recall_score(col_true, col_pred, zero_division=0)),
            'f1': float(f1_score(col_true, col_pred, zero_division=0)),
            'auc': auc,
        }
    return out


def _evaluate(model, X, Y, class_names, task_type):
    if X.shape[0] == 0:
        return {}

    if task_type == 'multilabel':
        y_pred = model.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        try:
            y_score = model.predict_proba(X)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)
        except Exception:
            y_score = None
        return _per_target_metrics(Y, y_score, y_pred, class_names)

    y_true_idx = Y.argmax(axis=1)
    y_pred_idx = model.predict(X)
    n = len(class_names)
    y_pred_oh = np.eye(n, dtype=np.int8)[y_pred_idx]
    y_true_oh = np.eye(n, dtype=np.int8)[y_true_idx]
    try:
        y_score = model.predict_proba(X)
    except Exception:
        y_score = None
    return _per_target_metrics(y_true_oh, y_score, y_pred_oh, class_names)


def _write_predictions_csv(path, names, splits, item_ids, Y, predictions, class_names):
    headers = ['item_id', 'name', 'split']
    for c in class_names:
        headers.append(f'true_{c}')
    for c in class_names:
        headers.append(f'pred_{c}')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for i in range(len(names)):
            row = [item_ids[i], names[i], splits[i]]
            row.extend(int(v) for v in Y[i])
            row.extend(int(v) for v in predictions[i])
            w.writerow(row)


def _post_results(gc, output_folder_id, exp_name, job_dir, summary):
    if not output_folder_id:
        me = gc.get('user/me')
        if not me or not me.get('_id'):
            print('  Cannot resolve current user; skipping result upload.')
            return None
        private = gc.loadOrCreateFolder('Private', me['_id'], 'user')
        output_folder_id = private['_id']

    runs_folder = gc.loadOrCreateFolder('Slide Classifier Runs', output_folder_id, 'folder')
    exp_folder = gc.loadOrCreateFolder(exp_name or 'experiment', runs_folder['_id'], 'folder')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    item = gc.createItem(parentFolderId=exp_folder['_id'], name=f'run_{timestamp}')

    for fname in ('metrics.json', 'predictions.csv', 'model.pkl'):
        fpath = os.path.join(job_dir, fname)
        if os.path.exists(fpath):
            gc.uploadFileToItem(item['_id'], fpath)

    gc.addMetadataToItem(item['_id'], {'slideClassifierRun': summary})
    print(f'Uploaded results to item {item["_id"]} (folder {exp_folder["_id"]}).')
    return item


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
    if not args.job_dir:
        raise RuntimeError('job_dir is required.')

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    folder = gc.getFolder(args.folder_id)
    experiment = (folder.get('meta') or {}).get('_aiExperiment')
    if not experiment:
        raise RuntimeError(
            f'Folder {args.folder_id} has no _aiExperiment metadata. '
            'Configure the Slide Classifier panel first.'
        )

    classes = experiment.get('classes') or []
    if not classes:
        raise RuntimeError('Experiment has no classes configured.')

    print(f'Experiment: {experiment.get("name") or "(unnamed)"}')
    print(f'  Classes: {classes}')
    print(f'  Feature kind: {args.feature_kind}')
    print(f'  Model: {args.model_type} ({args.task_type})')

    os.makedirs(args.job_dir, exist_ok=True)

    X, Y, names, splits, item_ids = _gather_dataset(
        gc, args.folder_id, classes, args.feature_kind, args.slide_encoder)

    splits_arr = np.array(splits)
    train_mask = splits_arr == 'train'
    val_mask = splits_arr == 'val'
    test_mask = splits_arr == 'test'

    print(f'Splits: train={int(train_mask.sum())} val={int(val_mask.sum())} '
          f'test={int(test_mask.sum())}')

    if train_mask.sum() < 2:
        raise RuntimeError(
            f'At least 2 labeled training slides are required; '
            f'found {int(train_mask.sum())}.'
        )

    if args.task_type == 'multiclass':
        sums = Y[train_mask].sum(axis=1)
        if not np.all(sums == 1):
            bad = [names[i] for i, s in enumerate(sums[train_mask]) if s != 1]
            raise RuntimeError(
                f'multiclass requires exactly one label per slide. '
                f'{len(bad)} training slide(s) violate this (e.g. {bad[:3]}).'
            )

    model = _build_model(
        args.model_type, args.task_type, args.regularization,
        args.max_iter, args.random_seed,
    )

    print('Training...')
    if args.task_type == 'multiclass':
        model.fit(X[train_mask], Y[train_mask].argmax(axis=1))
    else:
        model.fit(X[train_mask], Y[train_mask])

    print('Predicting on all splits...')
    if args.task_type == 'multiclass':
        all_pred_idx = model.predict(X)
        n = len(classes)
        all_pred = np.eye(n, dtype=np.int8)[all_pred_idx]
    else:
        all_pred = model.predict(X)
        if all_pred.ndim == 1:
            all_pred = all_pred.reshape(-1, 1)

    metrics = {
        'train': _evaluate(model, X[train_mask], Y[train_mask], classes, args.task_type),
        'val': _evaluate(model, X[val_mask], Y[val_mask], classes, args.task_type),
        'test': _evaluate(model, X[test_mask], Y[test_mask], classes, args.task_type),
    }

    metrics_path = os.path.join(args.job_dir, 'metrics.json')
    with open(metrics_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(f'Wrote {metrics_path}')

    pred_path = os.path.join(args.job_dir, 'predictions.csv')
    _write_predictions_csv(pred_path, names, splits, item_ids, Y, all_pred, classes)
    print(f'Wrote {pred_path}')

    model_path = os.path.join(args.job_dir, 'model.pkl')
    with open(model_path, 'wb') as fh:
        pickle.dump({
            'model': model,
            'classes': classes,
            'task_type': args.task_type,
            'feature_kind': args.feature_kind,
            'slide_encoder': args.slide_encoder,
        }, fh)
    print(f'Wrote {model_path}')

    summary = {
        'experiment_name': experiment.get('name'),
        'classes': classes,
        'feature_kind': args.feature_kind,
        'slide_encoder': args.slide_encoder if args.feature_kind == 'slide' else None,
        'model_type': args.model_type,
        'task_type': args.task_type,
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'n_test': int(test_mask.sum()),
        'source_folder_id': args.folder_id,
        'metrics': metrics,
        'completed_at': datetime.now(timezone.utc).isoformat(),
    }

    _post_results(gc, args.output_folder_id or None,
                  experiment.get('name'), args.job_dir, summary)

    print('Training complete.')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
