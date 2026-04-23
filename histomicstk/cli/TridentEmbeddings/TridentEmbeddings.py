import glob
import json
import os

from histomicstk.cli.utils import CLIArgumentParser


def _or_none(value):
    """Return None for empty strings and zero integers used as sentinel values."""
    if isinstance(value, str):
        return value or None
    return value or None


def _parse_csv_list(value):
    """Split a comma-separated string into a list, or return None if empty."""
    if not value or not value.strip():
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def _resolve_api_url(url):
    """Ensure the Girder API URL is absolute.

    When slicer_cli_web injects girderApiUrl it should already be an absolute
    internal URL (http://girder:8080/api/v1).  If the browser passed a relative
    path (e.g. 'api/v1') the CLI container still needs a fully-qualified URL
    reachable from within the Docker network.
    """
    if not url:
        return url
    if url.startswith('http://') or url.startswith('https://'):
        return url
    # Relative URL received from the browser — resolve against the internal
    # Docker service name, which is always reachable from within the worker's
    # network namespace.
    return 'http://girder:8080/' + url.lstrip('/')


def _stage_slides_from_girder(item_ids, wsi_dir, api_url, token):
    """Download every file of each Girder item into a per-item subdir of wsi_dir.

    Returns a list of dicts, one per item, with keys:
      - item_id:  Girder item id
      - rel_path: representative slide file relative to wsi_dir (for the CSV)
      - stem:     basename of rep file without extension — the name TRIDENT
                  uses when writing per-slide outputs (geojson, h5, etc.)
    """
    import girder_client

    api_url = _resolve_api_url(api_url)

    if not api_url or not token:
        msg = (
            'slide_item_ids was provided but girderApiUrl/girderToken were not '
            'injected. Slicer CLI Web should populate these automatically.'
        )
        raise RuntimeError(msg)

    os.makedirs(wsi_dir, exist_ok=True)
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    staged = []
    for item_id in item_ids:
        item = gc.getItem(item_id)
        item_name = item.get('name') or item_id
        item_dir = os.path.join(wsi_dir, item_name)
        rep_file_id = (item.get('largeImage') or {}).get('fileId')

        files = list(gc.listFile(item_id))
        if not files:
            print(f'  WARNING: item {item_name} ({item_id}) has no files; skipping')
            continue

        if os.path.isdir(item_dir) and os.listdir(item_dir):
            print(f'  Skipping download for item {item_name}: {item_dir} already populated')
        else:
            print(f'  Downloading item {item_name} ({item_id}) -> {item_dir}')
            os.makedirs(item_dir, exist_ok=True)
            for f in files:
                dest = os.path.join(item_dir, f['name'])
                gc.downloadFile(f['_id'], dest)

        rep_name = next(
            (f['name'] for f in files if str(f['_id']) == str(rep_file_id)),
            files[0]['name'],
        )
        staged.append({
            'item_id': item_id,
            'rel_path': os.path.join(item_name, rep_name),
            'stem': os.path.splitext(os.path.basename(rep_name))[0],
        })

    return staged


def _geojson_to_histomicsui_annotation(geojson_path, name):
    """Convert a TRIDENT tissue-contour GeoJSON FeatureCollection to a
    HistomicsUI annotation dict.

    TRIDENT emits polygon coordinates in slide-pixel space at native
    magnification, which is what HistomicsUI expects.
    """
    with open(geojson_path) as fh:
        gj = json.load(fh)

    if gj.get('type') == 'FeatureCollection':
        features = gj.get('features') or []
    else:
        features = [gj]

    elements = []
    for feat in features:
        geom = feat.get('geometry') or {}
        gtype = geom.get('type')
        coords = geom.get('coordinates') or []
        if gtype == 'Polygon':
            rings = coords
        elif gtype == 'MultiPolygon':
            rings = [ring for poly in coords for ring in poly]
        else:
            continue
        for ring in rings:
            if len(ring) < 3:
                continue
            points = [[float(x), float(y), 0] for x, y in ring]
            elements.append({
                'type': 'polyline',
                'points': points,
                'closed': True,
                'fillColor': 'rgba(0, 0, 255, 0.25)',
                'lineColor': 'rgb(0, 0, 255)',
                'lineWidth': 2,
            })

    return {
        'name': name,
        'description': 'TRIDENT tissue segmentation',
        'elements': elements,
    }


def _delete_existing_annotations(gc, item_id, name):
    """Remove any annotations on item_id with the given name so re-runs don't
    accumulate duplicates. Silent on failure."""
    try:
        existing = gc.get('annotation', parameters={'itemId': item_id})
    except Exception:
        return
    for ann in existing or []:
        if (ann.get('annotation') or {}).get('name') == name:
            try:
                gc.delete(f'annotation/{ann["_id"]}')
            except Exception as e:
                print(f'    Failed to delete stale annotation {ann["_id"]}: {e}')


def _upload_results_to_girder(args, staged, api_url, token):
    """Post annotations, metadata stamps, and a job-summary folder back to Girder.

    `staged` is the list of dicts returned by _stage_slides_from_girder
    (or [] if slides weren't staged from Girder).

    Heavy artefacts (.h5 features, patches) stay on disk — their paths are
    stamped into source-slide item metadata so downstream CLIs (e.g., MIL)
    can read them directly from the shared mount.
    """
    api_url = _resolve_api_url(api_url)
    if not api_url or not token:
        print('  Skipping result upload: Girder API URL/token not available.')
        return

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.setToken(token)

    slide_item_map = {s['stem']: s['item_id'] for s in staged}
    job_dir = args.job_dir
    job_dir_name = os.path.basename(os.path.normpath(job_dir))
    coords_dir_name = args.coords_dir or (
        f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap'
    )
    coords_dir_abs = os.path.join(job_dir, coords_dir_name)
    geojson_dir = os.path.join(job_dir, 'contours_geojson')

    # --- 1. Tissue-segmentation annotations on source slides ---
    if slide_item_map and os.path.isdir(geojson_dir):
        print('Posting tissue-segmentation annotations to source slide items...')
        ann_name = 'TRIDENT tissue'
        for gj_name in sorted(os.listdir(geojson_dir)):
            if not gj_name.endswith('.geojson'):
                continue
            stem = os.path.splitext(gj_name)[0]
            item_id = slide_item_map.get(stem)
            if not item_id:
                print(f'  No matching slide item for {gj_name}; skipping.')
                continue
            try:
                annotation = _geojson_to_histomicsui_annotation(
                    os.path.join(geojson_dir, gj_name), name=ann_name,
                )
                if not annotation['elements']:
                    print(f'  {gj_name}: no polygons; skipping.')
                    continue
                _delete_existing_annotations(gc, item_id, ann_name)
                gc.post('annotation', parameters={'itemId': item_id}, json=annotation)
                print(f'  Posted {len(annotation["elements"])} element(s) to item {item_id}')
            except Exception as e:
                print(f'  Failed to post annotation for {gj_name}: {e}')

    # --- 2. Metadata stamps on source slide items (paths to on-disk artefacts) ---
    if slide_item_map:
        print('Stamping TRIDENT metadata onto source slide items...')
        for stem, item_id in slide_item_map.items():
            meta = {
                'job_dir': job_dir,
                'task': args.task,
                'mag': args.mag,
                'patch_size': args.patch_size,
                'overlap': args.overlap,
                'patch_encoder': args.patch_encoder,
                'slide_encoder': args.slide_encoder,
            }
            feat_candidates = glob.glob(
                os.path.join(coords_dir_abs, f'features_{args.patch_encoder}', f'{stem}.h5'))
            if feat_candidates:
                meta['features_h5'] = feat_candidates[0]
            patch_candidates = glob.glob(
                os.path.join(coords_dir_abs, 'patches', f'{stem}*.h5'))
            if patch_candidates:
                meta['coords_h5'] = patch_candidates[0]
            geojson_path = os.path.join(geojson_dir, f'{stem}.geojson')
            if os.path.exists(geojson_path):
                meta['tissue_geojson'] = geojson_path
            try:
                gc.addMetadataToItem(item_id, {'trident': meta})
                print(f'  Stamped metadata on item {item_id}')
            except Exception as e:
                print(f'  Failed to stamp metadata on item {item_id}: {e}')

    # --- 3. Job-summary folder + item (config + log only; points to on-disk dir) ---
    try:
        output_folder_id = _or_none(getattr(args, 'output_folder_id', None))
        if not output_folder_id:
            me = gc.get('user/me')
            if not me or not me.get('_id'):
                print('  Cannot resolve current user; skipping job-summary folder.')
                return
            private = gc.loadOrCreateFolder('Private', me['_id'], 'user')
            output_folder_id = private['_id']

        runs_folder = gc.loadOrCreateFolder('TRIDENT Runs', output_folder_id, 'folder')
        job_folder = gc.loadOrCreateFolder(job_dir_name, runs_folder['_id'], 'folder')
        # Stamp job_dir on the folder itself so the TRIDENT Cleanup plugin can
        # rm -rf the on-disk directory when this folder is deleted in the UI.
        gc.addMetadataToFolder(job_folder['_id'], {'trident': {'job_dir': job_dir}})
        summary_item = gc.createItem(
            parentFolderId=job_folder['_id'],
            name='job_summary', reuseExisting=True,
        )
        # Clear prior files so re-runs overwrite cleanly.
        for existing in gc.listFile(summary_item['_id']):
            try:
                gc.delete(f'file/{existing["_id"]}')
            except Exception:
                pass
        for fname in ('_config_segmentation.json', '_logs_segmentation.txt'):
            fpath = os.path.join(job_dir, fname)
            if os.path.exists(fpath):
                gc.uploadFileToItem(summary_item['_id'], fpath)
        summary_meta = {
            'job_dir': job_dir,
            'task': args.task,
            'patch_encoder': args.patch_encoder,
            'slide_encoder': args.slide_encoder,
            'mag': args.mag,
            'patch_size': args.patch_size,
            'overlap': args.overlap,
            'source_slide_item_ids': sorted(slide_item_map.values()),
        }
        gc.addMetadataToItem(summary_item['_id'], {'trident': summary_meta})
        print(
            f'Job summary: folder {job_folder["_id"]} / item {summary_item["_id"]}',
        )
    except Exception as e:
        print(f'  Failed to create job-summary folder/item: {e}')


def main(args):
    try:
        from trident import Processor
        from trident.segmentation_models.load import segmentation_model_factory
        from trident.patch_encoder_models.load import encoder_factory as patch_encoder_factory
        from trident.slide_encoder_models.load import encoder_factory as slide_encoder_factory
    except ImportError as e:
        msg = 'TRIDENT is not installed. Install it with: pip install /opt/TRIDENT'
        raise RuntimeError(msg) from e

    item_ids = _parse_csv_list(getattr(args, 'slide_item_ids', None))
    staged = []
    if item_ids:
        print(f'Staging {len(item_ids)} slide(s) from Girder into {args.wsi_dir}...')
        staged = _stage_slides_from_girder(
            item_ids=item_ids,
            wsi_dir=args.wsi_dir,
            api_url=getattr(args, 'girderApiUrl', None),
            token=getattr(args, 'girderToken', None),
        )

        if staged:
            os.makedirs(args.job_dir, exist_ok=True)
            csv_path = os.path.join(args.job_dir, '_slide_list.csv')
            with open(csv_path, 'w') as fh:
                fh.write('wsi\n')
                for s in staged:
                    fh.write(f'{s["rel_path"]}\n')
            args.custom_list_of_wsis = csv_path
            print(f'  Wrote slide list ({len(staged)} entries) to {csv_path}')

            # TRIDENT's WSIFactory routes .dcm to ImageWSI (PIL) in auto mode,
            # which cannot read DICOM WSI. Force OpenSlide (libopenslide >= 4.0
            # has DICOM support and auto-discovers sibling instances in the same
            # directory) when DICOM is present.
            if args.reader_type == 'auto' and any(
                    s['rel_path'].lower().endswith('.dcm') for s in staged):
                print('  Detected DICOM in staged slides; forcing reader_type=openslide')
                args.reader_type = 'openslide'

    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    seg_batch = args.seg_batch_size or args.batch_size
    feat_batch = args.feat_batch_size or args.batch_size

    reader_type = args.reader_type if args.reader_type != 'auto' else None
    max_workers = args.max_workers or None
    wsi_ext = _parse_csv_list(args.wsi_ext)
    custom_mpp_keys = _parse_csv_list(args.custom_mpp_keys)

    print('Initializing TRIDENT Processor...')
    processor = Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        wsi_ext=wsi_ext,
        skip_errors=args.skip_errors,
        custom_mpp_keys=custom_mpp_keys,
        custom_list_of_wsis=_or_none(args.custom_list_of_wsis),
        max_workers=max_workers,
        reader_type=reader_type,
        search_nested=args.search_nested,
    )
    print(f'  WSI source: {args.wsi_dir}')
    print(f'  Output dir: {args.job_dir}')
    print(f'  Task: {args.task}')
    print(f'  Device: {device}')

    task = args.task

    if task in ('seg', 'all'):
        print('Running tissue segmentation...')
        seg_model = segmentation_model_factory(
            args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        processor.run_segmentation_job(
            segmentation_model=seg_model,
            holes_are_tissue=not args.remove_holes,
            batch_size=seg_batch,
            device=device,
        )
        print('  Segmentation done.')

    if task in ('coords', 'all'):
        print(f'Running patch coordinate extraction (mag={args.mag}x, size={args.patch_size}px)...')
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            min_tissue_proportion=args.min_tissue_proportion,
            saveto=_or_none(args.coords_dir),
        )
        print('  Patch extraction done.')

    coords_dir = args.coords_dir or (
        f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap'
    )

    if task in ('feat', 'all'):
        print(f'Running feature extraction with {args.patch_encoder}...')
        ckpt = _or_none(args.patch_encoder_ckpt_path)
        patch_enc = patch_encoder_factory(args.patch_encoder, weights_path=ckpt)
        processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=patch_enc,
            device=device,
            saveas=args.saveas,
            batch_limit=feat_batch,
        )
        print('  Feature extraction done.')

        slide_enc_name = args.slide_encoder
        if slide_enc_name and slide_enc_name != 'none':
            print(f'Running slide-level encoding with {slide_enc_name}...')
            slide_ckpt = _or_none(args.slide_encoder_ckpt_path)
            slide_enc = slide_encoder_factory(slide_enc_name, weights_path=slide_ckpt)
            processor.run_slide_feature_extraction_job(
                slide_encoder=slide_enc,
                coords_dir=coords_dir,
                device=device,
                saveas=args.saveas,
                batch_limit=feat_batch,
            )
            print('  Slide encoding done.')

    processor.release()
    print('TRIDENT pipeline complete.')

    print('Uploading results to Girder...')
    _upload_results_to_girder(
        args, staged,
        api_url=getattr(args, 'girderApiUrl', None),
        token=getattr(args, 'girderToken', None),
    )


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
