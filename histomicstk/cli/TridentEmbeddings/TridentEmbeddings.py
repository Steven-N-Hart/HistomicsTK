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


def main(args):
    try:
        from trident import Processor
        from trident.segmentation_models.load import segmentation_model_factory
        from trident.patch_encoder_models.load import encoder_factory as patch_encoder_factory
        from trident.slide_encoder_models.load import encoder_factory as slide_encoder_factory
    except ImportError as e:
        msg = 'TRIDENT is not installed. Install it with: pip install /opt/TRIDENT'
        raise RuntimeError(msg) from e

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


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
