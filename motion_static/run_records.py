"""Small shared helpers for flat run records and their derived CSV index."""
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Tee:
    """Keep terminal output visible while recording the same text in run.log."""
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def read_report(run):
    """Read the compact record, or an explicitly supplied historical run."""
    run = Path(run)
    path = run / 'metrics.json'
    if not path.exists():
        path = run / 'summary.json'
    report = json.loads(path.read_text())
    report['config'] = json.loads((run / 'config.json').read_text())
    return report


SUMMARY_FIELDS = (
    'run', 'status', 'phantom', 'model', 'evaluation_split', 'size', 'cycles', 'frames',
    'rank', 'steps', 'seed', 'motion_levels', 'motion_min_resolution', 'motion_max_resolution',
    'intensity_levels', 'intensity_min_resolution', 'intensity_max_resolution',
    'image_rrmse', 'image_psnr_db', 'image_ssim', 'intensity_rrmse',
    'intensity_psnr_db', 'intensity_ssim', 'motion_vector_rrmse',
    'motion_endpoint_rmse_pixels', 'motion_mean_endpoint_error_pixels',
    'fixed_training_mse', 'fixed_validation_mse', 'training_seconds', 'git_commit',
)


def summary_row(run, config, report=None):
    row = {key: config.get(key, '') for key in SUMMARY_FIELDS}
    row.update(run=run.name, status=config.get('status', 'unknown'),
               git_commit=config.get('code_provenance', {}).get('git_commit', ''))
    if config['model'] != 'nested_low_rank_hash_nf':
        row['rank'] = ''
    intensity = config.get('encoding', {})
    motion = {**intensity, **config.get('motion_encoding', {})}
    for branch, settings in (('intensity', intensity), ('motion', motion)):
        if config['model'] not in ('nested_hash_nf', 'nested_low_rank_hash_nf'):
            continue
        for key in ('levels', 'min_resolution', 'max_resolution'):
            value = settings.get(key, '')
            row[f'{branch}_{key}'] = json.dumps(value) if isinstance(value, list) else value
    if report is not None:
        blocks = report['evaluation_split'] == 'blocks'
        image = report['image']['test_pixels' if blocks else 'unseen_time_midpoints']
        intensity = (report.get('reference_intensity_full_grid', {}) if blocks else
                     report.get('reference_intensity', {}).get('unseen_time_midpoints', {}))
        motion = report.get('motion', {})
        motion = (motion if blocks else motion.get('midpoint', {})).get('foreground', {})
        for name, scores in (('image', image), ('intensity', intensity)):
            for key in ('rrmse', 'psnr_db', 'ssim'):
                row[f'{name}_{key}'] = scores.get(key, '')
        for key in ('vector_rrmse', 'endpoint_rmse_pixels', 'mean_endpoint_error_pixels'):
            row[f'motion_{key}'] = motion.get(key, '')
        row['training_seconds'] = report['training_seconds']
    if (run / 'loss.csv').exists():
        with (run / 'loss.csv').open() as stream:
            last = None
            for last in csv.DictReader(stream):
                pass
        if last:
            for key in ('fixed_training_mse', 'fixed_validation_mse'):
                row[key] = last.get(key, '')
    return row


def update_summary(root):
    """Regenerate summary.csv from flat child runs; never mix evaluation protocols."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Lock the directory itself so no extra lock file is left in results/.
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        rows = []
        for path in sorted(root.glob('*/config.json')):
            config = json.loads(path.read_text())
            report_path = path.parent / 'metrics.json'
            report = (json.loads(report_path.read_text())
                      if config.get('status') == 'complete' and report_path.exists() else None)
            rows.append(summary_row(path.parent, config, report))
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / 'summary.csv')
    finally:
        os.close(descriptor)
