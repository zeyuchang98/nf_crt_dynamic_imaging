"""Validate compact sweep records without requiring GIFs or prediction archives."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from ..hash_sweep import sha256, validate_config
from .render import load_run

METRICS = ('image_rrmse_percent', 'intensity_rrmse_percent', 'motion_rmse_pixels')


def verify_run(run, expected, source_hashes):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    if config.get('status') != 'complete':
        raise ValueError(f'Incomplete run requires inspection: {run}')
    validate_config(config, expected)
    for name in ('config.json', 'metrics.json', 'loss.csv', 'loss.png', 'model.pt',
                 'monitor_samples.npz', 'source.zip', 'run.log'):
        if not (run / name).is_file() or not (run / name).stat().st_size:
            raise ValueError(f'Missing required run record: {name}')
    if config['code_provenance']['source_sha256'] != source_hashes:
        raise ValueError('Run source differs from the prepared comparison')
    import hashlib
    with ZipFile(run / 'source.zip') as archive:
        if set(archive.namelist()) != set(source_hashes):
            raise ValueError('Source archive inventory differs')
        for name, expected_hash in source_hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                raise ValueError(f'Source archive hash differs: {name}')
    model, _ = load_run(run)
    reference = torch.tensor([[.13, -.21, config['reference_time']]])
    if model.motion(reference).abs().max().item() != 0:
        raise ValueError('Reference anchor check failed')
    report = json.loads((run / 'metrics.json').read_text())
    if report['evaluation_split'] != 'midpoints':
        raise ValueError('Sweep requires midpoint evaluation')
    return dict(status='passed', artifact_sha256={p.name: sha256(p) for p in run.iterdir() if p.is_file()})


def nondominated(rows):
    """Minimize the same three errors; ties remain on the frontier."""
    values = np.array([[row[key] for key in METRICS] for row in rows])
    return [row['run_id'] for i, row in enumerate(rows)
            if not np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))]
