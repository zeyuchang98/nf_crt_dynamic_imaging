"""Generate selected GIFs from a saved model; no optimization or metric changes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .evaluate import predict_fields
from .motion_visuals import frozen_true_motion
from .plots import save_comparison


def unchanged_inference_source(current, archived):
    """Allow changes only to evaluate(), which saved-model rendering never calls.

    All imports, globals, prediction functions and other helpers must retain
    identical syntax. Model/phantom/data sources still require exact hashes.
    """
    def inference_tree(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.FunctionDef) and node.name == 'evaluate')]
        return ast.dump(tree, include_attributes=False)
    return inference_tree(current) == inference_tree(archived)


def load_run(run):
    """Require the saved model/configuration and matching numerical source files."""
    from ..run import build_model
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    runtime = {'status', 'finished_at', 'error'}
    if ({k: v for k, v in config.items() if k not in runtime} !=
            {k: v for k, v in checkpoint['config'].items() if k not in runtime}):
        raise ValueError('Saved config differs from model.pt; use the original run configuration')
    root = Path(__file__).resolve().parents[2]
    for name, expected in config.get('code_provenance', {}).get('source_sha256', {}).items():
        numerical = (name.startswith(('motion_static/models/', 'motion_static/phantoms/')) or
                     name in ('motion_static/run.py', 'motion_static/splits.py', 'motion_static/evaluation/evaluate.py'))
        if not numerical:
            continue
        path = root / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        # A revised GIF call in evaluate() does not change checkpoint inference.
        # Verify that against the hash-authenticated snapshot, never bypass a
        # changed prediction function or silently trust a different source file.
        if name == 'motion_static/evaluation/evaluate.py' and path.exists() and (run / 'source.zip').exists():
            with ZipFile(run / 'source.zip') as archive:
                archived = archive.read(name)
            if hashlib.sha256(archived).hexdigest() == expected and unchanged_inference_source(path.read_bytes(), archived):
                continue
        raise ValueError(f'Numerical source changed: {name}. Use the saved source.zip in a separate checkout.')
    torch.set_num_threads(config.get('threads', 4))
    model = build_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config


def saved_evaluation_data(run, config):
    from ..run import prepare_data
    data, split = prepare_data(config)
    if split is None:
        return data['midpoint'], None
    # Keep the saved mask authoritative; never silently replace it on re-evaluation.
    with np.load(Path(run) / 'split.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['test_mask'], split['test_mask']):
            raise ValueError('Regenerated block mask differs from the saved mask')
        return data['training'], saved['test_mask'].copy()


def save_gifs(out, data, predictions, *, config, reference_time=None, views=None, error_mask=None):
    nested = 'motion' in predictions
    available = ['image'] + (['intensity', 'motion', 'motion_only'] if nested else [])
    views = available if views is None else views
    if any(view not in available for view in views):
        raise ValueError(f'Available views: {available}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for view in views:
        options = dict(save_still=False)
        if view == 'image':
            truth, prediction = data['truth'], predictions['image']
            filename, title = 'image.gif', 'Image: ground truth vs prediction'
            options['error_mask'] = error_mask
        elif view == 'intensity':
            truth, prediction = data['reference_intensity'], predictions['reference_intensity']
            filename, title = 'reference_intensity.gif', 'Intensity at fixed reference coordinates; no motion applied'
        elif view == 'motion':
            truth, prediction = frozen_true_motion(data, predictions['motion'], config, reference_time=reference_time)
            filename, title = 'motion_effect.gif', 'True and predicted motion | True intensity fixed at t0'
            options['labels'] = ('True motion', 'Predicted motion', 'Absolute image error')
            options['error_mask'] = error_mask
        else:
            truth, prediction = predictions['frozen_reference_intensity'], predictions['motion_only_frozen_intensity']
            filename, title = 'motion_only.gif', 'Learned intensity frozen at the motion reference time'
            options['labels'] = ('Fixed learned intensity', 'Warped learned intensity', 'Absolute change (not error)')
        save_comparison(out / filename, truth, prediction, data['t'], title, **options)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--views', nargs='+', choices=('image', 'intensity', 'motion', 'motion_only'))
    args = parser.parse_args(argv)
    model, config = load_run(args.run)
    data, mask = saved_evaluation_data(args.run, config)
    predictions = predict_fields(model, data, config, include_visuals=True)
    t0 = float(model.motion.reference_time) if hasattr(model, 'motion') else None
    out = save_gifs(args.run / 'visuals', data, predictions, config=config,
                    reference_time=t0, views=args.views, error_mask=mask)
    print(f'Saved requested GIFs: {out}')


if __name__ == '__main__':
    main()
