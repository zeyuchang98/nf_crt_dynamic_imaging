"""Verify compact outputs, inference-only GIFs, and unchanged evaluation scopes."""
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from motion_static.run import build_model, execute, main, parse_configuration, prepare_data
from motion_static.run_records import update_summary
from motion_static.evaluation.evaluate import evaluate, motion_scores
from motion_static.evaluation.block_evaluate import evaluate_blocks
from motion_static.evaluation.metrics import error_metrics
from motion_static.evaluation.render import main as render
from motion_static.training_monitor import TrainingMonitor


def small_config():
    _, config, _ = parse_configuration(['--steps', '1'])
    config.update(size=16, frames=9, rank=2, head_width=8, temporal_width=8,
                  monitor_samples=16, batch_size=8, log_interval=1)
    encoding = dict(levels=2, features=2, log2_hash_size=8, min_resolution=4, max_resolution=8)
    config.update(encoding=encoding, motion_encoding=dict(encoding))
    return config


def record_initial_model(model, coordinates, intensities, config, *, out, validation, selection=None):
    """Integration fixture: save monitoring outputs without any optimizer update."""
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    monitor = TrainingMonitor(out, coordinates, intensities, validation, config, optimizer)
    monitor.record(model, optimizer, 0, 0.)
    return [], 0., None


class CompactRunTests(unittest.TestCase):
    def test_dry_run_does_not_create_output_or_call_training(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch('motion_static.run.train') as train, redirect_stdout(io.StringIO()):
                config = main(['--dry-run', '--out', str(out)])
            train.assert_not_called()
            self.assertFalse(out.exists())
            self.assertEqual(config['output'], dict(gifs=False, predictions=False))
            self.assertEqual(config['checkpoint_steps'], [])

    def test_template_output_switches_are_explicit_and_block_defaults_are_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chosen.json'
            config = small_config()
            config['output'] = dict(gifs=True, predictions=True)
            path.write_text(json.dumps(config))
            _, resolved, _ = parse_configuration(['--config', str(path), '--no-gifs', '--no-save-predictions'])
            self.assertEqual(resolved['output'], dict(gifs=False, predictions=False))
            with redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    main(['--split', 'blocks', '--dry-run'])

    def test_compact_and_optional_outputs_have_identical_metrics(self):
        config = small_config()
        data, _ = prepare_data(config)
        model = build_model(config).eval()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            first, second = Path(directory) / 'compact', Path(directory) / 'export'
            first.mkdir()
            second.mkdir()
            evaluate(model, data, config, first, 0.)
            self.assertEqual([p.name for p in first.iterdir()], ['metrics.json'])
            config['output'] = dict(gifs=True, predictions=True)
            evaluate(model, data, config, second, 0.)
            self.assertEqual((first / 'metrics.json').read_bytes(), (second / 'metrics.json').read_bytes())
            self.assertEqual(sorted(p.name for p in (second / 'visuals').iterdir()),
                             ['image.gif', 'motion_effect.gif', 'motion_only.gif', 'reference_intensity.gif'])
            with np.load(second / 'predictions.npz') as arrays:
                expected = error_metrics(data['midpoint']['truth'], arrays['midpoint_image'], data_range=1)
                motion = motion_scores(data['midpoint']['u_true'], arrays['midpoint_motion'],
                                       data['midpoint']['foreground_mask'], 1 / 15)
            report = json.loads((first / 'metrics.json').read_text())
            self.assertEqual(report['image']['unseen_time_midpoints']['rrmse'], expected['rrmse'])
            self.assertAlmostEqual(report['motion']['midpoint']['foreground']['endpoint_rmse_pixels'],
                                   motion['foreground']['endpoint_rmse_pixels'], places=12)
            self.assertIn('ux', report['motion_components'])
            self.assertIn('uy', report['motion_components'])

    def test_block_evaluation_stays_separate_and_compact(self):
        config = small_config()
        config.update(frames=16, evaluation_split='blocks',
                      block_split=dict(block_shape=[8, 8, 2], test_fraction=.2, seed=0))
        data, split = prepare_data(config)
        model = build_model(config).eval()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)
            report = evaluate_blocks(model, data['training'], split, config, out, 0.)
            self.assertEqual([p.name for p in out.iterdir()], ['metrics.json'])
            self.assertIn('test_pixels', report['image'])
            self.assertNotIn('unseen_time_midpoints', report['image'])
            self.assertIn('reference_intensity_full_grid', report)
            self.assertEqual(report['split']['mask_sha256'], split['settings']['mask_sha256'])

    def test_flat_run_and_later_render_keep_monitor_and_metrics_unchanged(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            out = root / 'run'
            with patch('motion_static.run.train', side_effect=record_initial_model):
                execute(config, out)
            expected = {'config.json', 'metrics.json', 'loss.csv', 'loss.png', 'model.pt',
                        'monitor_samples.npz', 'source.zip', 'run.log'}
            self.assertEqual({p.name for p in out.iterdir()}, expected)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
            with patch('motion_static.run.train') as train:
                render([str(out), '--views', 'intensity', 'motion'])
                train.assert_not_called()
            self.assertEqual(sorted(p.name for p in (out / 'visuals').iterdir()),
                             ['motion_effect.gif', 'reference_intensity.gif'])
            for name, digest in hashes.items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            with np.load(out / 'monitor_samples.npz') as samples:
                self.assertEqual(set(samples.files), {'training_indices', 'validation_indices'})
                self.assertEqual(len(samples['training_indices']), 16)
            update_summary(root)
            with (root / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['evaluation_split'], 'midpoints')
            self.assertEqual(rows[0]['status'], 'complete')
            self.assertIn('Phantom: complex', (out / 'run.log').read_text())
            saved = json.loads((out / 'config.json').read_text())
            saved['rank'] = 3
            (out / 'config.json').write_text(json.dumps(saved))
            with patch('motion_static.evaluation.render.saved_evaluation_data') as data:
                with self.assertRaisesRegex(ValueError, 'config differs'):
                    render([str(out), '--views', 'image'])
                data.assert_not_called()


    def test_failure_is_recorded_without_overwriting_or_marking_complete(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
            out = Path(directory) / 'failed'
            with patch('motion_static.run.prepare_data', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    execute(config, out)
            saved = json.loads((out / 'config.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('fixture failure', (out / 'run.log').read_text())
            self.assertFalse((out / 'metrics.json').exists())
            with self.assertRaises(FileExistsError):
                execute(small_config(), out)


if __name__ == '__main__':
    unittest.main()
