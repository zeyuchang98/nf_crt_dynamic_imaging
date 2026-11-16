"""Streamed source data: the source reader, frame shards and groups, partial evaluation sums,
the one-process equivalence with the in-memory run, and two gloo processes."""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

from motion_static.evaluation.metrics import (evaluate_sequence_in_blocks, sequence_block_partials,
                                              sequence_report_from_partials)
from motion_static.phantoms import moby, moby_source
from motion_static.phantoms.data import CoordinateGrid, VoxelSubsetGrid
from motion_static.phantoms.moby_slice import INDICATOR_TISSUES, extract_box
from motion_static.run import build_model, main
from motion_static.streaming import (StreamedSampler, StreamedTruth, frame_shards, groups_per_round,
                                     round_groups, scan_shard)
from motion_static.tests.test_moby_3d import config_3d


REPOSITORY = Path(__file__).resolve().parents[2]
SHAPE = (20, 20, 12)  # (x, y, z) of the source volume
CHUNKS = (10, 10, 6, 1)  # the source files' layout: spatial blocks, one frame per chunk
PHASES, FRAMES_PER_PHASE = 4, 6
BOX_START, BOX_SHAPE = (3, 2, 1), (16, 16, 9)  # run.py needs at least 16 per side
GEOMETRY = dict(fov=np.array([[0, 3.0], [0, 3.0], [1.0, 2.8]]), spacing=np.array([0.15] * 3), spacing_units="mm")
WINDOW = dict(first_frame=0, cycles=5, frames_per_cycle=4)  # 21 of the 24 frames


def write_source(root, zero_x=0):
    """Phase files in the real layout; phase p is 0.05 p s later, frames 10 s apart, so they interlace.

    Without zero_x almost every voxel is positive in some frame; with it, source x < zero_x
    is 0 in every frame (background outside a support mask)."""
    rng = np.random.default_rng(0)
    (root / "p0").mkdir()
    (root / "indicator").mkdir()
    for phase in range(PHASES):
        p0 = (rng.random((*SHAPE, FRAMES_PER_PHASE), dtype=np.float32) - 0.1).clip(0) * 0.4
        p0[:zero_x] = 0
        with h5py.File(root / "p0" / f"p0_{phase:05d}.h5", "w") as f:
            f.create_dataset("p0", data=p0, chunks=CHUNKS, compression="lzf")
            f["time"] = -36 + 0.05 * phase + 10.0 * np.arange(FRAMES_PER_PHASE)
            f["p0"].attrs.update(dict(GEOMETRY, resp_phase=phase, wavelength=800))
        with h5py.File(root / "indicator" / f"indicator_{phase:05d}.h5", "w") as f:
            for tissue in INDICATOR_TISSUES:
                f[tissue] = rng.random(SHAPE, dtype=np.float32)
                f[tissue].attrs.update(dict(GEOMETRY, phase=phase, dt=0.05))


def source_data(root, box=True):
    data = dict(WINDOW, source_root=str(root))
    if box:
        data.update(box_start=list(BOX_START), box_shape=list(BOX_SHAPE))
    return data


def source_config(root, **changes):
    config = config_3d(root, **{"steps": 4, **changes})
    config["data"] = dict(changes.get("data", source_data(root)))
    return config


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def process_environment(rank, world_size, port):
    return dict(os.environ, RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK="0",
                MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), PYTHONPATH=str(REPOSITORY),
                GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo", OMP_NUM_THREADS="1")


def run_processes(arguments, world_size=2):
    """Start `python arguments...` as `world_size` processes of one process group; wait for all."""
    port = free_port()
    processes = [subprocess.Popen([sys.executable, *arguments], cwd=REPOSITORY,
                                  env=process_environment(rank, world_size, port),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for rank in range(world_size)]
    outputs = [process.communicate(timeout=600)[0] for process in processes]
    for process, output in zip(processes, outputs):
        if process.returncode:
            raise AssertionError(f"Process failed with {process.returncode}:\n{output}")
    return outputs


def distributed_worker(task, out):
    """Run inside one process of a gloo group (see run_processes)."""
    from motion_static import distributed
    context = distributed.initialize("cpu")
    torch.manual_seed(0)
    config = dict(json.loads(Path(out, "config.json").read_text()))
    model = build_model(config)
    context.broadcast_parameters(model)
    if task == "gradients":
        # Each process: its own batch and weight; the summed gradient is saved by process 0.
        weights = [0.75, 0.25]
        context.weight = weights[context.rank]
        points, targets = torch.load(Path(out, f"batch{context.rank}.pt"))
        loss = (model(points) - targets).square().mean()
        (context.weight * loss).backward()
        context.all_reduce_gradients(model)
        if context.is_root:
            torch.save({name: p.grad for name, p in model.named_parameters()}, Path(out, "gradients.pt"))
    elif task == "divergence":
        if context.rank == 1:
            with torch.no_grad():
                next(model.parameters()).add_(1e-3)
        try:
            context.check_parameters(model)
            result = "agreed"
        except RuntimeError:
            result = "diverged"
        Path(out, f"divergence{context.rank}.txt").write_text(result)
    distributed.finalize(context)


class SourceReaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        write_source(self.root)

    def tearDown(self):
        self.folder.cleanup()

    def test_box_read_from_source_equals_the_extracted_box_file(self):
        box = extract_box(self.root, BOX_START, BOX_SHAPE, self.root / "box.h5")
        with moby_source.SourceFrames(self.root, BOX_START, BOX_SHAPE) as source, h5py.File(box, "r") as f:
            self.assertEqual(source.frame_count, PHASES * FRAMES_PER_PHASE)
            np.testing.assert_array_equal(source.times, f["time"][:])
            np.testing.assert_array_equal(source.phases, f["resp_phase"][:])
            np.testing.assert_array_equal(source.fov, f.attrs["fov"])
            for frame in range(source.frame_count):
                np.testing.assert_array_equal(source.read_raw(frame), f["p0"][..., frame])
        # Same run settings and the same normalized window as the extracted file.
        file_data, file_run = moby.resolve_settings(dict(WINDOW, file=str(box)))
        data, run = moby_source.resolve_settings(source_data(self.root))
        self.assertEqual(run, file_run)
        self.assertNotIn("intensity_scale_value", data)
        data["intensity_scale_value"] = file_data["intensity_scale_value"]
        expected, entry = moby.read_window_entry(file_data), moby.read_window_entry(data)
        for key in ("x", "y", "z", "t", "truth"):
            np.testing.assert_array_equal(entry[key], expected[key])

    def test_whole_volume_is_the_source_volume_with_its_own_field_of_view(self):
        with moby_source.SourceFrames(self.root) as source, h5py.File(self.root / "p0/p0_00002.h5", "r") as f:
            self.assertEqual(source.shape, SHAPE)
            np.testing.assert_array_equal(source.fov, GEOMETRY["fov"])
            frame = 3 * PHASES + 2  # fourth frame of phase 2 in the interlaced order
            np.testing.assert_array_equal(source.read_raw(frame), f["p0"][..., 3])
            np.testing.assert_array_equal(source.read(frame, 0.5), np.transpose(f["p0"][..., 3], (1, 0, 2)) / np.float32(0.5))

    def test_rejects_a_box_outside_the_volume_and_a_half_given_box(self):
        with self.assertRaisesRegex(ValueError, "do not fit"):
            moby_source.SourceFrames(self.root, (10, 0, 0), (16, 16, 8))
        with self.assertRaisesRegex(ValueError, "both"):
            moby_source.SourceFrames(self.root, BOX_START, None)


class ShardAndGroupTests(unittest.TestCase):
    def test_shards_are_contiguous_and_cover_every_frame_once(self):
        shards = frame_shards(10, 3)
        self.assertEqual([len(s) for s in shards], [4, 3, 3])
        np.testing.assert_array_equal(np.concatenate(shards), np.arange(10))
        with self.assertRaises(ValueError):
            frame_shards(2, 3)

    def test_each_round_partitions_the_shard_into_near_equal_groups(self):
        shard = np.arange(20, 31)
        rng = np.random.default_rng(0)
        self.assertEqual(groups_per_round([shard, np.arange(11)], 4), 3)
        self.assertEqual(groups_per_round([shard], None), 1)
        for _ in range(3):
            groups = round_groups(shard, 3, rng)
            self.assertEqual(sorted(len(g) for g in groups), [3, 4, 4])
            np.testing.assert_array_equal(np.sort(np.concatenate(groups)), shard)
        self.assertEqual(len(round_groups(shard, 1, rng)), 1)
        with self.assertRaisesRegex(ValueError, "empty"):
            groups_per_round([np.arange(5), np.arange(4)], 1)

    def test_sampler_loads_one_group_per_steps_per_group_and_draws_only_loaded_frames(self):
        x, y, z = np.arange(3.), np.arange(2.), np.arange(2.)
        times = np.linspace(0, 1, 9)
        # Frame `position` holds the value `position` everywhere, so targets name their frame.
        sampler = StreamedSampler(lambda k: np.full((2, 3, 2), k, dtype=np.float32), np.arange(9), (x, y, z), times,
                                  50, 3, 2, torch.Generator().manual_seed(0), np.random.default_rng(0), "cpu")
        loaded = []
        for step in range(1, 13):
            coordinates, targets = sampler.batch(step)
            if (step - 1) % 2 == 0:
                loaded.append(sampler.loaded.copy())
            self.assertTrue(set(targets.tolist()) <= set(sampler.loaded.tolist()))
            np.testing.assert_allclose(coordinates[:, 3].numpy(), times[targets.long().numpy()].astype(np.float32))
        self.assertEqual(len(loaded), 6)  # two rounds of three groups
        for first in (0, 3):
            np.testing.assert_array_equal(np.sort(np.concatenate(loaded[first:first + 3])), np.arange(9))

    def test_voxel_subset_rows_are_the_grid_rows_of_those_voxels(self):
        x, y, z, t = np.arange(3.), np.arange(2.) + 10, np.arange(4.) + 20, np.linspace(0, 1, 5)
        grid = CoordinateGrid(x, y, t, z)
        voxels = np.array([1, 4, 5, 11, 23])
        subset = VoxelSubsetGrid(x, y, t, z, voxels)
        self.assertEqual(len(subset), 5 * len(t))
        rows = torch.arange(len(subset))
        expected = (torch.as_tensor(voxels)[rows // len(t)] * len(t) + rows % len(t))
        self.assertTrue(torch.equal(subset[rows], grid[expected]))
        everything = VoxelSubsetGrid(x, y, t, z, np.arange(24))
        self.assertTrue(torch.equal(everything[torch.arange(len(grid))], grid[torch.arange(len(grid))]))

    def test_masked_sampler_draws_only_mask_voxels(self):
        x, y, z = np.arange(3.), np.arange(2.), np.arange(2.)
        voxels = np.array([0, 3, 7, 10])  # flat (y, x, z) indices
        # Frame k holds 100 k + voxel index, so a target names its voxel and frame.
        frames = lambda k: (100. * k + np.arange(12, dtype=np.float32)).reshape(2, 3, 2)
        sampler = StreamedSampler(frames, np.arange(4), (x, y, z), np.linspace(0, 1, 4), 200, 1, None,
                                  torch.Generator().manual_seed(0), np.random.default_rng(0), "cpu", voxels=voxels)
        coordinates, targets = sampler.batch(1)
        self.assertEqual(tuple(sampler.values.shape), (4, 4))
        voxel, frame = targets.long() % 100, targets.long() // 100
        self.assertTrue(set(voxel.tolist()) <= set(voxels.tolist()))
        grid = CoordinateGrid(x, y, np.linspace(0, 1, 4), z)
        self.assertTrue(torch.equal(coordinates, grid[voxel * 4 + frame]))

    def test_scan_finds_the_maximum_positive_voxels_and_sample_values(self):
        rng = np.random.default_rng(1)
        frames = (rng.random((5, 3, 4, 2), dtype=np.float32) - 0.5).clip(0)  # (x, y, z, frame)
        pool = np.transpose(frames, (1, 0, 2, 3)).reshape(-1)  # the in-memory order, frame fastest
        indices = rng.choice(pool.size, 10, replace=False)
        maximum, positive, values = scan_shard(lambda k: frames[..., k], [0, 1], 2, indices)
        self.assertEqual(maximum, float(frames.max()))
        np.testing.assert_array_equal(positive, np.transpose(frames, (1, 0, 2, 3)).max(axis=-1) > 0)
        np.testing.assert_array_equal(values, pool[indices])
        # Samples of a masked pool: index v * F + position names mask voxel voxels[v].
        voxels = np.flatnonzero(positive.reshape(-1))
        masked = pool.reshape(-1, 2)[voxels].reshape(-1)
        indices = rng.choice(masked.size, 10, replace=False)
        _, _, values = scan_shard(lambda k: frames[..., k], [0, 1], 2, indices, voxels)
        np.testing.assert_array_equal(values, masked[indices])


class PartialEvaluationTests(unittest.TestCase):
    def test_frame_ranges_combine_to_the_single_pass_report(self):
        rng = np.random.default_rng(2)
        truth = rng.random((9, 8, 8, 20)).astype(np.float32)
        prediction = (truth + 0.05 * rng.standard_normal(truth.shape)).astype(np.float32)
        times, roi = np.linspace(0, 1, 20), (slice(1, 8), slice(0, 7), slice(1, 8))

        def predict_frames(frames):
            return prediction[..., frames]

        single = evaluate_sequence_in_blocks(truth, predict_frames, data_range=1, times=times, roi=roi, block_frames=4)
        ranges = [(0, 7), (7, 14), (14, 20)]
        streamed = StreamedTruth(lambda k: truth[..., k], truth.shape[:-1], truth.shape[-1])
        partials = [sequence_block_partials(streamed, predict_frames, data_range=1, roi=roi, block_frames=4,
                                            frame_range=r) for r in ranges]
        combined = sequence_report_from_partials(partials, truth.shape, data_range=1, times=times, roi=roi,
                                                 block_frames=4)
        for scope in (single, single["roi"]):
            other = combined if scope is single else combined["roi"]
            for key in ("mse", "rrmse", "psnr_db", "ssim", "max_abs_error"):
                np.testing.assert_allclose(other[key], scope[key], rtol=1e-12)
            self.assertEqual(other["per_frame_rrmse"], scope["per_frame_rrmse"])
            self.assertEqual(other["per_frame_ssim"], scope["per_frame_ssim"])
        self.assertEqual(combined["activity_curve"]["truth"], single["activity_curve"]["truth"])
        np.testing.assert_allclose(combined["activity_curve"]["rrmse"], single["activity_curve"]["rrmse"], rtol=1e-12)
        with self.assertRaisesRegex(ValueError, "cover"):
            sequence_report_from_partials(partials[:2], truth.shape, data_range=1, times=times, roi=roi)


class StreamedRunTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        write_source(self.root)

    def tearDown(self):
        self.folder.cleanup()

    def run_config(self, config, name, distributed=False):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(config))
        out = self.root / "results" / name
        if distributed:
            run_processes(["-m", "motion_static.run", "--distributed", "--config", str(path), "--out", str(out)])
        else:
            main(["--config", str(path), "--out", str(out)])
        return out

    @staticmethod
    def read_loss(out):
        lines = (out / "loss.csv").read_text().splitlines()
        columns = lines[0].split(",")
        keep = [k for k, name in enumerate(columns) if name != "elapsed_seconds"]
        return [[line.split(",")[k] for k in keep] for line in lines]

    def test_one_process_streamed_run_reproduces_the_in_memory_run_of_the_extracted_box(self):
        box = extract_box(self.root, BOX_START, BOX_SHAPE, self.root / "box.h5")
        reference = config_3d(box, steps=4)
        reference["data"] = dict(WINDOW, file=str(box))
        expected = self.run_config(reference, "file")
        out = self.run_config(source_config(self.root), "source")

        saved = json.loads((out / "config.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["data_parallel"]["world_size"], 1)
        old = json.loads((expected / "config.json").read_text())
        self.assertEqual(saved["data"]["intensity_scale_value"], old["data"]["intensity_scale_value"])
        self.assertEqual(sorted(saved["data"]["source_sha256"]), [f"p0_{p:05d}.h5" for p in range(PHASES)])
        # Same batches, monitor samples and updates: loss.csv and the weights agree bit for bit.
        self.assertEqual(self.read_loss(out), self.read_loss(expected))
        state, old_state = (torch.load(path / "model.pt")["state_dict"] for path in (out, expected))
        for name in old_state:
            self.assertTrue(torch.equal(state[name], old_state[name]), name)
        metrics, old_metrics = (json.loads((path / "metrics.json").read_text()) for path in (out, expected))
        for report in (metrics, old_metrics):
            report.pop("training_seconds")
        self.assertEqual(metrics, old_metrics)

    def test_two_processes_with_frame_groups_complete_and_score_like_one_process(self):
        groups = dict(frames_per_group=4, steps_per_group=1)  # shards of 11 and 10 frames: 3 groups per round
        out = self.run_config(source_config(self.root, steps=6, frame_groups=groups), "two", distributed=True)
        saved = json.loads((out / "config.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["data_parallel"]["frame_shards"], [[0, 11], [11, 21]])
        self.assertEqual(saved["frame_groups"]["groups_per_round"], 3)
        self.assertEqual(saved["frame_groups"]["rounds"], 2)
        self.assertEqual(len(self.read_loss(out)), 1 + 7)  # header, step 0 and six logged steps
        # The two processes' partial sums give the one-process scores of the saved model.
        from motion_static.evaluation.evaluate import (roi_from_mask, streamed_embedding_partials,
                                                       streamed_embedding_report)
        from motion_static.evaluation.moby_embedding_figures import load_model
        model = load_model(out)
        data = saved["data"]
        window = moby.window_frames(data)
        with moby_source.open_source(data) as source:
            scale = data["intensity_scale_value"]
            truth = StreamedTruth(lambda k: source.read(window[k], scale), (BOX_SHAPE[1], BOX_SHAPE[0], BOX_SHAPE[2]),
                                  len(window))
            positive = np.any(np.stack([truth[..., k:k + 1][..., 0] > 0 for k in range(len(window))]), axis=0)
            roi = roi_from_mask(positive)
            entry = dict(x=source.x, y=source.y, z=source.z, t=source.times[window])
            partial = streamed_embedding_partials(model, entry, truth, (0, len(window)), roi)
            report = streamed_embedding_report(model, saved, 0., [partial], truth.shape, entry["t"], roi)
        image = json.loads((out / "metrics.json").read_text())["image"]["all_window_frames"]
        single = report["image"]["all_window_frames"]
        for key in ("mse", "rrmse", "psnr_db", "max_abs_error"):
            np.testing.assert_allclose(image[key], single[key], rtol=1e-10)
        self.assertEqual(image["per_frame_rrmse"], single["per_frame_rrmse"])

    def test_mask_of_every_voxel_reproduces_the_unmasked_run(self):
        expected = self.run_config(source_config(self.root), "plain")
        out = self.run_config(source_config(self.root, support_mask=True), "masked")
        details = json.loads((out / "config.json").read_text())["support_mask_details"]
        self.assertEqual(details["voxels"], int(np.prod(BOX_SHAPE)))
        self.assertEqual(self.read_loss(out), self.read_loss(expected))
        state, old_state = (torch.load(path / "model.pt")["state_dict"] for path in (out, expected))
        for name in old_state:
            self.assertTrue(torch.equal(state[name], old_state[name]), name)
        metrics, old_metrics = (json.loads((path / "metrics.json").read_text()) for path in (out, expected))
        for report in (metrics, old_metrics):
            report.pop("training_seconds")
        self.assertEqual(metrics, old_metrics)

    def test_configuration_errors(self):
        cases = [
            (dict(held_out_split="midpoints"), "held_out_split none"),
            (dict(frame_groups=dict(frames_per_group=4, steps_per_group=3)), "whole number of rounds"),
            (dict(frame_groups=dict(frames_per_group=4)), "positive integers"),
        ]
        box = extract_box(self.root, BOX_START, BOX_SHAPE, self.root / "box.h5")
        config = config_3d(box)
        config["data"] = dict(WINDOW, file=str(box))
        cases += [(config, "--distributed requires source data"),
                  (dict(config, frame_groups=dict(frames_per_group=4, steps_per_group=1)), "require source data"),
                  (dict(config, support_mask=True), "support_mask requires source data"),
                  (source_config(self.root, support_mask="yes"), "true or false")]
        for k, (changes, message) in enumerate(cases):
            path = self.root / f"bad{k}.json"
            path.write_text(json.dumps(changes if "data" in changes else source_config(self.root, **changes)))
            arguments = ["--config", str(path), "--out", str(self.root / f"bad{k}")]
            with captured_stderr() as errors, self.assertRaises(SystemExit):
                main(arguments + (["--distributed"] if "distributed" in message else []))
            self.assertIn(message, errors.getvalue())


class SupportMaskRunTests(unittest.TestCase):
    """Source data whose box has a background slab: the first 4 box columns (source x 3..6) are 0."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        write_source(self.root, zero_x=7)
        self.expected_mask = np.ones((BOX_SHAPE[1], BOX_SHAPE[0], BOX_SHAPE[2]), dtype=bool)  # (y, x, z)
        self.expected_mask[:, :4, :] = False

    def tearDown(self):
        self.folder.cleanup()

    def run_masked(self, name, distributed=False, **changes):
        return StreamedRunTests.run_config(self, source_config(self.root, support_mask=True, **changes), name,
                                           distributed=distributed)

    def test_masked_run_trains_inside_the_mask_and_scores_with_zero_outside(self):
        out = self.run_masked("masked")
        saved = json.loads((out / "config.json").read_text())
        count = int(self.expected_mask.sum())
        self.assertEqual(saved["support_mask_details"]["voxels"], count)
        np.testing.assert_array_equal(np.load(out / "support_mask.npz")["mask"], self.expected_mask)
        frames = len(moby.window_frames(saved["data"]))
        self.assertLess(int(np.load(out / "monitor_samples.npz")["training_indices"].max()), count * frames)

        # The saved scores equal the unmasked evaluation of the prediction set to 0 outside the mask.
        from motion_static.evaluation.evaluate import bounding_roi, predict_entry
        from motion_static.evaluation.moby_embedding_figures import load_model
        entry = moby.read_window_entry(saved["data"])
        prediction = predict_entry(load_model(out), entry)
        prediction[~self.expected_mask] = 0
        reference = evaluate_sequence_in_blocks(entry["truth"], lambda f: prediction[..., f], data_range=1,
                                                times=entry["t"], roi=bounding_roi([entry["truth"]]))
        image = json.loads((out / "metrics.json").read_text())["image"]["all_window_frames"]
        for key in ("mse", "rrmse", "psnr_db", "ssim", "max_abs_error"):
            np.testing.assert_allclose(image[key], reference[key], rtol=1e-6)
        np.testing.assert_allclose(image["per_frame_rrmse"], reference["per_frame_rrmse"], rtol=1e-6)
        np.testing.assert_allclose(image["activity_curve"]["prediction"], reference["activity_curve"]["prediction"],
                                   rtol=1e-6)

    def test_two_masked_processes_with_frame_groups_complete(self):
        out = self.run_masked("two", distributed=True, steps=6,
                              frame_groups=dict(frames_per_group=4, steps_per_group=1))
        saved = json.loads((out / "config.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["support_mask_details"]["voxels"], int(self.expected_mask.sum()))
        np.testing.assert_array_equal(np.load(out / "support_mask.npz")["mask"], self.expected_mask)


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.out = Path(self.folder.name)
        write_source(self.out)
        self.config = config_3d(self.out)
        self.config.update(coordinate_bounds=[[0, 1.2], [0, 1.2], [1, 2.35], [-36, -35]], reference_time=-36,
                           spatial_dimensions=3)
        (self.out / "config.json").write_text(json.dumps(self.config))

    def tearDown(self):
        self.folder.cleanup()

    def worker(self, task):
        run_processes(["-c", f"from motion_static.tests.test_streamed_training import distributed_worker; "
                             f"distributed_worker({task!r}, {str(self.out)!r})"])

    def test_summed_weighted_gradients_equal_the_single_process_gradient(self):
        generator = torch.Generator().manual_seed(3)
        batches = []
        for rank in range(2):
            points = torch.rand(40, 4, generator=generator) * torch.tensor([1.2, 1.2, 1.15, 1.]) + torch.tensor([0, 0, 1, -36])
            targets = torch.rand(40, generator=generator)
            torch.save((points, targets), self.out / f"batch{rank}.pt")
            batches.append((points, targets))
        self.worker("gradients")
        torch.manual_seed(0)
        model = build_model(self.config)
        loss = sum(w * (model(p) - t).square().mean() for w, (p, t) in zip([0.75, 0.25], batches))
        loss.backward()
        summed = torch.load(self.out / "gradients.pt")
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(summed[name], parameter.grad, rtol=1e-5, atol=1e-8, msg=name)

    def test_parameter_check_detects_diverged_processes(self):
        self.worker("divergence")
        self.assertEqual([(self.out / f"divergence{r}.txt").read_text() for r in range(2)], ["diverged"] * 2)


class ContiguousReductionTests(unittest.TestCase):
    def test_every_reduction_passes_contiguous_tensors_as_nccl_requires(self):
        from unittest import mock
        from motion_static import distributed

        def strict(tensor, *args, **kwargs):  # NCCL's check; gloo does not make it
            if not tensor.is_contiguous():
                raise ValueError("Tensors must be contiguous")

        context = distributed.Context(0, 2, 0, "gloo")
        model = torch.nn.Linear(3, 2)
        model(torch.ones(4, 3)).sum().backward()
        model.weight.grad = model.weight.grad.t().contiguous().t()  # a transposed gradient
        with mock.patch.object(distributed.dist, "all_reduce", strict), \
                mock.patch.object(distributed.dist, "broadcast", strict):
            context.any(np.ones((3, 4, 5), dtype=bool).transpose(1, 0, 2))
            context.sum_array(np.ones((3, 4), dtype=np.float32).T)
            context.all_reduce_gradients(model)
            context.broadcast_parameters(model)
            context.weighted_sum({"loss": 1.})
            context.maximum(1.)
            context.check_parameters(model)


class captured_stderr:
    """Collect argparse's error message while a SystemExit is expected."""

    def __enter__(self):
        self.stderr, sys.stderr = sys.stderr, io.StringIO()
        return sys.stderr

    def __exit__(self, *exc):
        sys.stderr = self.stderr


if __name__ == "__main__":
    unittest.main()
