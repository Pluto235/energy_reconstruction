from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from apply.tools.build_v6_poisson_pooling_manifest import (
    build_pooling_manifest,
    choose_one_standard_error,
    cross_validate_orders,
    make_grid_edges,
    manifest_payload_sha256,
    rectangle_basis_grid,
    scan_continuous_annulus_maps,
    write_self_hashed_manifest,
)


class PoolingTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = {1: 25_000, 2: 8_000, 3: 1_500, 4: 50}
        self.centers = {1: 2.25, 2: 2.75, 3: 3.125, 4: 3.375}

    def test_independent_and_sparse_tiers_obey_hard_thresholds(self) -> None:
        manifest = build_pooling_manifest(
            self.counts,
            self.centers,
            target_ids=(1, 3, 4),
            excluded_tail_cell_ids=(),
        )
        self.assertEqual(manifest[1]["mode"], "independent")
        self.assertEqual(manifest[1]["donor_cell_ids"], [1])
        # Cells 2+3+4 total only 9,550, so the 10k contract requires cell 1 too.
        self.assertEqual(manifest[3]["donor_cell_ids"], [1, 2, 3, 4])
        self.assertGreaterEqual(manifest[3]["pooled_continuous_annulus_count"], 10_000)
        self.assertFalse(manifest[4]["shape_contributor"])

    def test_neighbors_never_cross_nhit(self) -> None:
        nhit = {1: "low", 2: "low", 3: "high", 4: "high"}
        manifest = build_pooling_manifest(
            self.counts,
            self.centers,
            target_ids=(3,),
            nhit_by_cell=nhit,
            excluded_tail_cell_ids=(),
        )
        self.assertEqual(manifest[3]["donor_cell_ids"], [3, 4])
        self.assertEqual(manifest[3]["mode"], "shared_plane_fallback")

    def test_lower_pred_e_wins_equal_distance_tie(self) -> None:
        counts = {1: 6_000, 2: 1_000, 3: 6_000}
        centers = {1: 2.0, 2: 3.0, 3: 4.0}
        manifest = build_pooling_manifest(
            counts,
            centers,
            target_ids=(2,),
            excluded_tail_cell_ids=(),
        )
        self.assertEqual(manifest[2]["donor_cell_ids"], [1, 2, 3])
        self.assertEqual(list(manifest[2]["shape_contributor_by_donor"]), ["1", "2", "3"])

    def test_excluded_tail_never_appears(self) -> None:
        counts = {1: 1_000, 13: 50_000, 14: 20_000}
        centers = {1: 2.0, 13: 6.5, 14: 3.0}
        manifest = build_pooling_manifest(counts, centers, target_ids=(1,))
        self.assertNotIn(13, manifest[1]["donor_cell_ids"])


class ManifestHashTests(unittest.TestCase):
    def test_written_manifest_validates_its_canonical_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            written = write_self_hashed_manifest(path, {"schema_version": 1, "cells": {"1": {}}})
            loaded = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(written, loaded)
        self.assertEqual(loaded["manifest_sha256"], manifest_payload_sha256(loaded))

    def test_non_finite_values_are_rejected_as_non_standard_json(self) -> None:
        with self.assertRaises(ValueError):
            manifest_payload_sha256({"score": float("inf")})


class CrossValidationTests(unittest.TestCase):
    def test_uniform_surface_sector_folds_use_disjoint_exposure_masks(self) -> None:
        edges = make_grid_edges(0.5, radius_deg=2.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        x_grid, y_grid = np.meshgrid(centers, centers)
        rho = np.hypot(x_grid, y_grid)
        annulus = (rho >= 0.5) & (rho < 1.5)
        angle = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
        sector_id = np.minimum((angle * 8 / (2.0 * np.pi)).astype(np.int64), 7)
        counts = np.zeros_like(x_grid, dtype=np.int64)
        counts[annulus] = 100
        sector_maps = np.asarray(
            [np.where(annulus & (sector_id == value), counts, 0) for value in range(8)]
        )

        selected, evidence = cross_validate_orders(
            (1,),
            {1: counts},
            {1: sector_maps},
            rectangle_basis_grid(edges),
            {1: annulus},
            {1: True},
            positivity_radius_deg=2.0,
        )

        self.assertEqual(selected, 0)
        constant = evidence["candidates"]["0"]
        self.assertTrue(constant["valid"])
        np.testing.assert_allclose(constant["fold_scores"], np.zeros(8), atol=1.0e-10)

    def test_failed_fold_invalidates_candidate_for_one_se_selection(self) -> None:
        selected, evidence = choose_one_standard_error(
            {0: [1.0, 1.0], 1: [0.0, float("inf")], 2: [2.0, 2.0]}
        )
        self.assertEqual(selected, 0)
        failed = evidence["candidates"]["1"]
        self.assertFalse(failed["valid"])
        self.assertEqual(failed["failed_fold_count"], 1)
        self.assertEqual(failed["fold_scores"], [0.0, None])
        self.assertIsNone(failed["mean"])

    def test_all_failed_candidates_are_fatal(self) -> None:
        with self.assertRaisesRegex(ValueError, "All candidate surface orders"):
            choose_one_standard_error({0: [float("inf")], 1: [float("inf")]})


class ContinuousCountAndMapContractTests(unittest.TestCase):
    def test_maps_keep_full_pixels_while_threshold_count_uses_exact_radius(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = pa.table(
                {
                    "ra_mean_deg": [0.48, 0.52, 0.90],
                    "dec_mean_deg": [0.10, 0.10, 0.10],
                    "cell_id": [1, 1, 1],
                }
            )
            pq.write_table(table, root / "events.parquet")
            continuous, maps, sector_maps, edges = scan_continuous_annulus_maps(
                root,
                (1,),
                {1: 0.5},
                {1: 1.5},
                grid_step_deg=1.0,
                source_ra_deg=0.0,
                source_dec_deg=0.0,
                print_every=0,
            )

        self.assertEqual(continuous[1], 2)
        self.assertEqual(int(maps[1].sum()), 3)
        x_index = int(np.searchsorted(edges, 0.5, side="right") - 1)
        y_index = int(np.searchsorted(edges, 0.5, side="right") - 1)
        self.assertEqual(int(maps[1][y_index, x_index]), 3)
        self.assertEqual(int(sector_maps[1][:, y_index, x_index].sum()), 3)
        self.assertEqual(int(sector_maps[1][1, y_index, x_index]), 3)


if __name__ == "__main__":
    unittest.main()
