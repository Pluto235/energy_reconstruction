from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from apply.tools.build_v6_poisson_pooling_manifest import (
    build_pooling_manifest,
    manifest_payload_sha256,
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


if __name__ == "__main__":
    unittest.main()
