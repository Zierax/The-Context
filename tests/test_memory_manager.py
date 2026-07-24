# ============================================================================
# tests/test_memory_manager.py — Unit tests for virtual memory tree
# ============================================================================

import os
import tempfile

import numpy as np
import pytest

from the_context.core import VirtualMemoryTree


class TestVirtualMemoryTree:
    """Tests for the virtual memory tree."""

    def _write_page(self, tree: VirtualMemoryTree, page_id: str, text: str) -> None:
        """Helper: write a page to disk (simulates what _flush_buffer does)."""
        page_path = os.path.join(tree.pages_dir, f"{page_id}.txt")
        os.makedirs(tree.pages_dir, exist_ok=True)
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_init(self, tmp_path) -> None:
        """Tree must initialise with empty stores."""
        tree = VirtualMemoryTree(page_size=100, cache_size=10, persist_dir=str(tmp_path / "init"))
        assert len(tree.beacon_b1) == 0
        assert len(tree.beacon_b2) == 0
        assert len(tree.beacon_b3) == 0
        assert tree.page_to_beacon == {}

    def test_ingest_stream_yields_pages(self, tmp_path) -> None:
        """Stream of 5000 tokens must yield 5 page IDs (at 1000 tokens/page)."""
        tree = VirtualMemoryTree(
            page_size=1000, cache_size=50, persist_dir=str(tmp_path / "ingest")
        )
        tokens = [f"token_{i}" for i in range(5000)]
        page_ids = list(tree.ingest_stream(iter(tokens)))
        # Expect 5 pages (5000 / 1000 = 5)
        assert len(page_ids) == 5
        # All page IDs must be unique
        assert len(set(page_ids)) == 5

    def test_ingest_stream_partial_page(self, tmp_path) -> None:
        """A stream that doesn't fill a complete page must still yield one page."""
        tree = VirtualMemoryTree(
            page_size=1000, cache_size=10, persist_dir=str(tmp_path / "partial")
        )
        tokens = [f"tok_{i}" for i in range(500)]
        page_ids = list(tree.ingest_stream(iter(tokens)))
        assert len(page_ids) == 1

    def test_allocate_b1(self, tmp_path) -> None:
        """Allocating a B1 beacon must return a beacon ID and store the embedding."""
        tree = VirtualMemoryTree(
            page_size=1000, cache_size=10, persist_dir=str(tmp_path / "alloc")
        )
        embedding = np.random.RandomState(0).randn(512).astype(np.float64)
        # Write page text to disk first
        self._write_page(tree, "page_test", "test content")
        beacon_id = tree.allocate_b1("page_test", embedding)
        assert beacon_id.startswith("b1_")
        assert beacon_id in tree.beacon_b1
        assert tree.page_to_beacon["page_test"] == beacon_id

    def test_lru_eviction(self, tmp_path) -> None:
        """LRU scores must update correctly when pages are accessed."""
        persist = str(tmp_path / "lru_test")
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=persist
        )

        # Write pages to disk and allocate beacons
        for i, page_id in enumerate(["page_1", "page_2", "page_3"]):
            self._write_page(tree, page_id, f"Content of {page_id}")
            emb = np.random.RandomState(i).randn(512).astype(np.float64)
            tree.allocate_b1(page_id, emb)

        # Access page_1 to make it more recently used
        tree.get_page("page_1")

        score_1 = tree.lru_scores.get("page_1", 0.0)
        score_2 = tree.lru_scores.get("page_2", 0.0)
        assert score_1 > score_2

    def test_lru_scores_update(self, tmp_path) -> None:
        """Getting a page must increase its LRU score."""
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=str(tmp_path / "scores")
        )
        emb = np.random.RandomState(0).randn(512).astype(np.float64)
        self._write_page(tree, "page_score", "test content")
        tree.allocate_b1("page_score", emb)
        score_before = tree.lru_scores.get("page_score", 0.0)
        tree.get_page("page_score")
        score_after = tree.lru_scores.get("page_score", 0.0)
        assert score_after > score_before

    def test_disk_roundtrip(self, tmp_path) -> None:
        """Writing a page to disk then loading it must return identical text."""
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=str(tmp_path / "disk")
        )

        original_text = "This is the content of the page to be persisted."
        emb = np.random.RandomState(0).randn(512).astype(np.float64)

        # Write page to disk and allocate beacon
        self._write_page(tree, "disk_page", original_text)
        tree.allocate_b1("disk_page", emb)

        # Load from disk
        loaded = tree.get_page("disk_page")
        assert loaded == original_text

    def test_disk_spill_missing(self, tmp_path) -> None:
        """Loading a non-existent page from disk must return None."""
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=str(tmp_path / "missing")
        )
        assert tree.load_page_from_disk("nonexistent_page") is None

    def test_get_page_not_found(self, tmp_path) -> None:
        """Getting a page that doesn't exist must return None."""
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=str(tmp_path / "get_none")
        )
        assert tree.get_page("ghost_page") is None

    def test_beacon_mappings(self, tmp_path) -> None:
        """Beacon IDs must be mappable between B1 and pages, and B1 to B2."""
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=str(tmp_path / "beacon_map")
        )

        # Create 10 pages to trigger B2 compression
        for i in range(10):
            emb = np.random.RandomState(i).randn(512).astype(np.float64)
            self._write_page(tree, f"bmap_page_{i}", f"content {i}")
            tree.allocate_b1(f"bmap_page_{i}", emb)

        # Check mappings exist
        b1_ids = tree.get_all_b1_ids()
        assert len(b1_ids) == 10

        for i in range(10):
            b1_id = tree.get_beacon_for_page(f"bmap_page_{i}")
            assert b1_id is not None
            assert b1_id in b1_ids

    def test_cache_fill(self, tmp_path) -> None:
        """Cache fill ratio must be between 0 and 1."""
        tree = VirtualMemoryTree(
            page_size=100, cache_size=10, persist_dir=str(tmp_path / "fill")
        )
        assert 0.0 <= tree.get_cache_fill() <= 1.0

        self._write_page(tree, "fill_test", "content")
        emb = np.random.RandomState(0).randn(512).astype(np.float64)
        tree.allocate_b1("fill_test", emb)
        assert tree.get_cache_fill() > 0.0
