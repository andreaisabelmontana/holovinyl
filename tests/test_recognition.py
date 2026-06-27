"""Tests for the HoloVinyl recognition + pose pipeline.

Covered:
  * a warped/noisy query of a known cover matches the CORRECT album with
    inliers above threshold;
  * a random non-cover image returns "no match";
  * the recovered homography maps the cover corners into a plausible (convex,
    sensibly sized) quad inside the query frame;
  * recognition survives strong brightness change and heavy blur.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from holovinyl.covers import COVER_SPECS, render_cover
from holovinyl.recognize import CoverDatabase, MIN_INLIERS
from holovinyl.augment import photograph, non_cover_image


@pytest.fixture(scope="module")
def db() -> CoverDatabase:
    """A database built directly from rendered covers (no disk dependency)."""
    database = CoverDatabase()
    for spec in COVER_SPECS:
        database.enroll(spec.name, render_cover(spec))
    return database


def _quad_is_convex(quad: np.ndarray) -> bool:
    """True if the 4-point quad (shape (4,1,2)) is convex (no self-crossing)."""
    pts = quad.reshape(4, 2)
    signs = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cross = np.cross(b - a, c - b)
        signs.append(np.sign(cross))
    signs = [s for s in signs if s != 0]
    return len(set(signs)) == 1


def test_known_cover_matches_correct_album(db: CoverDatabase) -> None:
    rng = np.random.default_rng(1)
    for spec in COVER_SPECS:
        query = photograph(render_cover(spec), rng)
        result = db.recognize(query)
        assert result.matched, f"{spec.name} should match something"
        assert result.name == spec.name, (
            f"{spec.name} mis-identified as {result.name}"
        )
        assert result.inliers > MIN_INLIERS, (
            f"{spec.name} inliers {result.inliers} below threshold {MIN_INLIERS}"
        )


def test_non_cover_returns_no_match(db: CoverDatabase) -> None:
    rng = np.random.default_rng(99)
    rejected = 0
    trials = 10
    for _ in range(trials):
        neg = non_cover_image(rng)
        result = db.recognize(neg)
        if not result.matched:
            rejected += 1
    # Random clutter must be rejected essentially always.
    assert rejected == trials, f"only {rejected}/{trials} negatives rejected"


def test_homography_projects_plausible_quad(db: CoverDatabase) -> None:
    rng = np.random.default_rng(7)
    spec = COVER_SPECS[0]
    canvas = 800
    query = photograph(render_cover(spec), rng, canvas=canvas)
    result = db.recognize(query)
    assert result.matched and result.homography is not None

    quad = db.project_corners(result)
    assert quad is not None
    pts = quad.reshape(4, 2)

    # Corners land within a plausible margin of the query frame. The sleeve is
    # placed with random offset + perspective jitter, so a corner can sit a
    # little past the canvas edge; it must not fly off to absurd coordinates.
    margin = 0.2 * canvas
    assert pts[:, 0].min() > -margin and pts[:, 0].max() < canvas + margin
    assert pts[:, 1].min() > -margin and pts[:, 1].max() < canvas + margin

    # The projected sleeve is a convex quad covering a sensible area.
    assert _quad_is_convex(quad), "projected sleeve outline is not convex"
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    assert area > 0.05 * canvas * canvas, "projected sleeve implausibly small"
    assert area < 0.95 * canvas * canvas, "projected sleeve implausibly large"


def test_robust_to_brightness(db: CoverDatabase) -> None:
    rng = np.random.default_rng(3)
    spec = COVER_SPECS[2]
    # Force a strong darkening and then a strong brightening.
    for bright in (0.45, 1.6):
        query = photograph(render_cover(spec), rng, brightness=bright, blur=0)
        result = db.recognize(query)
        assert result.name == spec.name, (
            f"brightness {bright}: got {result.name}"
        )
        assert result.inliers > MIN_INLIERS


def test_robust_to_blur(db: CoverDatabase) -> None:
    rng = np.random.default_rng(5)
    spec = COVER_SPECS[3]
    query = photograph(render_cover(spec), rng, blur=5, brightness=1.0)
    result = db.recognize(query)
    assert result.name == spec.name
    assert result.inliers > MIN_INLIERS


def test_database_from_dir_roundtrip(tmp_path) -> None:
    from holovinyl.covers import build_database

    out = tmp_path / "covers"
    build_database(str(out))
    database = CoverDatabase.from_dir(str(out))
    assert len(database.entries) == len(COVER_SPECS)

    rng = np.random.default_rng(11)
    spec = COVER_SPECS[4]
    query = photograph(render_cover(spec), rng)
    assert database.recognize(query).name == spec.name
