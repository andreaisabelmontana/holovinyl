"""Synthesize a small database of distinct graphic album covers.

The recognition pipeline needs feature-rich, visually distinct images. Real
album art is copyrighted, so instead we generate a handful of original abstract
covers procedurally. Each cover packs in high-frequency detail (text, shapes,
gradients, grids) so ORB has plenty of corners to lock onto.

The covers are deterministic: the same `name` always yields the same image,
seeded off a hash of the name. That keeps the committed dataset reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

COVER_SIZE = 600  # covers are square, COVER_SIZE x COVER_SIZE pixels


@dataclass(frozen=True)
class CoverSpec:
    """Identity + colour palette for one synthetic album cover."""

    name: str
    title: str
    artist: str


# The shipped database. Each entry produces a visually distinct cover.
COVER_SPECS: list[CoverSpec] = [
    CoverSpec("midnight_signals", "Midnight Signals", "The Cartographers"),
    CoverSpec("paper_oceans", "Paper Oceans", "Halcyon Field"),
    CoverSpec("neon_arboretum", "Neon Arboretum", "Vela Mono"),
    CoverSpec("static_garden", "Static Garden", "Ruth Orlov"),
    CoverSpec("amber_transit", "Amber Transit", "Northbound 9"),
    CoverSpec("glass_cathedral", "Glass Cathedral", "Ivory Static"),
]


def _seed(name: str) -> int:
    """Deterministic 32-bit seed derived from the cover name."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _palette(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick a background, a mid and an accent colour (BGR)."""
    base = rng.integers(20, 230, size=3)
    accent = (base + rng.integers(60, 160, size=3)) % 256
    mid = ((base.astype(int) + accent.astype(int)) // 2) % 256
    return base.astype(np.uint8), mid.astype(np.uint8), accent.astype(np.uint8)


def render_cover(spec: CoverSpec, size: int = COVER_SIZE) -> np.ndarray:
    """Render a single deterministic cover image (BGR uint8)."""
    rng = np.random.default_rng(_seed(spec.name))
    bg, mid, accent = _palette(rng)

    # Diagonal gradient background so the whole canvas carries texture.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = (xx + yy) / (2.0 * size)
    img = (bg[None, None, :].astype(np.float32) * (1 - t[..., None])
           + accent[None, None, :].astype(np.float32) * t[..., None])
    img = img.astype(np.uint8)

    bg_t = tuple(int(c) for c in bg)
    mid_t = tuple(int(c) for c in mid)
    accent_t = tuple(int(c) for c in accent)

    # A scatter of geometric primitives. The mix differs per seed, so each
    # cover ends up with its own structure of corners/edges for ORB.
    n_shapes = int(rng.integers(14, 22))
    for _ in range(n_shapes):
        kind = rng.integers(0, 4)
        colour = mid_t if rng.random() < 0.5 else accent_t
        thickness = int(rng.integers(2, 9))
        if kind == 0:  # circle ring
            c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            r = int(rng.integers(20, size // 3))
            cv2.circle(img, c, r, colour, thickness, cv2.LINE_AA)
        elif kind == 1:  # rectangle
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p2 = (p1[0] + int(rng.integers(30, 220)), p1[1] + int(rng.integers(30, 220)))
            cv2.rectangle(img, p1, p2, colour, thickness, cv2.LINE_AA)
        elif kind == 2:  # line
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.line(img, p1, p2, colour, thickness, cv2.LINE_AA)
        else:  # triangle
            pts = rng.integers(0, size, size=(3, 2)).astype(np.int32)
            cv2.polylines(img, [pts], True, colour, thickness, cv2.LINE_AA)

    # A faint grid adds dense, repeatable corner structure.
    step = int(rng.integers(40, 70))
    for g in range(step, size, step):
        cv2.line(img, (g, 0), (g, size), mid_t, 1, cv2.LINE_AA)
        cv2.line(img, (0, g), (size, g), mid_t, 1, cv2.LINE_AA)

    # Title + artist text. Text is corner-rich and helps distinguish covers.
    text_col = (255, 255, 255) if np.mean(bg) < 128 else (10, 10, 10)
    cv2.putText(img, spec.title, (28, size - 70), cv2.FONT_HERSHEY_DUPLEX,
                1.4, text_col, 3, cv2.LINE_AA)
    cv2.putText(img, spec.artist.upper(), (30, size - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, text_col, 2, cv2.LINE_AA)

    # Border frame.
    cv2.rectangle(img, (6, 6), (size - 7, size - 7), accent_t, 4, cv2.LINE_AA)
    return img


def build_database(out_dir: str, size: int = COVER_SIZE) -> list[str]:
    """Render every cover spec to `out_dir` and return the written paths."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for spec in COVER_SPECS:
        path = os.path.join(out_dir, f"{spec.name}.png")
        cv2.imwrite(path, render_cover(spec, size))
        paths.append(path)
    return paths


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "data", "covers")
    written = build_database(out)
    for p in written:
        print("wrote", p)
