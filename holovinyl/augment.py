"""Turn a clean cover into a realistic "photographed in the wild" query image.

Real queries are never the pristine cover: the camera sees it at an angle, on
some background, under uneven light, slightly out of focus. We simulate that
with a random perspective warp, a background composite, brightness/contrast
jitter and Gaussian blur. This both builds the evaluation set for demo.py and
gives the tests deterministic, reproducible "hard" inputs.
"""

from __future__ import annotations

import cv2
import numpy as np


def _random_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A cluttered, feature-bearing background so the warp isn't on a void."""
    bg = rng.integers(30, 200, size=(h, w, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    # A few random strokes to add real edges that ORB might fire on.
    for _ in range(int(rng.integers(6, 14))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        col = tuple(int(c) for c in rng.integers(0, 256, size=3))
        cv2.line(bg, p1, p2, col, int(rng.integers(2, 10)), cv2.LINE_AA)
    return bg


def photograph(
    cover: np.ndarray,
    rng: np.random.Generator,
    canvas: int = 800,
    warp_strength: float = 0.18,
    brightness: float | None = None,
    blur: int | None = None,
) -> np.ndarray:
    """Render `cover` as if photographed: perspective + background + light + blur.

    Parameters let tests pin specific brightness/blur; otherwise they're random.
    """
    h, w = cover.shape[:2]

    # Place the cover somewhere inside a larger canvas, then jitter its 4
    # corners to induce a perspective (non-affine) warp.
    margin = canvas - max(h, w)
    if margin < 0:
        scale = (canvas * 0.7) / max(h, w)
        cover = cv2.resize(cover, (int(w * scale), int(h * scale)))
        h, w = cover.shape[:2]
        margin = canvas - max(h, w)
    off_x = int(rng.integers(0, max(1, margin)))
    off_y = int(rng.integers(0, max(1, margin)))

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [off_x, off_y], [off_x + w, off_y],
        [off_x + w, off_y + h], [off_x, off_y + h],
    ])
    jitter = warp_strength * max(h, w)
    dst += rng.uniform(-jitter, jitter, size=dst.shape).astype(np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(cover, H, (canvas, canvas), borderValue=(0, 0, 0))

    # Composite the warped cover over a cluttered background using its mask.
    mask = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), H, (canvas, canvas)
    )
    bg = _random_background(canvas, canvas, rng)
    mask3 = cv2.merge([mask, mask, mask]) > 0
    scene = np.where(mask3, warped, bg)

    # Brightness / contrast jitter (uneven lighting).
    if brightness is None:
        brightness = float(rng.uniform(0.6, 1.4))
    contrast = float(rng.uniform(0.85, 1.15))
    scene = np.clip(scene.astype(np.float32) * brightness * contrast
                    + rng.uniform(-15, 15), 0, 255).astype(np.uint8)

    # Defocus blur.
    if blur is None:
        blur = int(rng.choice([0, 3, 5]))
    if blur and blur >= 3:
        scene = cv2.GaussianBlur(scene, (blur, blur), 0)

    return scene


def non_cover_image(rng: np.random.Generator, size: int = 800) -> np.ndarray:
    """A random natural-ish image that is NOT any enrolled cover (negative)."""
    return _random_background(size, size, rng)
