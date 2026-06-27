"""Warp an overlay onto a recognised sleeve to prove the recovered pose is usable.

`recognize.py` gives us a homography mapping cover coordinates -> query image
coordinates. Here we use it two ways:

  * `warp_overlay`  : take an arbitrary overlay image, warp it through the same
                      homography and alpha-composite it onto the sleeve region.
                      This is the literal AR step — content sticks to the record.
  * `annotate`      : draw the detected sleeve outline (the projected cover
                      corners) plus the overlay, for a human-readable result.

If the pose is correct, the overlay lands exactly on the sleeve in the photo.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .recognize import CoverDatabase, MatchResult


def make_overlay_quad(size: int = 600) -> np.ndarray:
    """A synthetic overlay 'visual' (concentric rings, like a spinning disc)."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    for r in range(size // 2, 0, -28):
        col = (255, 200, 40) if (r // 28) % 2 == 0 else (40, 80, 255)
        cv2.circle(img, (c, c), r, col, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), size // 12, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), max(4, size // 60), (240, 240, 240), -1, cv2.LINE_AA)
    return img


def warp_overlay(
    scene: np.ndarray,
    overlay: np.ndarray,
    homography: np.ndarray,
    alpha: float = 0.85,
) -> np.ndarray:
    """Warp `overlay` (in cover coords) onto `scene` via `homography`."""
    h, w = scene.shape[:2]
    warped = cv2.warpPerspective(overlay, homography, (w, h))
    mask = cv2.warpPerspective(
        np.full(overlay.shape[:2], 255, np.uint8), homography, (w, h)
    )
    mask3 = (cv2.merge([mask, mask, mask]) > 0).astype(np.float32) * alpha
    out = scene.astype(np.float32) * (1 - mask3) + warped.astype(np.float32) * mask3
    return out.astype(np.uint8)


def annotate(
    scene: np.ndarray,
    db: CoverDatabase,
    result: MatchResult,
    overlay: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw sleeve outline + label + (optional) warped overlay onto a copy."""
    out = scene.copy()
    if not result.matched or result.homography is None:
        cv2.putText(out, "NO MATCH", (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    if overlay is None:
        # Scale a default overlay to the matched cover's size.
        w, h = next(e for e in db.entries if e.name == result.name).size
        overlay = cv2.resize(make_overlay_quad(), (w, h))
    out = warp_overlay(out, overlay, result.homography)

    corners = db.project_corners(result)
    if corners is not None:
        cv2.polylines(out, [np.int32(corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

    label = f"{result.name}  inliers={result.inliers}"
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out
