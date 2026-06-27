"""HoloVinyl — album-sleeve recognition + pose estimation for AR overlays.

The real computer-vision core lives here:
  * covers     — synthesize a database of distinct graphic album covers
  * augment    — render "photographed" query images from covers
  * recognize  — ORB + Lowe ratio test + RANSAC homography recognition
  * overlay    — warp an overlay onto the recovered sleeve pose
"""

from .recognize import CoverDatabase, MatchResult, CoverEntry
from .overlay import annotate, warp_overlay, make_overlay_quad

__all__ = [
    "CoverDatabase",
    "MatchResult",
    "CoverEntry",
    "annotate",
    "warp_overlay",
    "make_overlay_quad",
]

__version__ = "1.0.0"
