"""Album-sleeve recognition via ORB features + Lowe ratio test + RANSAC homography.

Pipeline, per query image:
  1. Detect ORB keypoints + binary descriptors on the query.
  2. For each known cover, kNN-match (k=2) query->cover descriptors with a
     Hamming BFMatcher, then apply Lowe's ratio test to keep only confident
     matches.
  3. Estimate a homography from the surviving matches with cv2.findHomography
     using RANSAC; count the geometric inliers.
  4. The cover with the most inliers wins, provided it clears MIN_INLIERS.
     Otherwise we report "no match".

The returned homography maps points in *cover* coordinates to *query* image
coordinates, so the cover's outline (and any overlay) can be warped onto the
sleeve as it appears in the photo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# --- Tunable constants -------------------------------------------------------
ORB_FEATURES = 1500       # max keypoints per image
RATIO_TEST = 0.75         # Lowe ratio threshold (lower = stricter)
MIN_GOOD_MATCHES = 12     # need at least this many ratio-test survivors to try
MIN_INLIERS = 15          # RANSAC inliers required to accept a match
RANSAC_REPROJ = 5.0       # RANSAC reprojection threshold in pixels


@dataclass
class CoverEntry:
    """One enrolled album cover: its image + precomputed ORB features."""

    name: str
    image: np.ndarray
    keypoints: tuple
    descriptors: Optional[np.ndarray]

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class MatchResult:
    """Outcome of recognising one query against the database."""

    name: Optional[str]              # matched cover name, or None for no-match
    inliers: int                     # geometric inlier count for the winner
    good_matches: int                # ratio-test survivors for the winner
    homography: Optional[np.ndarray] # 3x3 cover->query transform, or None
    query_keypoints: tuple = field(default=(), repr=False)

    @property
    def matched(self) -> bool:
        return self.name is not None


def _orb() -> "cv2.ORB":
    return cv2.ORB_create(nfeatures=ORB_FEATURES)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class CoverDatabase:
    """Holds enrolled covers and matches query images against them."""

    def __init__(self) -> None:
        self._orb = _orb()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.entries: list[CoverEntry] = []

    def enroll(self, name: str, image: np.ndarray) -> CoverEntry:
        """Add one cover image to the database, computing its features once."""
        gray = _to_gray(image)
        kps, desc = self._orb.detectAndCompute(gray, None)
        entry = CoverEntry(name=name, image=image, keypoints=kps, descriptors=desc)
        self.entries.append(entry)
        return entry

    @classmethod
    def from_dir(cls, covers_dir: str) -> "CoverDatabase":
        """Build a database from every image file in a directory."""
        db = cls()
        names = sorted(
            f for f in os.listdir(covers_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not names:
            raise FileNotFoundError(f"no cover images found in {covers_dir!r}")
        for fname in names:
            img = cv2.imread(os.path.join(covers_dir, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            db.enroll(os.path.splitext(fname)[0], img)
        return db

    def _ratio_matches(self, query_desc: np.ndarray, cover_desc: np.ndarray):
        """kNN match + Lowe ratio test; returns list of good DMatch."""
        if query_desc is None or cover_desc is None:
            return []
        if len(query_desc) < 2 or len(cover_desc) < 2:
            return []
        knn = self._matcher.knnMatch(query_desc, cover_desc, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)
        return good

    def _score_entry(self, entry: CoverEntry, q_kps, q_desc):
        """Return (inliers, good_count, homography) for one cover."""
        good = self._ratio_matches(q_desc, entry.descriptors)
        if len(good) < MIN_GOOD_MATCHES:
            return 0, len(good), None

        # queryIdx -> query image, trainIdx -> cover image. We want a
        # homography mapping cover points to query points, so cover is src.
        cover_pts = np.float32([entry.keypoints[m.trainIdx].pt for m in good])
        query_pts = np.float32([q_kps[m.queryIdx].pt for m in good])
        cover_pts = cover_pts.reshape(-1, 1, 2)
        query_pts = query_pts.reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            cover_pts, query_pts, cv2.RANSAC, RANSAC_REPROJ
        )
        if H is None or mask is None:
            return 0, len(good), None
        inliers = int(mask.sum())
        return inliers, len(good), H

    def recognize(self, query: np.ndarray) -> MatchResult:
        """Recognise the album in `query`. Returns the best MatchResult."""
        gray = _to_gray(query)
        q_kps, q_desc = self._orb.detectAndCompute(gray, None)

        best = MatchResult(None, 0, 0, None, query_keypoints=q_kps)
        if q_desc is None:
            return best

        for entry in self.entries:
            inliers, good, H = self._score_entry(entry, q_kps, q_desc)
            if inliers > best.inliers:
                best = MatchResult(
                    name=entry.name, inliers=inliers, good_matches=good,
                    homography=H, query_keypoints=q_kps,
                )

        if best.inliers < MIN_INLIERS:
            # Keep diagnostics but report no confident match.
            return MatchResult(
                None, best.inliers, best.good_matches, None,
                query_keypoints=q_kps,
            )
        return best

    def cover_corners(self, name: str) -> np.ndarray:
        """The 4 corners (TL,TR,BR,BL) of a cover in its own coordinates."""
        entry = next(e for e in self.entries if e.name == name)
        w, h = entry.size
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def project_corners(self, result: MatchResult) -> Optional[np.ndarray]:
        """Map the matched cover's corners into the query frame via the pose."""
        if not result.matched or result.homography is None:
            return None
        corners = self.cover_corners(result.name)
        return cv2.perspectiveTransform(corners, result.homography)
