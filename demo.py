"""HoloVinyl demo: evaluate recognition on photographed queries + render overlay.

Steps:
  1. Build (if missing) the synthetic cover database under data/covers/.
  2. Generate N photographed-style queries per cover (perspective + background +
     light + blur) and run recognition.
  3. Report match accuracy, mean inlier count, and the false-positive rate on
     random non-cover images.
  4. Save an annotated overlay image (data/output/overlay_demo.png) proving the
     recovered pose: detected sleeve outline + a warped 'spinning disc' overlay.

All numbers printed here are real, from this run.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from holovinyl import covers
from holovinyl.recognize import CoverDatabase, MIN_INLIERS
from holovinyl.augment import photograph, non_cover_image
from holovinyl.overlay import annotate

HERE = os.path.dirname(os.path.abspath(__file__))
COVERS_DIR = os.path.join(HERE, "data", "covers")
OUT_DIR = os.path.join(HERE, "data", "output")
QUERIES_PER_COVER = 12
NEGATIVES = 24
SEED = 20240607


def ensure_covers() -> None:
    if not os.path.isdir(COVERS_DIR) or not os.listdir(COVERS_DIR):
        covers.build_database(COVERS_DIR)


def main() -> None:
    ensure_covers()
    os.makedirs(OUT_DIR, exist_ok=True)
    db = CoverDatabase.from_dir(COVERS_DIR)
    rng = np.random.default_rng(SEED)

    correct = 0
    total = 0
    inliers: list[int] = []
    saved_example = False

    for entry in db.entries:
        for _ in range(QUERIES_PER_COVER):
            query = photograph(entry.image, rng)
            result = db.recognize(query)
            total += 1
            if result.name == entry.name:
                correct += 1
                inliers.append(result.inliers)
                if not saved_example:
                    annotated = annotate(query, db, result)
                    cv2.imwrite(os.path.join(OUT_DIR, "overlay_demo.png"), annotated)
                    cv2.imwrite(os.path.join(OUT_DIR, "query_demo.png"), query)
                    saved_example = True

    # Negatives: random non-cover images should be rejected as "no match".
    false_pos = 0
    for _ in range(NEGATIVES):
        neg = non_cover_image(rng)
        if db.recognize(neg).matched:
            false_pos += 1

    acc = 100.0 * correct / total
    mean_inl = float(np.mean(inliers)) if inliers else 0.0
    fpr = 100.0 * false_pos / NEGATIVES

    print("=" * 56)
    print("HoloVinyl recognition demo")
    print("=" * 56)
    print(f"covers enrolled      : {len(db.entries)}")
    print(f"queries evaluated    : {total} ({QUERIES_PER_COVER} per cover)")
    print(f"match accuracy       : {acc:.1f}%  ({correct}/{total})")
    print(f"mean inlier count    : {mean_inl:.1f}")
    print(f"min-inlier threshold : {MIN_INLIERS}")
    print(f"negatives tested     : {NEGATIVES}")
    print(f"false-positive rate  : {fpr:.1f}%  ({false_pos}/{NEGATIVES})")
    print(f"annotated overlay    : {os.path.join('data', 'output', 'overlay_demo.png')}")


if __name__ == "__main__":
    main()
