# HoloVinyl

Point a camera at an album sleeve and the system recognises *which* record it is
and *where* it sits in the frame — then warps an interactive overlay onto it. The
augmented-reality visual is the front end; the real work is the computer-vision
core that recognises the sleeve and recovers its pose.

- **Live demo (recognition engine explained + canvas AR toy):**
  https://andreaisabelmontana.github.io/holovinyl/

## What it does

Given a photo of an album sleeve taken at an angle, under uneven light, slightly
out of focus, on a cluttered background, HoloVinyl:

1. identifies which enrolled cover it is, and
2. estimates the homography (the sleeve's planar pose) so an overlay can be
   warped onto exactly where the sleeve appears.

It reports **no match** when the geometric evidence is too weak, so random
non-cover images don't get hallucinated as albums.

## Recognition pipeline (`holovinyl/recognize.py`)

For each query image:

1. **ORB features** — detect up to 1500 keypoints and compute their binary
   descriptors (`cv2.ORB_create`).
2. **Descriptor matching** — `BFMatcher(NORM_HAMMING)` kNN-matches query
   descriptors to each enrolled cover (`k=2`).
3. **Lowe's ratio test** — keep a match only if the best neighbour is clearly
   closer than the second (`distance < 0.75 * second`), discarding ambiguous
   matches.
4. **RANSAC homography** — `cv2.findHomography(..., cv2.RANSAC)` fits a planar
   transform from cover coordinates to query coordinates and counts geometric
   inliers.
5. **Decision** — the cover with the most inliers wins, provided it clears the
   inlier threshold (15). Otherwise: no match.

The returned 3×3 homography maps cover points to where they land in the photo, so
`project_corners()` gives the sleeve's outline and any overlay can be warped onto
it.

## Pose / overlay (`holovinyl/overlay.py`)

To prove the recovered pose is usable, `warp_overlay()` pushes an overlay image
through the same homography (`cv2.warpPerspective`) and alpha-composites it onto
the sleeve. `annotate()` draws the detected sleeve outline plus a warped
"spinning disc" overlay — see `data/output/overlay_demo.png` after running the
demo. If the pose is correct, the overlay lands exactly on the tilted sleeve.

## Dataset (`holovinyl/covers.py`, `holovinyl/augment.py`)

Real album art is copyrighted, so the database is **6 original abstract covers
generated procedurally** (gradients, geometric primitives, a grid, title/artist
text — all corner-rich for ORB), committed under `data/covers/`. They're
deterministic: each cover is seeded from a hash of its name.

Query images are produced by **photographing-style augmentation**: a random
perspective warp, composite over a cluttered background, brightness/contrast
jitter, and Gaussian blur (`holovinyl/augment.py`).

## Results

From `python demo.py` (72 queries, 12 per cover; 24 random negatives):

| Metric | Value |
| --- | --- |
| Match accuracy | **100.0%** (72/72) |
| Mean inlier count | **201.3** |
| False-positive rate (negatives) | **0.0%** (0/24) |
| Inlier threshold | 15 |

## Run it

```bash
pip install -r requirements.txt
python -m holovinyl.covers   # (re)generate the cover database
python demo.py               # accuracy + mean inliers + annotated overlay
python -m pytest -q          # tests
```

### Tests (`tests/test_recognition.py`)

- a warped/noisy query of a known cover matches the **correct** album with
  inliers above threshold;
- a random non-cover image returns **no match**;
- the recovered homography maps the cover corners into a plausible (convex,
  sensibly sized, in-frame) quad;
- recognition survives strong brightness change and heavy blur;
- a build-from-disk database round-trips and still recognises queries.

```
......                                                                   [100%]
6 passed
```

## Layout

```
holovinyl/
  covers.py      synthesize the distinct cover database
  augment.py     render "photographed" query images
  recognize.py   ORB + ratio test + RANSAC homography recognition
  overlay.py     warp an overlay onto the recovered sleeve pose
data/covers/     the committed cover database
demo.py          end-to-end evaluation + annotated overlay
tests/           pytest suite
index.html       showcase page + browser AR toy
```

The recognition engine is real Python/OpenCV. The web page's AR effect is a
canvas demo of the front-end experience; the matching + pose math runs in the
Python core above.
