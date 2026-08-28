"""
Calibration helpers: turning pixels into real-world micrometers.

Important physical fact this module leans on: cropping a photo does NOT
change its pixel scale. If a camera+optics setup produces, say, 0.5 um per
pixel at a given slit-lamp magnification, that is true for every pixel in
the frame -- selecting a smaller rectangle out of that same, unresized
image still has 0.5 um per pixel. What DOES change the scale is *resizing*
(resampling) the image, e.g. if the capture software exports a downscaled
JPEG instead of the camera's native resolution.

So calibration here is tied to a (camera, magnification) combination via
its *native, full-frame pixel resolution*, not to whatever crop the user
later picks:

- A "calibration profile" records: a name (e.g. "40x"), the native
  full-frame pixel size (width x height) that camera+magnification
  produces, and the resulting micrometers-per-pixel.
- When an image is uploaded, if its pixel dimensions match a saved
  profile's native size, that profile's um/px applies directly -- to the
  whole frame AND to any later crop of it.
- If the uploaded image's resolution differs from the profile's native
  size (e.g. it was exported/resized smaller), um/px can be scaled
  proportionally *only if* the uploaded image is still the FULL field of
  view (just resampled) -- the user has to confirm that assumption,
  since we cannot tell resize from prior cropping just by looking at
  pixel counts.
- Profiles are obtained either by direct entry (if the user already knows
  their system's um/px or field-of-view-in-mm for a magnification) or via
  the built-in ruler/stage-micrometer calibration tool (measure a known
  real-world distance in pixels once, on a calibration photo, at native
  resolution).

No manufacturer calibration numbers are hard-coded anywhere in this file:
CSO does not publish field-of-view-per-magnification or pixel-pitch specs
for this camera in any source we could verify, so those numbers must come
from the user's own equipment (documented value or one-time ruler
calibration) rather than being guessed.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
import math


@dataclass
class CalibrationProfile:
    name: str
    native_w: int
    native_h: int
    um_per_px: float

    def label(self) -> str:
        return f"{self.name} ({self.native_w}x{self.native_h} px, {self.um_per_px:.4f} um/px)"


def profiles_to_json(profiles: list) -> str:
    return json.dumps([asdict(p) for p in profiles], indent=2, ensure_ascii=False)


def profiles_from_json(text: str) -> list:
    data = json.loads(text)
    return [CalibrationProfile(**row) for row in data]


def find_matching_profile(profiles: list, width: int, height: int) -> Optional[CalibrationProfile]:
    """Exact (or orientation-swapped) native-resolution match."""
    for p in profiles:
        if (p.native_w == width and p.native_h == height) or \
           (p.native_w == height and p.native_h == width):
            return p
    return None


def scaled_um_per_px(profile: CalibrationProfile, uploaded_w: int, uploaded_h: int) -> float:
    """
    Proportional rescale of a profile's um/px to an uploaded image of a
    different resolution, assuming the uploaded image is still the full
    field of view (just resampled), not a pre-crop. Uses the width ratio;
    warns implicitly via the caller if width/height ratios disagree
    (non-uniform resize), which the caller should surface to the user.
    """
    ratio = profile.native_w / uploaded_w
    return profile.um_per_px * ratio


def um_per_px_from_two_points(
    x1: float, y1: float, x2: float, y2: float, known_distance_mm: float
) -> float:
    """Compute um/px from a click-measured pixel distance and a known
    real-world distance (e.g. two marks on a stage micrometer / ruler
    photographed through the same optics at a fixed magnification)."""
    pixel_distance = math.hypot(x2 - x1, y2 - y1)
    if pixel_distance <= 0:
        raise ValueError("Dvije točke moraju biti različite (udaljenost > 0 px).")
    return (known_distance_mm * 1000.0) / pixel_distance


@dataclass
class RectangleCalibrationResult:
    top_width_px: float
    bottom_width_px: float
    left_height_px: float
    right_height_px: float
    um_per_px_x: float
    um_per_px_y: float
    um_per_px_avg: float
    mismatch_percent: float  # |x - y| / avg, as a sanity check


def um_per_px_from_rectangle(
    corners: list, width_mm: float, height_mm: float
) -> RectangleCalibrationResult:
    """
    Compute um/px from 4 clicked corners of a rectangle of known real-world
    size (e.g. matched to millimeter graph paper), in order: top-left,
    top-right, bottom-right, bottom-left. Using all 4 sides (rather than a
    single 2-point line) averages out click-precision error over a larger
    baseline, and comparing the independently-derived x-axis and y-axis
    scales is a useful sanity check -- a large mismatch usually means a
    corner was misplaced (or, less commonly, non-square pixels/uncorrected
    lens distortion).

    This assumes calibration.py's aim throughout: it only estimates a
    single isotropic um/px (the app's analysis pipeline does not support
    separate x/y calibration) by averaging the two axis estimates -- for
    that reason a large mismatch_percent should prompt a re-check rather
    than being silently trusted.
    """
    if len(corners) != 4:
        raise ValueError("Potrebne su točno 4 točke (4 kuta).")
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Širina i visina moraju biti pozitivne (u mm).")

    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners

    top_w = math.hypot(x1 - x0, y1 - y0)
    bottom_w = math.hypot(x2 - x3, y2 - y3)
    left_h = math.hypot(x3 - x0, y3 - y0)
    right_h = math.hypot(x2 - x1, y2 - y1)

    if min(top_w, bottom_w, left_h, right_h) <= 0:
        raise ValueError("Kutovi se ne smiju preklapati (sve 4 točke moraju biti različite).")

    avg_w_px = (top_w + bottom_w) / 2.0
    avg_h_px = (left_h + right_h) / 2.0

    um_per_px_x = (width_mm * 1000.0) / avg_w_px
    um_per_px_y = (height_mm * 1000.0) / avg_h_px
    um_per_px_avg = (um_per_px_x + um_per_px_y) / 2.0
    mismatch = abs(um_per_px_x - um_per_px_y) / um_per_px_avg * 100.0 if um_per_px_avg > 0 else 0.0

    return RectangleCalibrationResult(
        top_width_px=top_w, bottom_width_px=bottom_w,
        left_height_px=left_h, right_height_px=right_h,
        um_per_px_x=um_per_px_x, um_per_px_y=um_per_px_y,
        um_per_px_avg=um_per_px_avg, mismatch_percent=mismatch,
    )
