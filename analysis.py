"""
Core image-processing / morphometry pipeline for corneal endothelium
specular images.

The approach is a *classical* (non-deep-learning) segmentation pipeline
loosely inspired by the general strategy described in:

    "Low-Cost, Smartphone-Based Specular Imaging and Automated Analysis
    of the Corneal Endothelium" (Translational Vision Science & Technology,
    2021; PMC8024782): grayscale -> smoothing -> local illumination
    normalization -> border/edge detection -> thinning/segmentation ->
    per-cell morphometry (density, hexagonality, coefficient of variation).

This is NOT a re-implementation of that paper's proprietary directional
filtering + triple-point algorithm. Instead it uses a well established,
openly documented classical technique for the same job: marker-controlled
watershed segmentation on a distance transform, with cell "sides" counted
by label adjacency (a standard proxy for the triple-point method, since in
a cellular mosaic each interior vertex is shared by exactly three cells).

It is intended for research / educational exploration, not as a
validated clinical instrument.
"""

import warnings
from dataclasses import dataclass, field
import numpy as np
import cv2

# scikit-image is mid-transition on a couple of parameter names
# (binary_closing's positional footprint, remove_small_objects' min_size);
# both still work correctly, only the spelling is changing in a future
# release, so these specific FutureWarnings are silenced to keep the app's
# console output readable.
warnings.filterwarnings("ignore", category=FutureWarning)
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries, clear_border
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, disk, binary_closing
from skimage import exposure


@dataclass
class PipelineParams:
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    gaussian_sigma: float = 1.0
    invert: bool = False
    adaptive_block: int = 41
    adaptive_c: int = 2
    min_cell_px: int = 25
    min_marker_distance: int = 6
    exclude_border_cells: bool = True
    min_signal_level: int = 30  # 0-255; pixels this dark (in the RAW image,
    # before any contrast enhancement) are treated as empty/background, not
    # tissue -- see segment_cells()'s valid_mask for why this has to be
    # judged on the raw brightness rather than after CLAHE.
    use_directional_filter: bool = False  # see _directional_edge_map()
    directional_filter_strength: float = 0.6
    um_per_px: float = 1.0  # calibration: micrometers per pixel


@dataclass
class CellStats:
    labels_img: np.ndarray
    boundaries: np.ndarray
    n_cells: int
    areas_um2: np.ndarray
    sides: np.ndarray
    centroids: list
    ecd: float
    mean_area: float
    sd_area: float
    cv_percent: float
    hex_percent: float
    analyzed_area_mm2: float
    min_area: float = field(default=0.0)
    max_area: float = field(default=0.0)


def load_grayscale(image_bgr_or_gray: np.ndarray) -> np.ndarray:
    if image_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_bgr_or_gray.copy()
    return gray


def _directional_edge_map(img: np.ndarray) -> np.ndarray:
    """
    Bank of 4 oriented edge-ridge kernels (0/90/45/135 degrees), merged by
    taking the strongest response per pixel -- an openly documented,
    classical directional-filter edge enhancer in the spirit of Kirsch's
    compass edge detector (Kirsch, 1971). This targets the same general
    idea as the "several algorithms for better cell-edge sharpening" used
    in the smartphone-endothelium paper this app is inspired by (a bank of
    directional masks merged into one border map), without reproducing
    their specific (undisclosed/likely proprietary) implementation.

    Unlike CLAHE -- which reacts to local contrast but has no notion of
    "edge" or direction -- each kernel here is a ridge detector tuned to
    one orientation (a positive center row/column/diagonal flanked by
    negative neighbors), so a faint, blurred cell-border line still gets a
    strong response as long as it's locally brighter than what's on both
    sides of it, in ANY of the 4 checked orientations. Taking the max
    across all 4 means a border is detected regardless of which way it
    runs, which plain (non-directional) sharpening can't do as well.
    """
    img_f = img.astype(np.float32)
    k0 = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)       # horizontal ridge
    k90 = k0.T                                                                      # vertical ridge
    k45 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)       # diagonal "/"
    k135 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)      # diagonal "\"

    responses = [cv2.filter2D(img_f, -1, k) for k in (k0, k90, k45, k135)]
    merged = np.maximum.reduce(responses)
    merged = np.clip(merged, 0, None)
    peak = merged.max()
    if peak > 0:
        merged = merged / peak * 255.0
    return merged.astype(np.uint8)


def preprocess(gray: np.ndarray, p: PipelineParams) -> np.ndarray:
    """Grayscale -> denoise -> illumination flattening -> contrast boost."""
    img = gray.astype(np.float32)

    # Mild denoise while preserving edges
    if p.gaussian_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=p.gaussian_sigma)

    # Flatten uneven slit-lamp illumination: divide by a heavily blurred
    # version of itself (a simple, robust rolling-ball style correction).
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=25)
    bg[bg == 0] = 1
    flat = (img / bg) * np.mean(bg)
    flat = np.clip(flat, 0, 255).astype(np.uint8)

    # CLAHE local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip,
                             tileGridSize=(p.clahe_grid, p.clahe_grid))
    enhanced = clahe.apply(flat)

    # Optional directional-filter border enhancement (off by default -- see
    # _directional_edge_map()). Boosts faint/blurred borders that CLAHE's
    # local-contrast stretching alone tends to leave weak, which is exactly
    # the failure mode seen on out-of-focus photos: real border structure is
    # there in the pixels, just low-contrast, and CLAHE can't tell "this is
    # an edge" from "this is a slightly brighter smudge" the way a
    # direction-aware ridge filter can.
    if p.use_directional_filter and p.directional_filter_strength > 0:
        edge_map = _directional_edge_map(enhanced)
        boosted = enhanced.astype(np.float32) + p.directional_filter_strength * edge_map.astype(np.float32)
        enhanced = np.clip(boosted, 0, 255).astype(np.uint8)

    if p.invert:
        enhanced = 255 - enhanced

    return enhanced


def segment_cells(enhanced: np.ndarray, p: PipelineParams, valid_mask: np.ndarray = None):
    """
    Marker-controlled watershed segmentation.

    Convention used here: after optional inversion, cell BODIES are
    relatively dark and cell BORDERS (specular reflections at cell
    junctions) are relatively bright -- the common appearance in
    specular endothelium photography. Adjust the "invert" toggle in the
    UI if your image has the opposite polarity.

    valid_mask (optional): boolean array, same shape as enhanced, marking
    pixels that are allowed to become part of a cell at all. Used to keep
    genuinely empty/black background (e.g. everything away from the
    specular reflection band, off in a badly-cropped or badly-framed
    image) from being labeled as fake "cells" -- see analyze()'s docstring
    note on why this has to be computed from the RAW image.
    """
    # Global Otsu threshold on a lightly pre-smoothed copy of the image.
    # Because illumination was already flattened in preprocess(), a single
    # global split between "dark cell body" and "bright border network"
    # works far better here than a per-pixel adaptive threshold: a local
    # adaptive threshold compares each pixel only to its own neighbourhood
    # mean, so in the middle of a large, fairly uniform cell body (far from
    # any border) that local mean is close to the pixel's own value and
    # nothing gets flagged as foreground -- only a thin rim near borders
    # would pass, fragmenting every cell into scattered slivers. Otsu
    # avoids that by using one threshold for the whole (already flattened)
    # image. `adaptive_block` doubles as the pre-threshold smoothing
    # radius and `adaptive_c` as a manual offset from the Otsu value, both
    # exposed as tunable sliders for images where auto-thresholding needs
    # a nudge.
    smooth_sigma = max(0.5, p.adaptive_block / 20.0)
    presmoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=smooth_sigma)
    otsu_val, _ = cv2.threshold(presmoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = float(np.clip(otsu_val + p.adaptive_c * 3, 1, 254))
    interior_mask = presmoothed < thresh_val
    if valid_mask is not None:
        interior_mask = interior_mask & valid_mask

    # Clean up: close small gaps in cell interiors, drop tiny noise blobs
    closed = binary_closing(interior_mask, footprint=disk(2))
    cleaned = remove_small_objects(closed, min_size=p.min_cell_px)

    # Distance transform -> local maxima as watershed markers.
    # The distance map itself is smoothed first so small intensity
    # fluctuations inside one cell body don't create several spurious
    # peaks (and therefore several markers) within a single real cell.
    distance = ndi.distance_transform_edt(cleaned)
    distance_smooth = ndi.gaussian_filter(distance, sigma=1.5)
    coords = peak_local_max(
        distance_smooth, min_distance=max(1, p.min_marker_distance),
        labels=cleaned, exclude_border=False
    )
    marker_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords):
        marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)

    # Watershed floods upward from low elevation (the markers, placed at
    # cell centers) until it hits high-elevation ridges. `enhanced` is
    # already exactly that surface: dark cell bodies (low = basins) and
    # bright borders (high = ridges/walls), so it is used directly, NOT
    # inverted -- inverting it would put the markers on top of hills
    # instead of inside basins and break the segmentation. The flooding
    # domain is restricted to a modest dilation of the detected
    # cell-interior candidates (rather than the whole frame) so that
    # flat, featureless regions with no real cell structure (glare,
    # out-of-focus corners, dark artifacts) cannot be swallowed whole by
    # a single runaway basin -- such pixels are simply left unlabeled
    # (label 0) and excluded from the cell statistics instead of
    # becoming a giant fake "cell".
    elevation = enhanced
    dilation_radius = max(3, p.min_marker_distance)
    watershed_mask = ndi.binary_dilation(cleaned, structure=disk(dilation_radius))
    if valid_mask is not None:
        watershed_mask = watershed_mask & valid_mask
    labels_img = watershed(elevation, markers=markers, mask=watershed_mask)

    return labels_img


def compute_neighbor_sides(labels_img: np.ndarray, cell_ids: np.ndarray) -> dict:
    """
    Count number of distinct neighboring labels for each cell = number of
    'sides' of the polygon in the cellular mosaic (a standard proxy for
    the triple-point vertex count used in true endothelial morphometry).
    """
    sides = {cid: set() for cid in cell_ids}
    # Check right and down neighbors (sufficient to catch every adjacency once)
    h, w = labels_img.shape
    right_a = labels_img[:, :-1]
    right_b = labels_img[:, 1:]
    down_a = labels_img[:-1, :]
    down_b = labels_img[1:, :]

    for a, b in ((right_a, right_b), (down_a, down_b)):
        diff = a != b
        pairs = np.stack([a[diff], b[diff]], axis=1)
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        for x, y in np.unique(pairs, axis=0):
            if x in sides:
                sides[x].add(y)
            if y in sides:
                sides[y].add(x)

    return {cid: len(neighbors) for cid, neighbors in sides.items()}


def analyze(gray: np.ndarray, p: PipelineParams) -> CellStats:
    enhanced = preprocess(gray, p)

    # Gate out empty/background pixels using the RAW image's own absolute
    # brightness, not `enhanced`. CLAHE's whole job is *local* contrast
    # stretching -- inside a nearly-black, featureless patch (typical of
    # everything away from the specular reflection band, e.g. in a loosely
    # cropped or badly framed photo) that stretches pure sensor noise into
    # full-range fake "texture", which the illumination-flattening step in
    # preprocess() can amplify further (dividing a near-zero pixel by a
    # near-zero local background is unstable). By the time that reaches
    # `enhanced`, a dead-black region can look just as textured as real
    # tissue -- so the only reliable signal for "is there anything here at
    # all" is the ORIGINAL brightness, checked before any of that.
    #
    # A per-pixel comparison isn't robust enough on its own: real camera
    # sensor/JPEG noise in a near-black region is spiky, with individual
    # pixels randomly popping well above the mean (e.g. a region averaging
    # ~10/255 can have stray pixels near 30/255), so a lone-pixel threshold
    # lets speckles of "valid" pixels through, which is exactly what
    # produced the ring of tiny fake cells along dark edges. Smoothing
    # first means a pixel only counts as real signal if its neighborhood is
    # consistently bright, not just noisy.
    raw_smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=3.0)
    valid_mask = raw_smooth >= p.min_signal_level

    labels_img = segment_cells(enhanced, p, valid_mask=valid_mask)

    if p.exclude_border_cells:
        interior_labels = clear_border(labels_img)
    else:
        interior_labels = labels_img

    props = regionprops(interior_labels)
    props = [r for r in props if r.area >= p.min_cell_px]

    # Safety net against runaway/artifact regions (glare, out-of-focus
    # patches, large unstructured areas): a real endothelial cell should
    # not be a huge outlier relative to the rest of the mosaic. Anything
    # far larger than the bulk of detected cells is dropped from the
    # statistics rather than skewing ECD/CV/HEX.
    if len(props) >= 5:
        areas_tmp = np.array([r.area for r in props], dtype=float)
        median_area = np.median(areas_tmp)
        if median_area > 0:
            props = [r for r, a in zip(props, areas_tmp) if a <= median_area * 8]

    cell_ids = np.array([r.label for r in props])
    areas_px = np.array([r.area for r in props], dtype=float)
    centroids = [r.centroid for r in props]

    px_area_um2 = p.um_per_px ** 2
    areas_um2 = areas_px * px_area_um2

    sides_map = compute_neighbor_sides(labels_img, cell_ids) if len(cell_ids) else {}
    sides = np.array([sides_map.get(cid, 0) for cid in cell_ids], dtype=int)

    boundaries = find_boundaries(labels_img, mode="thick")

    n_cells = len(props)
    if n_cells > 0:
        mean_area = float(np.mean(areas_um2))
        sd_area = float(np.std(areas_um2, ddof=1)) if n_cells > 1 else 0.0
        cv_percent = float(sd_area / mean_area * 100) if mean_area > 0 else 0.0
        hex_percent = float(np.mean(sides == 6) * 100)
        min_area = float(np.min(areas_um2))
        max_area = float(np.max(areas_um2))
    else:
        mean_area = sd_area = cv_percent = hex_percent = min_area = max_area = 0.0

    h, w = gray.shape
    analyzed_area_mm2 = (h * w * px_area_um2) / 1_000_000.0  # um^2 -> mm^2
    ecd = (n_cells / analyzed_area_mm2) if analyzed_area_mm2 > 0 else 0.0

    return CellStats(
        labels_img=labels_img,
        boundaries=boundaries,
        n_cells=n_cells,
        areas_um2=areas_um2,
        sides=sides,
        centroids=centroids,
        ecd=ecd,
        mean_area=mean_area,
        sd_area=sd_area,
        cv_percent=cv_percent,
        hex_percent=hex_percent,
        analyzed_area_mm2=analyzed_area_mm2,
        min_area=min_area,
        max_area=max_area,
    )
