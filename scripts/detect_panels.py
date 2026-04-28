"""
Comic Panel Segmentor - Production Ready
========================================
Handles: uniform grids, dark artwork backgrounds, thin borders (1-3px)

Core insight:
- Horizontal borders = easy to detect via full-row projection (they span full width)
- Vertical borders = hard because dark artwork bleeds into adjacent columns
- FIX: Use Hough lines for H-borders, then equal-spacing + local search for V-borders
"""

import cv2
import numpy as np
import os
import sys


# ─────────────────────────────────────────────
#  NORMALIZATION CONFIG
# ─────────────────────────────────────────────
TARGET_W = 1080
TARGET_H = 1350


def find_horizontal_borders(gray):
    """
    Detect horizontal border rows using Hough line detection.
    These span the full width so they're reliably detected.
    Returns list of y-coordinates (border centers), sorted ascending.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    H, W = gray.shape

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=200,
        minLineLength=W * 0.6,   # must span 60%+ of image width
        maxLineGap=20
    )

    h_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # horizontal line
                h_positions.append((y1 + y2) // 2)

    if not h_positions:
        return []

    # Cluster close positions into single border
    h_positions = sorted(set(h_positions))
    clusters = [[h_positions[0]]]
    for p in h_positions[1:]:
        if p - clusters[-1][-1] < 15:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    result = [int(np.median(c)) for c in clusters]

    # Remove outer image edges (very close to 0 or H)
    result = [y for y in result if 5 < y < H - 5]
    return sorted(result)


def find_vertical_borders(gray, h_borders, expected_cols=None):
    """
    Detect vertical borders using two strategies:
    1. Hough lines (works if borders are long enough)
    2. Equal-spacing prior + local dark-pixel search (fallback)
    Returns list of x-coordinates sorted ascending.
    """
    H, W = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Strategy 1: Hough vertical lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=H * 0.5,
        maxLineGap=30
    )

    v_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # vertical line
                v_positions.append((x1 + x2) // 2)

    # Remove outer edges
    v_positions = [x for x in v_positions if 20 < x < W - 20]

    def cluster(positions, gap=30):
        if not positions:
            return []
        positions = sorted(set(positions))
        groups = [[positions[0]]]
        for p in positions[1:]:
            if p - groups[-1][-1] < gap:
                groups[-1].append(p)
            else:
                groups.append([p])
        return [int(np.median(g)) for g in groups]

    v_from_hough = cluster(v_positions)

    # Strategy 2: Equal-spacing + local search
    outer_left  = _find_outer_border_left(gray)
    outer_right = _find_outer_border_right(gray)

    if expected_cols is None:
        expected_cols = len(v_from_hough) + 1 if v_from_hough else 4

    v_from_equal = _equal_spacing_search(gray, outer_left, outer_right, expected_cols, h_borders)

    # Merge: prefer whichever strategy found the right count
    if len(v_from_hough) == expected_cols - 1:
        final = v_from_hough
    elif len(v_from_equal) == expected_cols - 1:
        final = v_from_equal
    else:
        merged = v_from_hough + v_from_equal
        final  = cluster(merged, gap=40)
        final  = [x for x in final if 20 < x < W - 20]

    return sorted(final)


def _find_outer_border_left(gray, search_range=50, dark_thresh=30):
    col_mins  = np.min(gray[:, :search_range], axis=0)
    dark_cols = np.where(col_mins < dark_thresh)[0]
    return int(np.median(dark_cols)) if len(dark_cols) > 0 else 0


def _find_outer_border_right(gray, search_range=50, dark_thresh=30):
    W         = gray.shape[1]
    col_mins  = np.min(gray[:, W - search_range:], axis=0)
    dark_cols = np.where(col_mins < dark_thresh)[0]
    return (W - search_range + int(np.median(dark_cols))) if len(dark_cols) > 0 else W


def _equal_spacing_search(gray, x_left, x_right, num_cols, h_borders,
                           search_range=25, dark_thresh=40):
    """
    Estimate internal borders at equal spacing, then refine by finding
    the darkest column within ±search_range pixels.
    """
    H, W = gray.shape
    inner_width  = x_right - x_left
    est_border_w = max(3, inner_width // (num_cols * 50))
    panel_w      = (inner_width - (num_cols - 1) * est_border_w) // num_cols

    v_borders = []
    for i in range(1, num_cols):
        expected_x = x_left + i * (panel_w + est_border_w)
        x_start    = max(0, expected_x - search_range)
        x_end      = min(W, expected_x + search_range)

        best_x, best_score = expected_x, 0
        for x in range(x_start, x_end):
            dark_count = int(np.sum(gray[:, x] < dark_thresh))
            if dark_count > best_score:
                best_score = dark_count
                best_x     = x

        v_borders.append(best_x)

    return v_borders


def estimate_grid_size(gray, h_borders):
    """
    Score each candidate column count (2-7) by how many dark pixels
    are found near evenly-spaced positions. Returns best estimate.
    """
    H, W        = gray.shape
    outer_left  = _find_outer_border_left(gray)
    outer_right = _find_outer_border_right(gray)
    inner_w     = outer_right - outer_left

    best_score, best_cols = -1, 4

    for num_cols in range(2, 8):
        panel_w = inner_w // num_cols
        score   = 0
        for i in range(1, num_cols):
            expected_x = outer_left + i * panel_w
            x_start    = max(0, expected_x - 20)
            x_end      = min(W, expected_x + 20)
            strip      = gray[:, x_start:x_end]
            col_dark   = np.sum(strip < 30, axis=0)
            score     += int(np.max(col_dark))

        score_per_border = score / (num_cols - 1)
        if score_per_border > best_score:
            best_score = score_per_border
            best_cols  = num_cols

    return best_cols


def normalize_panel(crop, target_w=TARGET_W, target_h=TARGET_H):
    """Scale panel to fit target canvas, centered on black background."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    aspect = w / h
    new_w  = target_w
    new_h  = int(new_w / aspect)

    if new_h > target_h:
        new_h = target_h
        new_w = int(new_h * aspect)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off   = (target_h - new_h) // 2
    x_off   = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def generate_debug_image(img, gray, h_borders, v_borders):
    """Draw detected grid lines on the image and save as debug_grid.jpg."""
    H, W  = gray.shape
    debug = img.copy()

    for y in [0] + h_borders + [H]:
        cv2.line(debug, (0, y), (W, y), (0, 255, 0), 3)
    for x in [0] + v_borders + [W]:
        cv2.line(debug, (x, 0), (x, H), (0, 100, 255), 3)

    cv2.imwrite("output/debug_grid.jpg", debug)
    print("Debug grid saved: output/debug_grid.jpg")


def segment_comic(image_path, output_dir="output", padding=5, expected_cols=None):
    """
    Main segmentation pipeline.
    Reads image_path, detects panels, normalizes and saves them.
    Returns list of saved panel paths in reading order.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    print(f"Image: {W}x{H}px")

    # ── Step 1: Horizontal borders (Hough — reliable) ──────────────────────
    h_borders = find_horizontal_borders(gray)
    print(f"Horizontal borders: {h_borders}")

    # ── Step 2: Estimate columns ────────────────────────────────────────────
    if expected_cols is None:
        expected_cols = estimate_grid_size(gray, h_borders)
    print(f"Expected columns: {expected_cols}")

    # ── Step 3: Vertical borders (Hough + equal-spacing) ───────────────────
    v_borders = find_vertical_borders(gray, h_borders, expected_cols=expected_cols)
    print(f"Vertical borders: {v_borders}")

    # ── Step 4: Build cut lines ─────────────────────────────────────────────
    h_cuts = [0] + h_borders + [H]
    v_cuts = [0] + v_borders + [W]
    print(f"Grid: {len(h_cuts)-1} rows × {len(v_cuts)-1} cols")

    # ── Debug image ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    generate_debug_image(img, gray, h_borders, v_borders)

    # ── Step 5: Crop + normalize ────────────────────────────────────────────
    saved      = []
    panel_num  = 1

    for ri in range(len(h_cuts) - 1):
        y1 = h_cuts[ri]     + padding
        y2 = h_cuts[ri + 1] - padding
        if y2 - y1 < 30:
            continue  # skip thin strips (border noise)

        for ci in range(len(v_cuts) - 1):
            x1 = v_cuts[ci]     + padding
            x2 = v_cuts[ci + 1] - padding
            if x2 - x1 < 30:
                continue

            crop       = img[y1:y2, x1:x2]
            normalized = normalize_panel(crop)

            out_path = os.path.join(output_dir, f"panel_{panel_num:03d}.jpg")
            cv2.imwrite(out_path, normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(out_path)
            print(f"  Panel {panel_num}: src {x2-x1}×{y2-y1}px → {out_path}")
            panel_num += 1

    print(f"\n✓ Extracted {len(saved)} panels to '{output_dir}/'")
    return saved


# ─────────────────────────────────────────────
#  ENTRY POINT (called by GitHub Actions)
# ─────────────────────────────────────────────
def main():
    image_path = "input.jpg"

    if not os.path.exists(image_path):
        print(f"ERROR: {image_path} not found.")
        sys.exit(1)

    saved = segment_comic(
        image_path=image_path,
        output_dir="output",
        padding=5,
        expected_cols=None,   # fully auto-detect
    )

    if not saved:
        print("WARNING: No panels extracted. Saving full image as fallback.")
        img    = cv2.imread(image_path)
        canvas = normalize_panel(img)
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/panel_001.jpg", canvas)


if __name__ == "__main__":
    main()
