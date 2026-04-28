import cv2
import numpy as np
import os


# ─────────────────────────────────────────────
#  TUNING PARAMETERS
# ─────────────────────────────────────────────
DARK_THRESHOLD  = 80   # Row/col mean below this = border (lower = stricter)
BRIGHT_THRESHOLD = 200 # For white-border fallback
MIN_GAP         = 30   # Minimum panel size in pixels
PADDING         = 5    # Pixels to trim from each panel edge (removes border residue)
TARGET_W        = 1080
TARGET_H        = 1350


# ─────────────────────────────────────────────
#  STAGE 1: Projection-based border detection
# ─────────────────────────────────────────────
def find_splits(means, threshold, min_gap, dark=True):
    """
    Finds the center of each border band in a brightness projection.
    dark=True  → finds dark bands (black borders)
    dark=False → finds bright bands (white borders)
    """
    is_border = means < threshold if dark else means > threshold

    splits = []
    in_border = False
    border_start = 0

    for i, b in enumerate(is_border):
        if b and not in_border:
            in_border = True
            border_start = i
        elif not b and in_border:
            in_border = False
            splits.append((border_start + i) // 2)

    if in_border:
        splits.append((border_start + len(means)) // 2)

    # Filter out splits that are too close together (noise)
    if not splits:
        return []

    filtered = [splits[0]]
    for s in splits[1:]:
        if s - filtered[-1] > min_gap:
            filtered.append(s)

    return filtered


def projection_detect(img):
    """
    Primary detection strategy.
    Tries dark-border projection first, then white-border fallback.
    Returns list of panel dicts with x1,y1,x2,y2,row,col.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    row_means = np.mean(gray, axis=1)   # shape: (H,)
    col_means = np.mean(gray, axis=0)   # shape: (W,)

    # --- Try dark borders first ---
    h_splits = find_splits(row_means, DARK_THRESHOLD, MIN_GAP, dark=True)
    v_splits = find_splits(col_means, DARK_THRESHOLD, MIN_GAP, dark=True)

    print(f"[Dark borders] H splits: {len(h_splits)}  V splits: {len(v_splits)}")

    # --- Fallback: white/bright borders ---
    if len(h_splits) < 1 or len(v_splits) < 1:
        h_splits = find_splits(row_means, BRIGHT_THRESHOLD, MIN_GAP, dark=False)
        v_splits = find_splits(col_means, BRIGHT_THRESHOLD, MIN_GAP, dark=False)
        print(f"[White borders fallback] H splits: {len(h_splits)}  V splits: {len(v_splits)}")

    # --- Fallback: Otsu + morphological closing ---
    if len(h_splits) < 1 or len(v_splits) < 1:
        print("[Otsu fallback] Attempting adaptive threshold projection...")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        row_means = np.mean(closed, axis=1)
        col_means = np.mean(closed, axis=0)
        h_splits = find_splits(row_means, 127, MIN_GAP, dark=False)
        v_splits = find_splits(col_means, 127, MIN_GAP, dark=False)
        print(f"[Otsu fallback] H splits: {len(h_splits)}  V splits: {len(v_splits)}")

    if not h_splits and not v_splits:
        return []  # Signal to use contour fallback

    # Build grid lines (add image edges)
    h_lines = [0] + h_splits + [H]
    v_lines = [0] + v_splits + [W]

    panels = []
    panel_num = 1

    for row_idx in range(len(h_lines) - 1):
        y1 = h_lines[row_idx]
        y2 = h_lines[row_idx + 1]

        for col_idx in range(len(v_lines) - 1):
            x1 = v_lines[col_idx]
            x2 = v_lines[col_idx + 1]

            # Skip panels that are too small (noise from thin gaps)
            if (y2 - y1) < 50 or (x2 - x1) < 50:
                continue

            panels.append({
                "id":  panel_num,
                "x1":  x1, "y1": y1,
                "x2":  x2, "y2": y2,
                "row": row_idx,
                "col": col_idx
            })
            panel_num += 1

    return panels


# ─────────────────────────────────────────────
#  FALLBACK: Contour-based detection
# ─────────────────────────────────────────────
def contour_detect(img):
    """
    Fallback for irregular-layout comics with no clear grid.
    Finds large rectangular contours that look like panels.
    """
    print("[Contour fallback] No clear grid found, switching to contour detection...")
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W  = gray.shape
    img_area = H * W

    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blur, 50, 150)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Must be 2%–80% of total image area
        if 0.02 * img_area < area < 0.80 * img_area:
            # Reasonable aspect ratio for a panel
            if 0.15 < (w / h) < 6.0:
                raw_boxes.append((x, y, w, h))

    # Deduplicate (remove near-identical boxes)
    deduped = []
    for box in raw_boxes:
        duplicate = False
        for d in deduped:
            if abs(box[0] - d[0]) < 20 and abs(box[1] - d[1]) < 20:
                duplicate = True
                break
        if not duplicate:
            deduped.append(box)

    # Convert to panel dicts and sort by row then col
    panels = []
    y_tol  = H * 0.05
    for i, (x, y, w, h) in enumerate(deduped):
        panels.append({
            "id":  i + 1,
            "x1":  x,     "y1": y,
            "x2":  x + w, "y2": y + h,
            "row": int(y // y_tol),
            "col": x
        })

    panels.sort(key=lambda p: (p["row"], p["col"]))
    return panels


# ─────────────────────────────────────────────
#  STAGE 4: Reading-order sort
# ─────────────────────────────────────────────
def sort_panels(panels):
    """Sort panels left-to-right, top-to-bottom (standard comic reading order)."""
    panels.sort(key=lambda p: (p["row"], p["col"]))
    return panels


# ─────────────────────────────────────────────
#  STAGE 5: Normalize to target canvas
# ─────────────────────────────────────────────
def normalize_panel(crop, target_w=TARGET_W, target_h=TARGET_H):
    """
    Scale panel to fit target dimensions while maintaining aspect ratio.
    Centers result on a black canvas.
    """
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


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs("output", exist_ok=True)

    if not os.path.exists("input.jpg"):
        print("ERROR: input.jpg not found.")
        return

    img = cv2.imread("input.jpg")
    if img is None:
        print("ERROR: Failed to read input.jpg")
        return

    H, W = img.shape[:2]
    print(f"Input image: {W}x{H}")

    # --- Primary: projection-based ---
    panels = projection_detect(img)

    # --- Fallback: contour-based ---
    if not panels:
        panels = contour_detect(img)

    # --- Last resort: save full image ---
    if not panels:
        print("WARNING: No panels detected at all. Saving full image as single panel.")
        cv2.imwrite("output/panel_001.jpg", normalize_panel(img))
        return

    panels = sort_panels(panels)
    print(f"\nTotal panels detected: {len(panels)}")

    for i, panel in enumerate(panels):
        x1 = panel["x1"] + PADDING
        y1 = panel["y1"] + PADDING
        x2 = panel["x2"] - PADDING
        y2 = panel["y2"] - PADDING

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        crop = img[y1:y2, x1:x2]
        normalized = normalize_panel(crop)

        out_path = f"output/panel_{i + 1:03d}.jpg"
        cv2.imwrite(out_path, normalized)
        print(f"  Saved {out_path}  (src: {x1},{y1} → {x2},{y2}  |  size: {x2-x1}x{y2-y1})")


if __name__ == "__main__":
    main()
