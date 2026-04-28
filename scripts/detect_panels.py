import cv2
import os
import numpy as np

def detect_grid_panels(img):
    """
    Smart grid-based panel detection.
    Looks for the white/bright separator lines that form a grid,
    then slices the image between those lines.
    This is far more reliable than contour detection for comic strips.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Step 1: Find Horizontal Separators ---
    # A separator row is one where almost all pixels are bright (white border)
    row_brightness = np.mean(gray, axis=1)  # Average brightness of each row
    h_separators = []
    threshold = 200  # Rows brighter than this are considered separators
    
    in_separator = False
    sep_start = 0
    for y, brightness in enumerate(row_brightness):
        if brightness > threshold and not in_separator:
            in_separator = True
            sep_start = y
        elif brightness <= threshold and in_separator:
            in_separator = False
            sep_mid = (sep_start + y) // 2
            h_separators.append(sep_mid)

    if in_separator:  # Handle last separator at edge
        h_separators.append((sep_start + h) // 2)

    # --- Step 2: Find Vertical Separators ---
    col_brightness = np.mean(gray, axis=0)  # Average brightness of each col
    v_separators = []

    in_separator = False
    sep_start = 0
    for x, brightness in enumerate(col_brightness):
        if brightness > threshold and not in_separator:
            in_separator = True
            sep_start = x
        elif brightness <= threshold and in_separator:
            in_separator = False
            sep_mid = (sep_start + x) // 2
            v_separators.append(sep_mid)

    if in_separator:
        v_separators.append((sep_start + w) // 2)

    print(f"Found {len(h_separators)} horizontal separators: {h_separators}")
    print(f"Found {len(v_separators)} vertical separators: {v_separators}")

    # --- Step 3: Define Rows and Columns from separators ---
    # Convert separator positions into cell boundaries
    row_boundaries = []
    prev = 0
    for sep in h_separators:
        if sep - prev > h * 0.05:  # Ignore tiny gaps (less than 5% of height)
            row_boundaries.append((prev, sep))
        prev = sep
    if h - prev > h * 0.05:
        row_boundaries.append((prev, h))

    col_boundaries = []
    prev = 0
    for sep in v_separators:
        if sep - prev > w * 0.05:  # Ignore tiny gaps
            col_boundaries.append((prev, sep))
        prev = sep
    if w - prev > w * 0.05:
        col_boundaries.append((prev, w))

    print(f"Detected {len(row_boundaries)} rows and {len(col_boundaries)} columns")

    # --- Step 4: Crop each cell ---
    panels = []
    for r_start, r_end in row_boundaries:
        for c_start, c_end in col_boundaries:
            panel = img[r_start:r_end, c_start:c_end]
            # Only include if the panel has reasonable dimensions
            if panel.shape[0] > 50 and panel.shape[1] > 50:
                panels.append(panel)

    return panels


def normalize_panel(panel, target_w=1080, target_h=1350):
    """
    Resize panel to fit within target dimensions (maintain aspect ratio),
    then center it on a black canvas.
    """
    h, w = panel.shape[:2]
    aspect = w / h

    # Scale to fit
    new_w = target_w
    new_h = int(new_w / aspect)
    if new_h > target_h:
        new_h = target_h
        new_w = int(new_h * aspect)

    resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create black canvas and center
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def main():
    os.makedirs("output", exist_ok=True)

    if not os.path.exists("input.jpg"):
        print("ERROR: input.jpg not found.")
        return

    img = cv2.imread("input.jpg")
    if img is None:
        print("ERROR: Failed to read input.jpg")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    panels = detect_grid_panels(img)

    if not panels:
        print("ERROR: No panels detected. Saving full image as fallback.")
        canvas = normalize_panel(img)
        cv2.imwrite("output/panel_001.jpg", canvas)
        return

    print(f"Total panels detected: {len(panels)}")
    for i, panel in enumerate(panels):
        normalized = normalize_panel(panel)
        path = f"output/panel_{i+1:03d}.jpg"
        cv2.imwrite(path, normalized)
        print(f"Saved {path} (original size: {panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    main()
