"""
Comic Panel Segmentor - Gemini AI Powered
==========================================
Strategy:
  PRIMARY  → Gemini Vision API (asks AI to return exact panel bounding boxes)
  FALLBACK → Hough-line + equal-spacing grid detection (pure OpenCV)

Gemini ko image dete hain aur poochte hain:
  "Is image mein saare panels ke coordinates batao JSON format mein"
Gemini precise boxes return karta hai → perfect crops.
"""

import cv2
import numpy as np
import os
import sys
import json
import base64
import re
import urllib.request
import urllib.error


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TARGET_W       = 1080
TARGET_H       = 1080    # Square canvas (matches panel aspect ratio)
PADDING        = 4       # Pixels to trim off each panel edge


# ─────────────────────────────────────────────
#  GEMINI AI PANEL DETECTION  (PRIMARY)
# ─────────────────────────────────────────────
def detect_panels_with_gemini(img, api_key, model_name):
    """
    Sends the comic image to Gemini Vision API.
    Asks it to return bounding boxes for every panel.
    Returns list of panel dicts: [{x1,y1,x2,y2}, ...]
    """
    H, W = img.shape[:2]

    # Encode image to base64 JPEG
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    prompt = f"""
You are a comic panel detection expert. Analyze this comic collage.
The image contains multiple panels. Notice that each panel has a NUMBER in the top-left corner (1, 2, 3, etc.).

Your task:
1. Identify EVERY individual comic panel using the numbers as a guide.
2. For each panel, provide the bounding box in NORMALIZED coordinates (0-1000). 
   - ymin, xmin, ymax, xmax.
3. CRITICAL: Do NOT group multiple panels together. Even if they are in the same row, every numbered panel MUST have its own separate bounding box.
4. Detect the ACTUAL content of each panel, cropping INSIDE the borders.
5. Return the panels in numerical order.

Return ONLY valid JSON. Format:
[
  {{"id": 1, "ymin": 0, "xmin": 0, "ymax": 250, "xmax": 250}},
  ...
]
"""

    # Build API request payload (Gemini REST API)
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"[Gemini] Sending image to {model_name} (using normalized coords)...")
    response_text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    response_json = json.loads(response_text)

    # Extract the text content from the response
    raw_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
    
    # Extract JSON from response
    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if not json_match:
        print(f"[Gemini] Error: No JSON found in response: {raw_text}")
        raise ValueError("Gemini did not return a valid JSON array")

    panels_data = json.loads(json_match.group())

    # Convert normalized coordinates back to pixels
    panels = []
    for p in panels_data:
        # ymin, xmin, ymax, xmax (normalized 0-1000)
        y1 = int(p["ymin"] * H / 1000)
        x1 = int(p["xmin"] * W / 1000)
        y2 = int(p["ymax"] * H / 1000)
        x2 = int(p["xmax"] * W / 1000)

        # Skip if box is invalid or too small
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        panels.append({"id": p.get("id", len(panels)+1), "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    print(f"[Gemini] Detected {len(panels)} panels")
    return panels


# ─────────────────────────────────────────────
#  HOUGH-LINE FALLBACK  (if Gemini fails)
# ─────────────────────────────────────────────
def find_horizontal_borders(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    H, W    = gray.shape

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=200, minLineLength=W * 0.6, maxLineGap=20
    )
    h_pos = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:
                h_pos.append((y1 + y2) // 2)

    if not h_pos:
        return []

    h_pos = sorted(set(h_pos))
    clusters = [[h_pos[0]]]
    for p in h_pos[1:]:
        if p - clusters[-1][-1] < 15:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    result = [int(np.median(c)) for c in clusters]
    return [y for y in result if 5 < y < H - 5]


def find_vertical_borders(gray, h_borders, expected_cols=4):
    H, W    = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=H * 0.5, maxLineGap=30
    )
    v_pos = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:
                v_pos.append((x1 + x2) // 2)

    v_pos = [x for x in v_pos if 20 < x < W - 20]

    def cluster(pos, gap=30):
        if not pos: return []
        pos = sorted(set(pos))
        grps = [[pos[0]]]
        for p in pos[1:]:
            if p - grps[-1][-1] < gap:
                grps[-1].append(p)
            else:
                grps.append([p])
        return [int(np.median(g)) for g in grps]

    v_hough = cluster(v_pos)
    if len(v_hough) == expected_cols - 1:
        return sorted(v_hough)

    # Equal-spacing fallback
    inner_w  = W
    panel_w  = inner_w // expected_cols
    v_equal  = []
    for i in range(1, expected_cols):
        expected_x = i * panel_w
        x_start = max(0, expected_x - 30)
        x_end   = min(W, expected_x + 30)
        best_x, best_score = expected_x, 0
        for x in range(x_start, x_end):
            dark = int(np.sum(gray[:, x] < 40))
            if dark > best_score:
                best_score = dark
                best_x = x
        v_equal.append(best_x)

    merged = cluster(v_hough + v_equal, gap=40)
    return sorted([x for x in merged if 20 < x < W - 20])


def estimate_cols(gray, h_borders):
    H, W  = gray.shape
    best_score, best_cols = -1, 4
    for num_cols in range(2, 8):
        panel_w = W // num_cols
        score = 0
        for i in range(1, num_cols):
            ex = i * panel_w
            strip    = gray[:, max(0, ex-20):min(W, ex+20)]
            col_dark = np.sum(strip < 30, axis=0)
            score   += int(np.max(col_dark))
        spp = score / (num_cols - 1)
        if spp > best_score:
            best_score = spp
            best_cols  = num_cols
    return best_cols


def hough_detect(img):
    """Hough-line grid detection as fallback."""
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W      = gray.shape
    h_borders = find_horizontal_borders(gray)
    exp_cols  = estimate_cols(gray, h_borders)
    v_borders = find_vertical_borders(gray, h_borders, exp_cols)

    print(f"[Hough] H-borders: {h_borders}")
    print(f"[Hough] V-borders: {v_borders}  (expected {exp_cols} cols)")

    h_cuts = [0] + h_borders + [H]
    v_cuts = [0] + v_borders + [W]

    panels = []
    pid = 1
    for ri in range(len(h_cuts) - 1):
        y1 = h_cuts[ri]
        y2 = h_cuts[ri + 1]
        if y2 - y1 < 30: continue
        for ci in range(len(v_cuts) - 1):
            x1 = v_cuts[ci]
            x2 = v_cuts[ci + 1]
            if x2 - x1 < 30: continue
            panels.append({"id": pid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            pid += 1

    return panels


# ─────────────────────────────────────────────
#  NORMALIZE EACH PANEL
# ─────────────────────────────────────────────
def normalize_panel(crop, target_w=TARGET_W, target_h=TARGET_H):
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
    canvas[(target_h - new_h) // 2 : (target_h - new_h) // 2 + new_h,
           (target_w - new_w) // 2 : (target_w - new_w) // 2 + new_w] = resized
    return canvas


# ─────────────────────────────────────────────
#  DEBUG IMAGE
# ─────────────────────────────────────────────
def save_debug_image(img, panels, path="output/debug_grid.jpg"):
    debug = img.copy()
    for p in panels:
        cv2.rectangle(debug, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 3)
        cv2.putText(debug, str(p["id"]),
                    (p["x1"] + 10, p["y1"] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3)
    cv2.imwrite(path, debug)
    print(f"[Debug] Grid overlay saved: {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    image_path = "input.jpg"
    if not os.path.exists(image_path):
        print(f"ERROR: {image_path} not found.")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Failed to read input.jpg")
        sys.exit(1)

    H, W = img.shape[:2]
    print(f"Image loaded: {W}x{H}px")
    os.makedirs("output", exist_ok=True)

    # ── Read Gemini credentials ──────────────────────────────────────────────
    api_key    = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")

    panels = []

    # ── PRIMARY: Gemini AI ───────────────────────────────────────────────────
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            panels = detect_panels_with_gemini(img, api_key, model_name)
        except Exception as e:
            print(f"[Gemini] Failed: {e}")
            print("[Gemini] Falling back to Hough-line detection...")
            panels = []
    else:
        print("[Gemini] No API key — using Hough-line fallback.")

    # ── FALLBACK: Hough + Equal-spacing ─────────────────────────────────────
    if not panels:
        panels = hough_detect(img)

    # ── Last resort ──────────────────────────────────────────────────────────
    if not panels:
        print("WARNING: No panels detected. Saving full image.")
        cv2.imwrite("output/panel_001.jpg", normalize_panel(img))
        return

    print(f"\nTotal panels to crop: {len(panels)}")

    # Save debug image showing detected boxes
    save_debug_image(img, panels)

    # ── Crop + normalize + save ──────────────────────────────────────────────
    for i, panel in enumerate(panels):
        x1 = max(0, panel["x1"] + PADDING)
        y1 = max(0, panel["y1"] + PADDING)
        x2 = min(W, panel["x2"] - PADDING)
        y2 = min(H, panel["y2"] - PADDING)

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        crop       = img[y1:y2, x1:x2]
        normalized = normalize_panel(crop)
        out_path   = f"output/panel_{i + 1:03d}.jpg"
        cv2.imwrite(out_path, normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved {out_path}  ({x2-x1}×{y2-y1}px crop)")


if __name__ == "__main__":
    main()
