"""
Comic Panel Segmentor - Gemini AI Powered (Ultra Smart Version)
==============================================================
Strategy:
  - Ask Gemini to detect EVERY numbered panel.
  - Use normalized coordinates (0-1000) which Gemini is best at.
  - Automatically handle both [ymin, xmin, ymax, xmax] and {ymin, xmin, ...} formats.
  - Sort by number to ensure perfect story sequence.
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
TARGET_H       = 1080
PADDING        = 3       # Minimal padding to keep as much content as possible


def detect_panels_with_gemini(img, api_key, model_name):
    H, W = img.shape[:2]
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    # This prompt is designed to trigger Gemini's object detection capabilities
    prompt = """
Detect every individual comic panel in this image. 
There are multiple panels, each marked with a small number in the top-left corner (1, 2, 3, etc.).

Your task:
1. Find every numbered panel.
2. Return a JSON list where each item is a panel with its ID and bounding box.
3. Use normalized coordinates [ymin, xmin, ymax, xmax] where each value is 0 to 1000.
   - ymin: top edge
   - xmin: left edge
   - ymax: bottom edge
   - xmax: right edge
4. CRITICAL: Do NOT merge panels. Each number must be its own separate crop.
5. Crop exactly inside the black or white borders of each panel.

Return ONLY JSON:
[
  {"id": 1, "box_2d": [ymin, xmin, ymax, xmax]},
  {"id": 2, "box_2d": [ymin, xmin, ymax, xmax]},
  ...
]
"""

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                {"text": prompt}
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), 
                                   headers={"Content-Type": "application/json"}, method="POST")
        response = urllib.request.urlopen(req, timeout=60)
        response_json = json.loads(response.read().decode("utf-8"))
        raw_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
        print(f"[Gemini Response] {raw_text[:1000]}") # Log for debugging
    except Exception as e:
        print(f"[Gemini API Error] {e}")
        return []

    # Parse JSON
    try:
        # Find anything that looks like a JSON array
        match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
    except:
        return []

    panels = []
    for item in data:
        # Handle different potential formats from Gemini
        box = item.get("box_2d") or [item.get("ymin"), item.get("xmin"), item.get("ymax"), item.get("xmax")]
        
        if not box or None in box:
            continue
            
        ymin, xmin, ymax, xmax = box
        
        # Scale back to pixels
        y1 = int(ymin * H / 1000)
        x1 = int(xmin * W / 1000)
        y2 = int(ymax * H / 1000)
        x2 = int(xmax * W / 1000)
        
        # Validation
        if x2 <= x1 or y2 <= y1:
            continue
            
        panels.append({
            "id": int(item.get("id", 0)),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    # Sort by ID (numerical order)
    panels.sort(key=lambda x: x["id"])
    return panels


def normalize_panel(crop, target_w=TARGET_W, target_h=TARGET_H):
    h, w = crop.shape[:2]
    if h == 0 or w == 0: return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    aspect = w / h
    new_w = target_w
    new_h = int(new_w / aspect)
    
    if new_h > target_h:
        new_h = target_h
        new_w = int(new_h * aspect)
        
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def main():
    if not os.path.exists("input.jpg"): return
    img = cv2.imread("input.jpg")
    if img is None: return
    
    H, W = img.shape[:2]
    os.makedirs("output", exist_ok=True)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    
    panels = []
    if api_key:
        panels = detect_panels_with_gemini(img, api_key, model_name)
    
    # If AI fails, use basic grid fallback (4x4)
    if not panels:
        print("[Fallback] AI detection failed or returned nothing. Using 4x4 grid fallback.")
        rows, cols = 4, 4
        rh, cw = H // rows, W // cols
        for r in range(rows):
            for c in range(cols):
                panels.append({
                    "id": r*cols + c + 1,
                    "x1": c*cw, "y1": r*rh, "x2": (c+1)*cw, "y2": (r+1)*rh
                })

    # Crop and Save
    for i, p in enumerate(panels):
        # Apply padding safely
        x1 = max(0, p["x1"] + PADDING)
        y1 = max(0, p["y1"] + PADDING)
        x2 = min(W, p["x2"] - PADDING)
        y2 = min(H, p["y2"] - PADDING)
        
        if x2 - x1 < 20 or y2 - y1 < 20: continue
        
        crop = img[y1:y2, x1:x2]
        final = normalize_panel(crop)
        cv2.imwrite(f"output/panel_{i+1:03d}.jpg", final)
        
    # Debug Image
    debug = img.copy()
    for p in panels:
        cv2.rectangle(debug, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 2)
    cv2.imwrite("output/debug_grid.jpg", debug)

if __name__ == "__main__":
    main()
