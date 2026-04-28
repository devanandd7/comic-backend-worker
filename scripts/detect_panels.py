import cv2
import os
import numpy as np

def sort_contours(cnts):
    if not cnts:
        return []
    
    # Get bounding boxes
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    # Sort from top to bottom first
    # We group panels by row by checking if they are on a similar y-level
    boundingBoxes.sort(key=lambda b: b[1])
    
    rows = []
    current_row = [boundingBoxes[0]]
    
    for i in range(1, len(boundingBoxes)):
        # If the top of the next box is below the middle of the current box, it's a new row
        _, y, _, h = current_row[0]
        if boundingBoxes[i][1] > y + h / 2:
            rows.append(current_row)
            current_row = [boundingBoxes[i]]
        else:
            current_row.append(boundingBoxes[i])
    rows.append(current_row)
    
    # Sort each row from left to right
    sorted_boxes = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        sorted_boxes.extend(row)
        
    return sorted_boxes

def main():
    os.makedirs("output", exist_ok=True)

    if not os.path.exists("input.jpg"):
        print("input.jpg not found. Exiting.")
        return

    img = cv2.imread("input.jpg")
    if img is None:
        print("Failed to read input.jpg")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding and edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out small contours and the whole page itself
    valid_contours = []
    img_h, img_w = img.shape[:2]
    img_area = img_w * img_h

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Must be at least 1% of image but less than 80% to avoid picking up the whole page
        if 0.01 * img_area < area < 0.8 * img_area:
            valid_contours.append(c)

    sorted_boxes = sort_contours(valid_contours)

    # Remove duplicates or near-duplicates
    deduped_boxes = []
    for box in sorted_boxes:
        is_duplicate = False
        for d_box in deduped_boxes:
            # If boxes are 90% same, skip
            if abs(box[0]-d_box[0]) < 20 and abs(box[1]-d_box[1]) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped_boxes.append(box)

    i = 1
    for box in deduped_boxes:
        x, y, w, h = box
        # Add a little padding if possible
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        crop = img[y1:y2, x1:x2]
        
        # --- NORMALIZATION ---
        target_w, target_h = 1080, 1350
        
        # Calculate scaling to fit width 1080
        aspect = crop.shape[1] / crop.shape[0]
        new_w = target_w
        new_h = int(target_w / aspect)
        
        # If height is still too large, scale by height instead
        if new_h > target_h:
            new_h = target_h
            new_w = int(target_h * aspect)
            
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create black canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        cv2.imwrite(f"output/panel_{i:03d}.jpg", canvas)
        print(f"Saved normalized output/panel_{i:03d}.jpg")
        i += 1

if __name__ == "__main__":
    main()
