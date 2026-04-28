import cv2
import os

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
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out small contours
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h > 200: # Threshold for panel size
            valid_contours.append(c)

    sorted_boxes = sort_contours(valid_contours)

    i = 1
    for box in sorted_boxes:
        x, y, w, h = box
        # Add a little padding if possible
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"output/panel_{i:03d}.jpg", crop)
        print(f"Saved output/panel_{i:03d}.jpg")
        i += 1

if __name__ == "__main__":
    main()
