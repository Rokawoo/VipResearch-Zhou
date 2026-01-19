import cv2
from pathlib import Path
import numpy as np

# -------- CONFIGURATION --------
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]
# --------------------------------

# Get all image paths
image_files = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
num_images = len(image_files)
idx = 0

print("Instructions: ")
print("  n - next image")
print("  p - previous image")
print("  q - quit")

def draw_polygons(img_path):
    """Draw YOLO polygon masks"""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label_path = LABELS_DIR / (img_path.stem + ".txt")
    
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # at least class + 3 points (triangle)
                    cls = parts[0]
                    coords = list(map(float, parts[1:]))
                    # Convert normalized coordinates to pixels
                    pts = np.array([(int(coords[i]*w), int(coords[i+1]*h)) 
                                    for i in range(0, len(coords), 2)], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # Draw polygon
                    cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
                    cv2.putText(img, str(cls), tuple(pts[0][0]-np.array([0,5])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

while True:
    img = draw_polygons(image_files[idx])
    cv2.imshow("Polygon Label Viewer", img)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("n"):
        idx = (idx + 1) % num_images
    elif key == ord("p"):
        idx = (idx - 1) % num_images
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
