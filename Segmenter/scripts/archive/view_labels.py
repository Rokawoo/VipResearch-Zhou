import cv2
from pathlib import Path
import numpy as np
import sys
import os

# -------- CONFIGURATION --------
BASE_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]
# --------------------------------

# Get all image paths
image_files = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
num_images = len(image_files)
idx = 0

print("Instructions: ")
print("  q/e - previous/next image")
print("  d   - delete current image and label")
print("  ESC - quit")

def draw_polygons(img_path, index, total):
    """Draw YOLO polygon masks"""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label_path = LABELS_DIR / (img_path.stem + ".txt")

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    cls = parts[0]
                    coords = list(map(float, parts[1:]))
                    pts = np.array([(int(coords[i]*w), int(coords[i+1]*h))
                                    for i in range(0, len(coords), 2)], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
                    cv2.putText(img, str(cls), tuple(pts[0][0]-np.array([0,5])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Draw image number in top left
    text = f"{index + 1}/{total}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img

while num_images > 0:
    idx = idx % num_images
    img = draw_polygons(image_files[idx], idx, num_images)
    cv2.imshow("Polygon Label Viewer", img)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("e"):
        idx = (idx + 1) % num_images
    elif key == ord("q"):
        idx = (idx - 1) % num_images
    elif key == ord("d"):
        # Delete image and label
        img_path = image_files[idx]
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        print(f"Deleting: {img_path.name}", end="")
        os.remove(img_path)
        if label_path.exists():
            os.remove(label_path)
            print(f" + {label_path.name}")
        else:
            print()
        image_files.pop(idx)
        num_images -= 1
        if num_images == 0:
            print("No images remaining.")
            break
        if idx >= num_images:
            idx = num_images - 1
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()