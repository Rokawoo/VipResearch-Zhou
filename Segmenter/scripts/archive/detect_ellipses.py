# scripts/detect_ellipses.py
import cv2
import numpy as np
from ultralytics import YOLO
import sys

def detect_and_draw_ellipses(image_path, model_path='../runs/garlic_seg/weights/best.pt'):
    # Load model
    model = YOLO(model_path)
    
    # Load image
    img = cv2.imread(image_path)
    
    # Run inference
    results = model.predict(img, retina_masks=True, conf=0.5)
    
    # Process each detection
    for result in results:
        if result.masks is None:
            print("No garlic detected")
            continue
            
        # Get masks
        masks = result.masks.data
        
        for mask in masks:
            # Convert mask to numpy
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            
            # Find contours
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and len(contours[0]) >= 5:
                # Fit ellipse
                ellipse = cv2.fitEllipse(contours[0])
                
                # Draw ellipse in green
                cv2.ellipse(img, ellipse, (0, 255, 0), 2)
                
                # Draw center in red
                center = tuple(map(int, ellipse[0]))
                cv2.circle(img, center, 4, (0, 0, 255), -1)
    
    # Save result
    output_path = '../results/output_with_ellipses.jpg'
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")
    
    # Show image (optional)
    cv2.imshow('Garlic with Ellipses', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_and_draw_ellipses(sys.argv[1])
    else:
        print("Usage: python detect_ellipses.py <image_path>")