# scripts/detect_polygons.py
import cv2
import numpy as np
from ultralytics import YOLO
import sys

def detect_and_draw_polygons(image_path, model_path='../runs/garlic_seg/weights/best.pt'):
    # Load model
    model = YOLO(model_path)
    
    # Load image
    img = cv2.imread(image_path)
    original = img.copy()  # Keep original for comparison
    
    # Run inference
    results = model.predict(img, retina_masks=True, conf=0.5)
    
    # Process each detection
    garlic_count = 0
    for result in results:
        if result.masks is None:
            print("No garlic detected")
            continue
            
        # Get masks and boxes
        masks = result.masks.data
        boxes = result.boxes if result.boxes is not None else None
        
        for i, mask in enumerate(masks):
            garlic_count += 1
            
            # Convert mask to numpy
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            
            # Find contours (these are the actual edges/ridges)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (main garlic outline)
                main_contour = max(contours, key=cv2.contourArea)
                
                # Draw the polygon outline (exact edge)
                cv2.drawContours(img, [main_contour], -1, (0, 255, 0), 2)  # Green outline
                
                # Optional: Draw a smoother approximation
                # epsilon = 0.002 * cv2.arcLength(main_contour, True)
                # approx = cv2.approxPolyDP(main_contour, epsilon, True)
                # cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)  # Blue approximation
                
                # Calculate and draw centroid
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)  # Red center
                    
                    # Add label with confidence if available
                    if boxes is not None and i < len(boxes):
                        conf = float(boxes[i].conf[0])
                        label = f"Garlic #{garlic_count}: {conf:.2f}"
                        cv2.putText(img, label, (cx - 40, cy - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Optional: Fill the polygon with transparency
                # overlay = img.copy()
                # cv2.fillPoly(overlay, [main_contour], (0, 255, 0))
                # img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # Save result
    output_path = '../results/output_with_polygons.jpg'
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")
    print(f"Detected {garlic_count} garlic bulbs")
    
    # Show both original and result side by side
    combined = np.hstack((original, img))
    cv2.imshow('Original | Polygons', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_with_options(image_path, model_path='../runs/garlic_seg/weights/best.pt', 
                       draw_type='polygon', smooth=False, fill=False):
    """
    Advanced version with options
    
    Args:
        draw_type: 'polygon', 'ellipse', or 'both'
        smooth: Apply polygon smoothing
        fill: Fill the detection area
    """
    # Load model
    model = YOLO(model_path)
    
    # Load image
    img = cv2.imread(image_path)
    overlay = img.copy()
    
    # Run inference
    results = model.predict(img, retina_masks=True, conf=0.5)
    
    # Process each detection
    for result in results:
        if result.masks is None:
            continue
            
        masks = result.masks.data
        
        for mask in masks:
            # Convert mask to numpy
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            
            # Find contours
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                
                # Apply smoothing if requested
                if smooth:
                    epsilon = 0.005 * cv2.arcLength(main_contour, True)
                    main_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                
                # Draw based on type
                if draw_type == 'polygon' or draw_type == 'both':
                    cv2.drawContours(img, [main_contour], -1, (0, 255, 0), 2)
                
                if draw_type == 'ellipse' or draw_type == 'both':
                    if len(main_contour) >= 5:
                        ellipse = cv2.fitEllipse(main_contour)
                        cv2.ellipse(img, ellipse, (255, 0, 0), 2)
                
                # Fill if requested
                if fill:
                    cv2.fillPoly(overlay, [main_contour], (0, 255, 0))
    
    # Apply overlay if filled
    if fill:
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    
    return img

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_and_draw_polygons(sys.argv[1])
    else:
        print("Usage: python detect_polygons.py <image_path>")
        print("\nExample:")
        print("  python detect_polygons.py ../test_images/garlic.jpg")