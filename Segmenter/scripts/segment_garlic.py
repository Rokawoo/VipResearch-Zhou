"""
Garlic Video Segmentation

Colors:
  Blue = Garlic (right-side up)
  Red  = Garlic with root visible (upside-down)

Logic:
  1. Track detections across frames by centroid proximity
  2. After 0.5s of being seen, mark track as "confirmed"
  3. Draw CURRENT frame's contour for confirmed tracks (no lag)
  4. Ignore detections that haven't been confirmed yet (filters noise)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'

# Classes
GARLIC_BULB = 0
GARLIC_ROOT = 1

# Colors (BGR)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

# Thresholds
CONFIDENCE_THRESHOLD = 0.70
MIN_VISIBLE_SECONDS = 0.5
MAX_MATCH_DISTANCE = 100  # pixels


class Track:
    """Lightweight track - just stores state, not contours."""
    _next_id = 0
    
    def __init__(self, centroid):
        self.id = Track._next_id
        Track._next_id += 1
        self.centroid = centroid
        self.frames_seen = 1
        self.frames_missing = 0
        self.confirmed = False
    
    def update(self, centroid):
        self.centroid = centroid
        self.frames_seen += 1
        self.frames_missing = 0
    
    def mark_missing(self):
        self.frames_missing += 1


def distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def get_contour_and_centroid(mask, frame_shape):
    mask_resized = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    
    if M["m00"] == 0:
        return contour, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return contour, (cx, cy)


def point_inside_contour(point, contour):
    return cv2.pointPolygonTest(contour, point, False) >= 0


class GarlicSegmenter:
    def __init__(self, model_path, fps=30):
        self.model = YOLO(model_path)
        self.fps = fps
        self.min_frames = int(MIN_VISIBLE_SECONDS * fps)
        self.max_missing = fps // 3  # Remove after ~0.33s missing
        
        self.bulb_tracks = []
        self.root_tracks = []
    
    def _update_tracks(self, tracks, detections):
        """
        Match detections to tracks, return list of (detection, is_confirmed) pairs.
        """
        matched_tracks = set()
        matched_detections = set()
        results = []  # (detection_dict, confirmed)
        
        # Match each detection to closest track
        for i, det in enumerate(detections):
            best_track = None
            best_dist = MAX_MATCH_DISTANCE
            
            for track in tracks:
                if track in matched_tracks:
                    continue
                d = distance(det['centroid'], track.centroid)
                if d < best_dist:
                    best_dist = d
                    best_track = track
            
            if best_track:
                # Update existing track
                best_track.update(det['centroid'])
                if best_track.frames_seen >= self.min_frames:
                    best_track.confirmed = True
                matched_tracks.add(best_track)
                matched_detections.add(i)
                results.append((det, best_track.confirmed))
            else:
                # New track
                new_track = Track(det['centroid'])
                tracks.append(new_track)
                results.append((det, False))  # Not confirmed yet
        
        # Mark unmatched tracks as missing
        for track in tracks:
            if track not in matched_tracks:
                track.mark_missing()
        
        # Remove tracks missing too long
        tracks[:] = [t for t in tracks if t.frames_missing < self.max_missing]
        
        return results
    
    def process_frame(self, frame):
        result = self.model.predict(
            frame,
            retina_masks=True,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )[0]
        
        # Extract detections from current frame
        bulb_detections = []
        root_detections = []
        
        if result.masks is not None:
            for mask, cls, conf in zip(
                result.masks.data,
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy()
            ):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                contour, centroid = get_contour_and_centroid(mask_np, frame.shape)
                
                if contour is None or centroid is None:
                    continue
                
                det = {'contour': contour, 'centroid': centroid, 'conf': conf}
                
                if cls == GARLIC_BULB:
                    bulb_detections.append(det)
                elif cls == GARLIC_ROOT:
                    root_detections.append(det)
        
        # Update tracks and get confirmed status for each detection
        bulb_results = self._update_tracks(self.bulb_tracks, bulb_detections)
        root_results = self._update_tracks(self.root_tracks, root_detections)
        
        # Get confirmed root centroids
        confirmed_root_centroids = [
            det['centroid'] for det, confirmed in root_results if confirmed
        ]
        
        # Draw confirmed bulbs using CURRENT frame contours (no lag!)
        output = frame.copy()
        overlay = frame.copy()
        
        for det, confirmed in bulb_results:
            if not confirmed:
                continue
            
            contour = det['contour']
            has_root = any(point_inside_contour(rc, contour) for rc in confirmed_root_centroids)
            color = RED if has_root else BLUE
            
            cv2.fillPoly(overlay, [contour], color)
            cv2.drawContours(output, [contour], -1, color, 2)
        
        return cv2.addWeighted(output, 1.0, overlay, 0.3, 0)
    
    def reset(self):
        self.bulb_tracks = []
        self.root_tracks = []
        Track._next_id = 0


def process_video(video_path, model_path=None, output_path=None):
    video_path = Path(video_path)
    model_path = Path(model_path) if model_path else DEFAULT_MODEL
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if output_path is None:
        output_path = video_path.parent / f"segmented_{video_path.name}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Video:  {video_path}")
    print(f"Model:  {model_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps}fps, {total} frames")
    print(f"Confidence: >{CONFIDENCE_THRESHOLD*100:.0f}%  |  Confirm after: {MIN_VISIBLE_SECONDS}s")
    
    segmenter = GarlicSegmenter(model_path, fps)
    cap = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for _ in tqdm(range(total), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(segmenter.process_frame(frame))
    
    cap.release()
    out.release()
    print(f"✅ Done: {output_path}")


def preview_video(video_path, model_path=None):
    video_path = Path(video_path)
    model_path = Path(model_path) if model_path else DEFAULT_MODEL
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    segmenter = GarlicSegmenter(model_path, fps)
    
    print("Controls: q=quit, space=pause, r=reset")
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Garlic Segmentation", segmenter.process_frame(frame))
        
        key = cv2.waitKey(1000 // fps) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            segmenter.reset()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Garlic Video Segmentation")
        print()
        print("Usage:")
        print("  python detect_video.py <video>                # Process and save")
        print("  python detect_video.py <video> --preview      # Live preview")
        print("  python detect_video.py <video> -o <out>       # Custom output")
        print()
        print(f"Model: {DEFAULT_MODEL}")
        print(f"Confidence: >{CONFIDENCE_THRESHOLD*100:.0f}%  |  Confirm after: {MIN_VISIBLE_SECONDS}s")
        print()
        print("Colors: Blue = right-side up  |  Red = upside-down (root visible)")
        sys.exit(1)
    
    video = sys.argv[1]
    
    if "--preview" in sys.argv:
        preview_video(video)
    elif "-o" in sys.argv:
        idx = sys.argv.index("-o")
        process_video(video, output_path=sys.argv[idx + 1])
    else:
        process_video(video)