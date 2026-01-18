"""
Garlic Video Segmentation
=========================

Detects and highlights garlic bulbs in video footage using YOLO segmentation.

Colors:
    Blue = Garlic (right-side up)
    Red  = Garlic with root visible (upside-down)

Pipeline:
    1. Run YOLO segmentation on each frame
    2. Validate detections (size, shape, confidence)
    3. Track detections across frames by centroid proximity
    4. After confirmation period, draw contours for stable tracks
    5. Check if root centroid is inside bulb to determine orientation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from tqdm import tqdm


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'


# =============================================================================
# CLASS IDS (must match training labels)
# =============================================================================

GARLIC_BULB = 0
GARLIC_ROOT = 1


# =============================================================================
# DISPLAY COLORS (BGR format for OpenCV)
# =============================================================================

COLOR_UPRIGHT = (255, 0, 0)    # Blue - garlic is right-side up
COLOR_INVERTED = (0, 0, 255)  # Red  - garlic is upside-down (root visible)
OVERLAY_ALPHA = 0.3            # Transparency for filled regions


# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

CONFIDENCE_THRESHOLD = 0.70    # Minimum YOLO confidence to consider detection


# =============================================================================
# SHAPE VALIDATION THRESHOLDS
# These filter out obviously wrong detections before tracking
# =============================================================================

# Size constraints (as fraction of frame area)
MIN_AREA_RATIO = 0.001   # Detection must be at least 0.1% of frame
MAX_AREA_RATIO = 0.25    # Detection can't exceed 25% of frame (way too big)

# Absolute pixel limits (fallback for edge cases)
MIN_AREA_PIXELS = 500    # Minimum 500 pixels (filters tiny noise)
MAX_AREA_PIXELS = None   # Set dynamically based on frame size

# Shape constraints
MIN_ASPECT_RATIO = 0.3   # Width/height ratio - filters very elongated shapes
MAX_ASPECT_RATIO = 3.0   # Inverse also applies
MIN_SOLIDITY = 0.5       # Contour area / convex hull area - filters jagged shapes


# =============================================================================
# TRACKING THRESHOLDS
# =============================================================================

MIN_VISIBLE_SECONDS = 0.5  # How long before confirming a track
MAX_MISSING_SECONDS = 0.33 # How long before dropping a lost track
MAX_MATCH_DISTANCE = 100   # Max pixels between frames to consider same object


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def compute_contour_metrics(contour):
    """
    Compute shape metrics for a contour.
    
    Returns dict with:
        - area: contour area in pixels
        - centroid: (x, y) center point
        - aspect_ratio: width / height of bounding rect
        - solidity: area / convex_hull_area (1.0 = perfectly convex)
        - bounding_rect: (x, y, w, h)
    
    Returns None if contour is invalid (e.g., zero area).
    """
    area = cv2.contourArea(contour)
    if area == 0:
        return None
    
    # Centroid from moments
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Bounding rectangle for aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Solidity: how "filled in" the shape is
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return {
        'area': area,
        'centroid': (cx, cy),
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'bounding_rect': (x, y, w, h)
    }


def validate_detection(contour, frame_shape, min_area=None, max_area=None):
    """
    Check if a detection passes all validation criteria.
    
    Args:
        contour: OpenCV contour
        frame_shape: (height, width, channels) of the frame
        min_area: Override minimum area (pixels)
        max_area: Override maximum area (pixels)
    
    Returns:
        (is_valid, metrics, rejection_reason)
        - is_valid: True if detection should be kept
        - metrics: dict from compute_contour_metrics (or None if invalid)
        - rejection_reason: string explaining why rejected (or None if valid)
    """
    metrics = compute_contour_metrics(contour)
    
    # Basic validity check
    if metrics is None:
        return False, None, "invalid_contour"
    
    frame_area = frame_shape[0] * frame_shape[1]
    area = metrics['area']
    
    # --- Size validation ---
    
    # Compute dynamic thresholds based on frame size
    effective_min = max(
        MIN_AREA_PIXELS,
        int(frame_area * MIN_AREA_RATIO)
    )
    effective_max = min(
        MAX_AREA_PIXELS or float('inf'),
        int(frame_area * MAX_AREA_RATIO)
    )
    
    # Allow overrides
    if min_area is not None:
        effective_min = min_area
    if max_area is not None:
        effective_max = max_area
    
    if area < effective_min:
        return False, metrics, f"too_small ({area} < {effective_min})"
    
    if area > effective_max:
        return False, metrics, f"too_large ({area} > {effective_max})"
    
    # --- Shape validation ---
    
    aspect = metrics['aspect_ratio']
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False, metrics, f"bad_aspect_ratio ({aspect:.2f})"
    
    if metrics['solidity'] < MIN_SOLIDITY:
        return False, metrics, f"low_solidity ({metrics['solidity']:.2f})"
    
    # All checks passed
    return True, metrics, None


# =============================================================================
# TRACKING
# =============================================================================

class Track:
    """
    Represents a tracked object across frames.
    
    Lifecycle:
        1. Created when new detection appears
        2. Updated each frame it's matched to a detection
        3. Marked as confirmed after MIN_VISIBLE_SECONDS
        4. Marked as missing when no match found
        5. Deleted after MAX_MISSING_SECONDS of being missing
    """
    _next_id = 0
    
    def __init__(self, centroid):
        self.id = Track._next_id
        Track._next_id += 1
        self.centroid = centroid
        self.frames_seen = 1
        self.frames_missing = 0
        self.confirmed = False
    
    def update(self, centroid):
        """Update track with new detection position."""
        self.centroid = centroid
        self.frames_seen += 1
        self.frames_missing = 0
    
    def mark_missing(self):
        """Called when track has no matching detection this frame."""
        self.frames_missing += 1
    
    @classmethod
    def reset_ids(cls):
        """Reset ID counter (call when starting new video)."""
        cls._next_id = 0


def euclidean_distance(point1, point2):
    """Compute distance between two (x, y) points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def point_inside_contour(point, contour):
    """Check if a point lies inside or on a contour."""
    return cv2.pointPolygonTest(contour, point, False) >= 0


# =============================================================================
# MASK PROCESSING
# =============================================================================

def mask_to_contour(mask, target_shape):
    """
    Convert YOLO mask to OpenCV contour at frame resolution.
    
    Args:
        mask: Binary mask from YOLO (may be different resolution)
        target_shape: (height, width) to resize mask to
    
    Returns:
        Largest contour found, or None if no contours
    """
    # Resize mask to match frame dimensions
    mask_resized = cv2.resize(
        mask, 
        (target_shape[1], target_shape[0]),  # cv2 uses (width, height)
        interpolation=cv2.INTER_NEAREST
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_resized, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Return largest contour (handles cases with multiple disconnected regions)
    return max(contours, key=cv2.contourArea)


# =============================================================================
# MAIN SEGMENTER CLASS
# =============================================================================

class GarlicSegmenter:
    """
    Processes video frames to detect and highlight garlic bulbs.
    
    Usage:
        segmenter = GarlicSegmenter(model_path, fps=30)
        for frame in video:
            output_frame = segmenter.process_frame(frame)
    """
    
    def __init__(self, model_path, fps=30):
        """
        Initialize segmenter.
        
        Args:
            model_path: Path to YOLO model weights
            fps: Video frame rate (used to calculate timing thresholds)
        """
        self.model = YOLO(model_path)
        self.fps = fps
        
        # Convert time thresholds to frame counts
        self.min_frames_to_confirm = int(MIN_VISIBLE_SECONDS * fps)
        self.max_frames_missing = int(MAX_MISSING_SECONDS * fps)
        
        # Separate track lists for each class
        self.bulb_tracks = []
        self.root_tracks = []
        
        # Stats for debugging
        self.stats = {
            'total_detections': 0,
            'rejected_detections': 0,
            'rejection_reasons': {}
        }
    
    def _extract_detections(self, result, frame_shape):
        """
        Extract and validate detections from YOLO result.
        
        Returns:
            (bulb_detections, root_detections)
            Each is a list of dicts with 'contour', 'centroid', 'conf'
        """
        bulb_detections = []
        root_detections = []
        
        if result.masks is None:
            return bulb_detections, root_detections
        
        # Process each detection
        for mask, cls, conf in zip(
            result.masks.data,
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy()
        ):
            self.stats['total_detections'] += 1
            
            # Convert mask to contour
            mask_np = mask.cpu().numpy().astype(np.uint8)
            contour = mask_to_contour(mask_np, frame_shape[:2])
            
            if contour is None:
                self._record_rejection("no_contour")
                continue
            
            # Validate detection
            is_valid, metrics, reason = validate_detection(contour, frame_shape)
            
            if not is_valid:
                self._record_rejection(reason)
                continue
            
            # Build detection dict
            detection = {
                'contour': contour,
                'centroid': metrics['centroid'],
                'conf': conf,
                'area': metrics['area']
            }
            
            # Sort by class
            if cls == GARLIC_BULB:
                bulb_detections.append(detection)
            elif cls == GARLIC_ROOT:
                root_detections.append(detection)
        
        return bulb_detections, root_detections
    
    def _record_rejection(self, reason):
        """Track rejection statistics for debugging."""
        self.stats['rejected_detections'] += 1
        # Extract just the reason name (before any parenthetical details)
        reason_key = reason.split(' ')[0] if reason else 'unknown'
        self.stats['rejection_reasons'][reason_key] = \
            self.stats['rejection_reasons'].get(reason_key, 0) + 1
    
    def _update_tracks(self, tracks, detections):
        """
        Match detections to existing tracks using nearest-neighbor.
        
        Algorithm:
            1. For each detection, find closest track within MAX_MATCH_DISTANCE
            2. Update matched tracks with new position
            3. Create new tracks for unmatched detections
            4. Mark unmatched tracks as missing
            5. Remove tracks that have been missing too long
        
        Returns:
            List of (detection_dict, is_confirmed) pairs
        """
        matched_tracks = set()
        results = []
        
        # Match each detection to closest track
        for detection in detections:
            best_track = None
            best_distance = MAX_MATCH_DISTANCE
            
            for track in tracks:
                if track in matched_tracks:
                    continue
                
                dist = euclidean_distance(detection['centroid'], track.centroid)
                if dist < best_distance:
                    best_distance = dist
                    best_track = track
            
            if best_track:
                # Update existing track
                best_track.update(detection['centroid'])
                
                # Check if track has been seen long enough to confirm
                if best_track.frames_seen >= self.min_frames_to_confirm:
                    best_track.confirmed = True
                
                matched_tracks.add(best_track)
                results.append((detection, best_track.confirmed))
            else:
                # Create new track for unmatched detection
                new_track = Track(detection['centroid'])
                tracks.append(new_track)
                results.append((detection, False))  # Not confirmed yet
        
        # Mark unmatched tracks as missing
        for track in tracks:
            if track not in matched_tracks:
                track.mark_missing()
        
        # Remove tracks that have been missing too long
        tracks[:] = [t for t in tracks if t.frames_missing < self.max_frames_missing]
        
        return results
    
    def process_frame(self, frame):
        """
        Process a single frame and return annotated output.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            Annotated frame with garlic highlighted
        """
        # Run YOLO inference
        result = self.model.predict(
            frame,
            retina_masks=True,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )[0]
        
        # Extract and validate detections
        bulb_detections, root_detections = self._extract_detections(result, frame.shape)
        
        # Update tracking for both classes
        bulb_results = self._update_tracks(self.bulb_tracks, bulb_detections)
        root_results = self._update_tracks(self.root_tracks, root_detections)
        
        # Get centroids of confirmed roots (for orientation check)
        confirmed_root_centroids = [
            det['centroid'] for det, confirmed in root_results if confirmed
        ]
        
        # Draw annotations
        output = frame.copy()
        overlay = frame.copy()
        
        for detection, is_confirmed in bulb_results:
            if not is_confirmed:
                continue  # Skip unconfirmed detections
            
            contour = detection['contour']
            
            # Check if any confirmed root centroid is inside this bulb
            has_visible_root = any(
                point_inside_contour(root_centroid, contour)
                for root_centroid in confirmed_root_centroids
            )
            
            # Color based on orientation
            color = COLOR_INVERTED if has_visible_root else COLOR_UPRIGHT
            
            # Draw filled region on overlay
            cv2.fillPoly(overlay, [contour], color)
            
            # Draw outline on output
            cv2.drawContours(output, [contour], -1, color, 2)
        
        # Blend overlay with output
        return cv2.addWeighted(output, 1.0, overlay, OVERLAY_ALPHA, 0)
    
    def reset(self):
        """Reset all tracking state (call between videos)."""
        self.bulb_tracks = []
        self.root_tracks = []
        Track.reset_ids()
        self.stats = {
            'total_detections': 0,
            'rejected_detections': 0,
            'rejection_reasons': {}
        }
    
    def get_stats(self):
        """Return detection statistics for debugging."""
        return self.stats.copy()


# =============================================================================
# VIDEO PROCESSING FUNCTIONS
# =============================================================================

def process_video(video_path, model_path=None, output_path=None):
    """
    Process entire video and save result.
    
    Args:
        video_path: Input video file
        model_path: YOLO model weights (uses default if None)
        output_path: Output file (auto-generated if None)
    """
    # Resolve paths
    video_path = Path(video_path)
    model_path = Path(model_path) if model_path else DEFAULT_MODEL
    
    # Validate inputs
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Generate output path if not specified
    if output_path is None:
        output_path = video_path.parent / f"segmented_{video_path.name}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Print configuration
    print("=" * 60)
    print("Garlic Video Segmentation")
    print("=" * 60)
    print(f"Input:      {video_path}")
    print(f"Output:     {output_path}")
    print(f"Model:      {model_path}")
    print(f"Resolution: {width}x{height} @ {fps}fps")
    print(f"Frames:     {total_frames}")
    print("-" * 60)
    print("Thresholds:")
    print(f"  Confidence:    > {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Confirm after: {MIN_VISIBLE_SECONDS}s ({int(MIN_VISIBLE_SECONDS * fps)} frames)")
    print(f"  Min area:      {MIN_AREA_RATIO*100:.1f}% of frame or {MIN_AREA_PIXELS}px")
    print(f"  Max area:      {MAX_AREA_RATIO*100:.0f}% of frame")
    print("=" * 60)
    
    # Initialize
    segmenter = GarlicSegmenter(model_path, fps)
    cap = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Process frames
    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(segmenter.process_frame(frame))
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print stats
    stats = segmenter.get_stats()
    print("-" * 60)
    print("Statistics:")
    print(f"  Total detections:    {stats['total_detections']}")
    print(f"  Rejected detections: {stats['rejected_detections']}")
    if stats['rejection_reasons']:
        print("  Rejection reasons:")
        for reason, count in sorted(stats['rejection_reasons'].items()):
            print(f"    {reason}: {count}")
    print("=" * 60)
    print(f"✅ Done: {output_path}")


def preview_video(video_path, model_path=None):
    """
    Live preview with keyboard controls.
    
    Controls:
        q     - Quit
        Space - Pause/resume
        r     - Reset tracking
    """
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
    
    print("=" * 40)
    print("Live Preview")
    print("=" * 40)
    print("Controls:")
    print("  q     - Quit")
    print("  Space - Pause/resume")
    print("  r     - Reset tracking")
    print("=" * 40)
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            output = segmenter.process_frame(frame)
            cv2.imshow("Garlic Segmentation (q=quit, space=pause, r=reset)", output)
        
        key = cv2.waitKey(1000 // fps) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            segmenter.reset()
            print("Tracking reset")
    
    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def print_help():
    """Print usage information."""
    print("""
Garlic Video Segmentation
=========================

Detects garlic bulbs in video and highlights their orientation.
  Blue = right-side up (bulb only visible)
  Red  = upside-down (root visible)

Usage:
  python detect_video.py <video>              Process and save
  python detect_video.py <video> --preview    Live preview
  python detect_video.py <video> -o <out>     Custom output path

Configuration:
  Model:      {model}
  Confidence: > {conf:.0f}%
  Confirm:    {confirm}s visible before drawing
  Min size:   {min_pct:.1f}% of frame or {min_px}px
  Max size:   {max_pct:.0f}% of frame

The tracking system filters out noise by requiring detections to
persist for {confirm}s before displaying them. Detections that are
too small, too large, or have unusual shapes are also filtered out.
""".format(
        model=DEFAULT_MODEL,
        conf=CONFIDENCE_THRESHOLD * 100,
        confirm=MIN_VISIBLE_SECONDS,
        min_pct=MIN_AREA_RATIO * 100,
        min_px=MIN_AREA_PIXELS,
        max_pct=MAX_AREA_RATIO * 100
    ))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    video = sys.argv[1]
    
    if "--preview" in sys.argv:
        preview_video(video)
    elif "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 >= len(sys.argv):
            print("❌ Error: -o requires an output path")
            sys.exit(1)
        process_video(video, output_path=sys.argv[idx + 1])
    else:
        process_video(video)