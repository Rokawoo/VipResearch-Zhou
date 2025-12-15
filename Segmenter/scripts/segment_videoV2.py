"""
Video Segmentation Script for Garlic Detection with Single-Garlic Tracking
Location: scripts/detect_video.py

Processes videos with special logic:
- Garlic Bulb: Blue (if no root detected inside)
- Garlic Bulb: Red (if root centroid is inside the bulb)
- Garlic Root: Not drawn (only used to detect overlap)
- Gripper: Green
- Width Points: Purple dots on the LEFT and RIGHT sides of the garlic (horizontal)
- Tracking: Tracks ONE garlic at a time with visual highlight

Tracking Logic:
1. Wait until garlic is detected
2. Gather candidates for 1-2 seconds
3. Select best garlic (biggest + closest to gripper)
4. Track until out of frame, then repeat
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Class IDs
CLASS_GARLIC_BULB = 0
CLASS_GARLIC_ROOT = 1
CLASS_GRIPPER = 2

# Colors (BGR)
COLOR_BULB_NORMAL = (255, 0, 0)      # Blue
COLOR_BULB_WITH_ROOT = (0, 0, 255)   # Red
COLOR_GRIPPER = (0, 255, 0)          # Green
COLOR_WIDTH_POINT = (255, 0, 255)    # Purple (Magenta)
COLOR_TRACKED = (0, 255, 255)        # Yellow - for tracked garlic outline
COLOR_TRACKED_FILL = (0, 200, 200)   # Darker yellow for fill

# Width point visualization settings
WIDTH_POINT_RADIUS = 8
WIDTH_POINT_THICKNESS = -1  # Filled circle

# Horizontal constraint for width points
WIDTH_POINTS_MAX_Y_DIFF_RATIO = 0.4   # Max Y difference as ratio of garlic height
WIDTH_POINTS_MIN_X_DIFF_RATIO = 0.3   # Min X difference as ratio of garlic width

# Tracking settings
TRACKING_WAIT_INITIAL_SEC = 1.0      # Wait this long before starting to gather
TRACKING_GATHER_SEC = 1.5            # Gather candidates for this long
TRACKING_IOU_THRESHOLD = 0.15        # Minimum IoU to consider same garlic
TRACKING_CENTROID_MAX_DIST = 150     # Max centroid movement (pixels) per frame
TRACKING_LOST_FRAMES = 10            # Frames without match before considered lost

# Selection weights - prioritize bottom of screen, then size, then confidence
SELECTION_BOTTOM_WEIGHT = 0.5        # Weight for closeness to bottom of screen (higher Y = better)
SELECTION_AREA_WEIGHT = 0.35         # Weight for area (bigger = better)
SELECTION_CONF_WEIGHT = 0.15         # Weight for confidence (minor tiebreaker)


class TrackingState(Enum):
    """State machine states for garlic tracking"""
    WAITING = "waiting"
    GATHERING = "gathering"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass
class GarlicCandidate:
    """Represents a detected garlic with its properties"""
    contour: np.ndarray
    centroid: Tuple[int, int]
    area: float
    confidence: float
    has_root: bool
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass 
class TrackedGarlic:
    """State for the currently tracked garlic"""
    contour: np.ndarray
    centroid: Tuple[int, int]
    area: float
    mask: np.ndarray
    frames_since_seen: int = 0
    
    def update(self, candidate: GarlicCandidate):
        """Update tracked garlic with new detection"""
        self.contour = candidate.contour
        self.centroid = candidate.centroid
        self.area = candidate.area
        self.mask = candidate.mask
        self.frames_since_seen = 0


class GarlicTracker:
    """
    Manages single-garlic tracking with state machine logic.
    Selects ONE garlic based on: biggest area + closest to gripper.
    """
    
    def __init__(self, fps: int = 30, frame_height: int = 1080):
        self.fps = fps
        self.frame_height = frame_height
        self.state = TrackingState.WAITING
        self.tracked: Optional[TrackedGarlic] = None
        
        # Timing (in frames)
        self.wait_frames = int(TRACKING_WAIT_INITIAL_SEC * fps)
        self.gather_frames = int(TRACKING_GATHER_SEC * fps)
        
        # Counters
        self.frames_waiting = 0
        self.frames_gathering = 0
        
        # Candidate collection during GATHERING
        self.candidates_seen: List[GarlicCandidate] = []
        self.best_candidate: Optional[GarlicCandidate] = None
        
    def reset(self):
        """Reset tracker to initial state"""
        self.state = TrackingState.WAITING
        self.tracked = None
        self.frames_waiting = 0
        self.frames_gathering = 0
        self.candidates_seen = []
        self.best_candidate = None
        
    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    
    def compute_centroid_dist(self, c1: Tuple[int, int], c2: Tuple[int, int]) -> float:
        """Compute Euclidean distance between centroids"""
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    def find_best_match(self, candidates: List[GarlicCandidate]) -> Optional[GarlicCandidate]:
        """Find the candidate that best matches the currently tracked garlic."""
        if not self.tracked or not candidates:
            return None
        
        best_match = None
        best_score = -1
        
        for candidate in candidates:
            iou = self.compute_iou(self.tracked.mask, candidate.mask)
            dist = self.compute_centroid_dist(self.tracked.centroid, candidate.centroid)
            
            if iou >= TRACKING_IOU_THRESHOLD or dist <= TRACKING_CENTROID_MAX_DIST:
                dist_score = max(0, 1 - dist / TRACKING_CENTROID_MAX_DIST)
                score = iou * 0.7 + dist_score * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = candidate
        
        return best_match
    
    def select_best_candidate(self, frame_height: int) -> Optional[GarlicCandidate]:
        """
        Select the ONE best garlic from gathered candidates.
        Priority: 
          1. Closest to bottom of screen (higher Y = closer to camera)
          2. Biggest area
          3. Highest confidence (minor tiebreaker)
        """
        if not self.candidates_seen:
            return None
        
        # Find max values for normalization
        max_area = max(c.area for c in self.candidates_seen)
        max_y = max(c.centroid[1] for c in self.candidates_seen)
        min_y = min(c.centroid[1] for c in self.candidates_seen)
        y_range = max_y - min_y if max_y != min_y else 1
        
        # Score all candidates
        best = None
        best_score = -1
        
        for candidate in self.candidates_seen:
            # Normalize Y position (0-1, higher Y = higher score = closer to bottom)
            # Use centroid Y relative to frame height for absolute positioning
            bottom_score = candidate.centroid[1] / frame_height
            
            # Normalize area (0-1, bigger = higher)
            area_score = candidate.area / max_area if max_area > 0 else 0
            
            # Confidence is already 0-1
            conf_score = candidate.confidence
            
            # Combined score
            score = (SELECTION_BOTTOM_WEIGHT * bottom_score + 
                    SELECTION_AREA_WEIGHT * area_score + 
                    SELECTION_CONF_WEIGHT * conf_score)
            
            if score > best_score:
                best_score = score
                best = candidate
        
        return best
    
    def update(
        self, 
        candidates: List[GarlicCandidate]
    ) -> Tuple[TrackingState, Optional[TrackedGarlic]]:
        """
        Update tracker with new frame's detections.
        """
        if self.state == TrackingState.WAITING:
            if candidates:
                self.state = TrackingState.GATHERING
                self.frames_gathering = 0
                self.candidates_seen = candidates.copy()
                self.best_candidate = self.select_best_candidate(self.frame_height)
            else:
                self.frames_waiting += 1
                
        elif self.state == TrackingState.GATHERING:
            if candidates:
                for c in candidates:
                    is_new = True
                    for i, existing in enumerate(self.candidates_seen):
                        if self.compute_centroid_dist(c.centroid, existing.centroid) < 50:
                            # Update existing if bigger
                            if c.area > existing.area:
                                self.candidates_seen[i] = c
                            is_new = False
                            break
                    if is_new:
                        self.candidates_seen.append(c)
                
                self.best_candidate = self.select_best_candidate(self.frame_height)
            
            self.frames_gathering += 1
            
            if self.frames_gathering >= self.gather_frames:
                if self.best_candidate:
                    self.tracked = TrackedGarlic(
                        contour=self.best_candidate.contour,
                        centroid=self.best_candidate.centroid,
                        area=self.best_candidate.area,
                        mask=self.best_candidate.mask
                    )
                    self.state = TrackingState.TRACKING
                    self.candidates_seen = []
                else:
                    self.reset()
                    
        elif self.state == TrackingState.TRACKING:
            match = self.find_best_match(candidates)
            
            if match:
                self.tracked.update(match)
            else:
                self.tracked.frames_since_seen += 1
                
                if self.tracked.frames_since_seen >= TRACKING_LOST_FRAMES:
                    self.state = TrackingState.LOST
                    
        elif self.state == TrackingState.LOST:
            self.reset()
        
        return self.state, self.tracked
    
    def get_status_text(self) -> str:
        """Get human-readable status for display"""
        if self.state == TrackingState.WAITING:
            return f"Waiting for garlic... ({self.frames_waiting} frames)"
        elif self.state == TrackingState.GATHERING:
            remaining = self.gather_frames - self.frames_gathering
            return f"Gathering: {len(self.candidates_seen)} found ({remaining} frames left)"
        elif self.state == TrackingState.TRACKING:
            return f"TRACKING (lost for {self.tracked.frames_since_seen} frames)" if self.tracked else "TRACKING"
        elif self.state == TrackingState.LOST:
            return "Lost tracking, resetting..."
        return "Unknown"


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using OpenCV"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def find_horizontal_width_points(contour):
    """
    Find the two points on the LEFT and RIGHT horizontal sides of the garlic.
    
    Algorithm:
    1. Define a horizontal band around the vertical center of the garlic
    2. Find the leftmost and rightmost contour points within this band
    3. Validate that the points are horizontally separated, not vertically
    
    This ensures dots are always on the SIDES regardless of garlic orientation.
    
    Args:
        contour: OpenCV contour (Nx1x2 array)
        
    Returns:
        Tuple of two points ((x1, y1), (x2, y2)) - left and right side points
    """
    if len(contour) < 5:
        return None, None
    
    try:
        contour_pts = contour.reshape(-1, 2).astype(np.float64)
        
        # Get bounding box dimensions
        x_min, y_min = contour_pts.min(axis=0)
        x_max, y_max = contour_pts.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        if width == 0 or height == 0:
            return None, None
        
        # Get centroid Y position
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cy = (y_min + y_max) / 2
        else:
            cy = M["m01"] / M["m00"]
        
        # Define horizontal band: ±30% of height around centroid
        band_half = height * 0.35
        band_top = cy - band_half
        band_bottom = cy + band_half
        
        # Find points within the horizontal band
        in_band_mask = (contour_pts[:, 1] >= band_top) & (contour_pts[:, 1] <= band_bottom)
        band_pts = contour_pts[in_band_mask]
        
        if len(band_pts) < 2:
            # Expand band if too few points
            band_half = height * 0.45
            band_top = cy - band_half
            band_bottom = cy + band_half
            in_band_mask = (contour_pts[:, 1] >= band_top) & (contour_pts[:, 1] <= band_bottom)
            band_pts = contour_pts[in_band_mask]
        
        if len(band_pts) < 2:
            # Fall back to simple left/right extremes
            left_idx = np.argmin(contour_pts[:, 0])
            right_idx = np.argmax(contour_pts[:, 0])
            return tuple(contour_pts[left_idx].astype(int)), tuple(contour_pts[right_idx].astype(int))
        
        # Find leftmost and rightmost points in the band
        left_idx = np.argmin(band_pts[:, 0])
        right_idx = np.argmax(band_pts[:, 0])
        
        left_pt = band_pts[left_idx]
        right_pt = band_pts[right_idx]
        
        # Validate horizontal alignment
        y_diff = abs(left_pt[1] - right_pt[1])
        x_diff = abs(left_pt[0] - right_pt[0])
        
        # Check constraints
        if y_diff > height * WIDTH_POINTS_MAX_Y_DIFF_RATIO:
            # Points too vertically separated, try to fix
            # Find points closer to centroid Y that are still left/right extremes
            target_y = cy
            
            # For left point: find leftmost point closest to target_y
            left_candidates = band_pts[band_pts[:, 0] < (x_min + width * 0.4)]
            if len(left_candidates) > 0:
                y_dists = np.abs(left_candidates[:, 1] - target_y)
                best_left_idx = np.argmin(y_dists)
                left_pt = left_candidates[best_left_idx]
            
            # For right point: find rightmost point closest to target_y
            right_candidates = band_pts[band_pts[:, 0] > (x_max - width * 0.4)]
            if len(right_candidates) > 0:
                y_dists = np.abs(right_candidates[:, 1] - target_y)
                best_right_idx = np.argmin(y_dists)
                right_pt = right_candidates[best_right_idx]
        
        if x_diff < width * WIDTH_POINTS_MIN_X_DIFF_RATIO:
            # Points not horizontally separated enough, fall back to extremes
            left_idx = np.argmin(contour_pts[:, 0])
            right_idx = np.argmax(contour_pts[:, 0])
            left_pt = contour_pts[left_idx]
            right_pt = contour_pts[right_idx]
        
        return tuple(left_pt.astype(int)), tuple(right_pt.astype(int))
        
    except Exception:
        return None, None


def find_width_points_extreme(contour):
    """
    Simpler fallback: Find points with maximum horizontal separation
    that are roughly at the same vertical level.
    
    Scans through pairs to find the best horizontal spread.
    """
    if len(contour) < 5:
        return None, None
    
    try:
        contour_pts = contour.reshape(-1, 2).astype(np.float64)
        n = len(contour_pts)
        
        # Get dimensions for thresholds
        y_min, y_max = contour_pts[:, 1].min(), contour_pts[:, 1].max()
        height = y_max - y_min
        max_y_diff = height * 0.35  # Points must be within 35% of height
        
        # Find pair with max X separation and acceptable Y difference
        best_pair = None
        best_x_diff = 0
        
        # Sample points for efficiency
        if n > 50:
            indices = np.linspace(0, n - 1, 50, dtype=int)
            sample_pts = contour_pts[indices]
        else:
            sample_pts = contour_pts
        
        n_sample = len(sample_pts)
        
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                p1, p2 = sample_pts[i], sample_pts[j]
                
                y_diff = abs(p1[1] - p2[1])
                x_diff = abs(p1[0] - p2[0])
                
                # Check if horizontally aligned enough
                if y_diff <= max_y_diff and x_diff > best_x_diff:
                    best_x_diff = x_diff
                    # Ensure left point first
                    if p1[0] < p2[0]:
                        best_pair = (tuple(p1.astype(int)), tuple(p2.astype(int)))
                    else:
                        best_pair = (tuple(p2.astype(int)), tuple(p1.astype(int)))
        
        if best_pair:
            return best_pair
        
        # Ultimate fallback: just use left/right extremes
        left_idx = np.argmin(contour_pts[:, 0])
        right_idx = np.argmax(contour_pts[:, 0])
        return tuple(contour_pts[left_idx].astype(int)), tuple(contour_pts[right_idx].astype(int))
        
    except Exception:
        return None, None


def get_width_points(contour):
    """
    Get the horizontal width points (left and right sides).
    Uses band-based method with fallback to extreme search.
    """
    p1, p2 = find_horizontal_width_points(contour)
    
    if p1 is not None and p2 is not None:
        # Final validation: ensure horizontal alignment
        y_diff = abs(p1[1] - p2[1])
        x_diff = abs(p1[0] - p2[0])
        
        # If still too vertical, try fallback
        if x_diff > 0 and y_diff / x_diff > 0.8:  # More than ~40 degrees from horizontal
            p1, p2 = find_width_points_extreme(contour)
    
    if p1 is None or p2 is None:
        p1, p2 = find_width_points_extreme(contour)
    
    return p1, p2


def validate_width_points(p1, p2, contour):
    """
    Validate that width points are horizontally aligned.
    Returns True if valid, False if points are too vertical.
    """
    if p1 is None or p2 is None:
        return False
    
    y_diff = abs(p1[1] - p2[1])
    x_diff = abs(p1[0] - p2[0])
    
    # Must have significant horizontal separation
    if x_diff < 10:
        return False
    
    # Y difference should be small relative to X difference
    # tan(45°) = 1, so y_diff/x_diff < 0.7 means roughly < 35 degrees from horizontal
    if y_diff / x_diff > 0.7:
        return False
    
    return True


def extract_garlic_candidates(
    frame: np.ndarray,
    masks,
    boxes,
    classes: np.ndarray,
    root_centroids: List[Tuple[int, int]]
) -> List[GarlicCandidate]:
    """Extract garlic candidates from detection results."""
    candidates = []
    confidences = boxes.conf.cpu().numpy()
    
    for i, (mask, cls) in enumerate(zip(masks, classes)):
        if cls != CLASS_GARLIC_BULB:
            continue
            
        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        
        contours, _ = cv2.findContours(
            mask_resized, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
        
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        bbox = (x, y, w, h)
        
        has_root = any(point_in_polygon(rc, main_contour) for rc in root_centroids)
        
        candidates.append(GarlicCandidate(
            contour=main_contour,
            centroid=centroid,
            area=area,
            confidence=confidences[i],
            has_root=has_root,
            mask=mask_resized,
            bbox=bbox
        ))
    
    return candidates


def process_frame_with_tracking(
    frame: np.ndarray,
    model,
    tracker: GarlicTracker,
    conf_threshold: float = 0.25,
    fill_alpha: float = 0.3,
    draw_width_points: bool = True,
    show_status: bool = True
) -> np.ndarray:
    """
    Process a single frame with segmentation and tracking logic.
    Only draws width points on the tracked garlic.
    """
    img = frame.copy()
    overlay = frame.copy()
    
    results = model.predict(frame, retina_masks=True, conf=conf_threshold, verbose=False)
    
    root_centroids = []
    gripper_data = []
    
    if results and results[0].masks is not None:
        result = results[0]
        masks = result.masks.data
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        
        # First pass: collect roots and grippers
        for mask, cls in zip(masks, classes):
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            
            contours, _ = cv2.findContours(
                mask_resized, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
            else:
                centroid = None
            
            if cls == CLASS_GARLIC_ROOT and centroid:
                root_centroids.append(centroid)
            elif cls == CLASS_GRIPPER:
                gripper_data.append({'contour': main_contour, 'centroid': centroid})
        
        # Extract garlic candidates
        candidates = extract_garlic_candidates(frame, masks, boxes, classes, root_centroids)
    else:
        candidates = []
    
    # Update tracker
    state, tracked = tracker.update(candidates)
    
    # Draw all garlic bulbs (dimmed if not tracked)
    width_points_to_draw = []
    
    for candidate in candidates:
        contour = candidate.contour
        
        # Check if this is the tracked garlic
        is_tracked = False
        if tracked and tracker.state == TrackingState.TRACKING:
            iou = tracker.compute_iou(tracked.mask, candidate.mask)
            dist = tracker.compute_centroid_dist(tracked.centroid, candidate.centroid)
            is_tracked = (iou >= TRACKING_IOU_THRESHOLD or dist <= TRACKING_CENTROID_MAX_DIST / 2)
        
        if is_tracked:
            # Tracked garlic: bright yellow highlight
            color = COLOR_TRACKED_FILL if not candidate.has_root else (0, 100, 255)
            outline_color = COLOR_TRACKED
            thickness = 4
            
            # Calculate width points ONLY for tracked garlic
            if draw_width_points:
                p1, p2 = get_width_points(contour)
                if validate_width_points(p1, p2, contour):
                    width_points_to_draw.append((p1, p2))
        else:
            # Non-tracked garlic: dimmed colors
            base_color = COLOR_BULB_WITH_ROOT if candidate.has_root else COLOR_BULB_NORMAL
            color = tuple(int(c * 0.4) for c in base_color)
            outline_color = tuple(int(c * 0.5) for c in base_color)
            thickness = 1
        
        cv2.fillPoly(overlay, [contour], color)
        cv2.drawContours(img, [contour], -1, outline_color, thickness)
    
    # Draw "gathering" indicator on best candidate
    if tracker.state == TrackingState.GATHERING and tracker.best_candidate:
        bc = tracker.best_candidate
        pulse = int(10 + 5 * np.sin(tracker.frames_gathering * 0.3))
        cv2.circle(img, bc.centroid, pulse + 20, (255, 255, 0), 2)
        cv2.putText(img, "SELECTING", (bc.centroid[0] - 40, bc.centroid[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw grippers
    for gripper in gripper_data:
        contour = gripper['contour']
        cv2.fillPoly(overlay, [contour], COLOR_GRIPPER)
        cv2.drawContours(img, [contour], -1, COLOR_GRIPPER, 2)
    
    # Blend overlay
    img = cv2.addWeighted(img, 1.0, overlay, fill_alpha, 0)
    
    # Draw width points on top (only for tracked garlic)
    for p1, p2 in width_points_to_draw:
        # Draw filled circles
        cv2.circle(img, p1, WIDTH_POINT_RADIUS, COLOR_WIDTH_POINT, WIDTH_POINT_THICKNESS)
        cv2.circle(img, p2, WIDTH_POINT_RADIUS, COLOR_WIDTH_POINT, WIDTH_POINT_THICKNESS)
        # Black outline for visibility
        cv2.circle(img, p1, WIDTH_POINT_RADIUS, (0, 0, 0), 2)
        cv2.circle(img, p2, WIDTH_POINT_RADIUS, (0, 0, 0), 2)
        
        # Optional: draw line between points to show width axis
        # cv2.line(img, p1, p2, COLOR_WIDTH_POINT, 1)
    
    # Draw tracking status overlay
    if show_status:
        status_text = tracker.get_status_text()
        
        status_bg_color = {
            TrackingState.WAITING: (100, 100, 100),
            TrackingState.GATHERING: (0, 150, 255),
            TrackingState.TRACKING: (0, 200, 0),
            TrackingState.LOST: (0, 0, 200)
        }.get(tracker.state, (100, 100, 100))
        
        cv2.rectangle(img, (10, 10), (400, 45), status_bg_color, -1)
        cv2.rectangle(img, (10, 10), (400, 45), (255, 255, 255), 2)
        cv2.putText(img, status_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if tracked and tracker.state == TrackingState.TRACKING:
            info_text = f"Area: {tracked.area:.0f}px  Pos: {tracked.centroid}"
            cv2.putText(img, info_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def process_frame(frame, model, conf_threshold=0.25, fill_alpha=0.3, draw_width_points=True):
    """
    Process a single frame WITHOUT tracking (legacy function).
    """
    img = frame.copy()
    overlay = frame.copy()
    
    results = model.predict(frame, retina_masks=True, conf=conf_threshold, verbose=False)
    
    if not results or results[0].masks is None:
        return img
    
    result = results[0]
    masks = result.masks.data
    boxes = result.boxes
    classes = boxes.cls.cpu().numpy().astype(int)
    
    bulb_data = []
    root_centroids = []
    gripper_data = []
    
    for mask, cls in zip(masks, classes):
        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        
        contours, _ = cv2.findContours(
            mask_resized, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
        
        main_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            centroid = None
        
        if cls == CLASS_GARLIC_BULB:
            bulb_data.append({'contour': main_contour, 'centroid': centroid})
        elif cls == CLASS_GARLIC_ROOT:
            if centroid:
                root_centroids.append(centroid)
        elif cls == CLASS_GRIPPER:
            gripper_data.append({'contour': main_contour, 'centroid': centroid})
    
    width_points_to_draw = []
    
    for bulb in bulb_data:
        contour = bulb['contour']
        
        has_root = any(point_in_polygon(rc, contour) for rc in root_centroids)
        color = COLOR_BULB_WITH_ROOT if has_root else COLOR_BULB_NORMAL
        
        cv2.fillPoly(overlay, [contour], color)
        cv2.drawContours(img, [contour], -1, color, 2)
        
        if draw_width_points:
            p1, p2 = get_width_points(contour)
            if validate_width_points(p1, p2, contour):
                width_points_to_draw.append((p1, p2))
    
    for gripper in gripper_data:
        contour = gripper['contour']
        cv2.fillPoly(overlay, [contour], COLOR_GRIPPER)
        cv2.drawContours(img, [contour], -1, COLOR_GRIPPER, 2)
    
    img = cv2.addWeighted(img, 1.0, overlay, fill_alpha, 0)
    
    for p1, p2 in width_points_to_draw:
        cv2.circle(img, p1, WIDTH_POINT_RADIUS, COLOR_WIDTH_POINT, WIDTH_POINT_THICKNESS)
        cv2.circle(img, p2, WIDTH_POINT_RADIUS, COLOR_WIDTH_POINT, WIDTH_POINT_THICKNESS)
        cv2.circle(img, p1, WIDTH_POINT_RADIUS, (0, 0, 0), 2)
        cv2.circle(img, p2, WIDTH_POINT_RADIUS, (0, 0, 0), 2)
    
    return img


def process_video(
    video_path,
    model_path=None,
    output_path=None,
    conf_threshold=0.5,
    fill_alpha=0.3,
    show_progress=True,
    draw_width_points=True,
    enable_tracking=True
):
    """
    Process video with garlic segmentation and optional tracking.
    """
    
    if model_path is None:
        model_path = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'
    else:
        model_path = Path(model_path)
    
    video_path = Path(video_path)
    
    if output_path is None:
        output_dir = PROJECT_ROOT / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"segmented_{video_path.stem}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        test_path = SCRIPT_DIR / video_path
        if test_path.exists():
            video_path = test_path
        else:
            test_path = PROJECT_ROOT / video_path
            if test_path.exists():
                video_path = test_path
            else:
                print(f"❌ Error: Video not found: {video_path}")
                return None
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        return None
    
    print(f"✅ Video: {video_path}")
    print(f"✅ Model: {model_path}")
    print(f"✅ Output: {output_path}")
    
    print(f"\n🤖 Loading model...")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video")
        cap.release()
        return None
    
    tracker = GarlicTracker(fps=fps, frame_height=height) if enable_tracking else None
    
    print(f"\n🎬 Processing video...")
    print(f"   🟣 Width points: {'ENABLED' if draw_width_points else 'DISABLED'}")
    print(f"   🎯 Tracking: {'ENABLED' if enable_tracking else 'DISABLED'}")
    
    frame_count = 0
    
    if show_progress:
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if enable_tracking and tracker:
            processed_frame = process_frame_with_tracking(
                frame, model, tracker, conf_threshold, fill_alpha, draw_width_points
            )
        else:
            processed_frame = process_frame(
                frame, model, conf_threshold, fill_alpha, draw_width_points
            )
        
        out.write(processed_frame)
        
        frame_count += 1
        
        if show_progress:
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    cap.release()
    out.release()
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {frame_count} frames")
    print(f"   Output: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path


def process_video_with_preview(
    video_path,
    model_path=None,
    conf_threshold=0.25,
    fill_alpha=0.3,
    draw_width_points=True,
    enable_tracking=True
):
    """
    Process video with live preview.
    Controls: q=quit, space=pause, w=width points, t=tracking, r=reset
    """
    
    if model_path is None:
        model_path = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'
    else:
        model_path = Path(model_path)
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        test_path = SCRIPT_DIR / video_path
        if test_path.exists():
            video_path = test_path
        else:
            test_path = PROJECT_ROOT / video_path
            if test_path.exists():
                video_path = test_path
    
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        return
    
    print(f"✅ Video: {video_path}")
    print(f"✅ Model: {model_path}")
    
    print(f"\n🤖 Loading model...")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = GarlicTracker(fps=fps, frame_height=frame_height)
    
    print(f"\n📹 Playing at {fps} FPS")
    print("   q = quit")
    print("   space = pause/resume")
    print("   w = toggle width points")
    print("   t = toggle tracking")
    print("   r = reset tracker")
    
    paused = False
    show_width = draw_width_points
    tracking_enabled = enable_tracking
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video finished")
                break
            
            if tracking_enabled:
                processed_frame = process_frame_with_tracking(
                    frame, model, tracker, conf_threshold, fill_alpha, show_width
                )
            else:
                processed_frame = process_frame(
                    frame, model, conf_threshold, fill_alpha, show_width
                )
            
            cv2.imshow('Garlic Segmentation (q=quit, space=pause, w=width, t=track, r=reset)', 
                      processed_frame)
        
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Stopped by user")
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸️  Paused" if paused else "▶️  Resumed")
        elif key == ord('w'):
            show_width = not show_width
            print(f"🟣 Width points: {'ON' if show_width else 'OFF'}")
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            if tracking_enabled:
                tracker.reset()
            print(f"🎯 Tracking: {'ON' if tracking_enabled else 'OFF'}")
        elif key == ord('r'):
            tracker.reset()
            print("🔄 Tracker reset")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GARLIC VIDEO SEGMENTATION WITH TRACKING")
    print("="*60)
    print(f"Script: {SCRIPT_DIR}")
    print(f"Project: {PROJECT_ROOT}")
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        preview_mode = '--preview' in sys.argv
        no_tracking = '--no-tracking' in sys.argv
        
        if preview_mode:
            print("\n🎥 Preview mode (no output file)")
            process_video_with_preview(video_path, enable_tracking=not no_tracking)
        else:
            print("\n🎥 Processing mode (will save output)")
            
            output_path = None
            for arg in sys.argv[2:]:
                if not arg.startswith('--'):
                    output_path = arg
                    break
            
            process_video(video_path, output_path=output_path, enable_tracking=not no_tracking)
    else:
        print("\n" + "="*60)
        print("USAGE:")
        print("="*60)
        print("\nProcess and save video:")
        print("  python detect_video.py <video_path>")
        print("  python detect_video.py <video_path> <output_path>")
        print("  python detect_video.py <video_path> --no-tracking")
        print("\nPreview mode (no output):")
        print("  python detect_video.py <video_path> --preview")
        print("  python detect_video.py <video_path> --preview --no-tracking")
        print("\nExamples:")
        print("  python detect_video.py ../videos/garlic.mp4")
        print("  python detect_video.py videos/garlic.mp4 results/output.mp4")
        print("  python detect_video.py videos/garlic.mp4 --preview")
        print("\n" + "="*60)
        print("SEGMENTATION LOGIC:")
        print("="*60)
        print("  🔵 Garlic Bulb - Blue (normal, dimmed if not tracked)")
        print("  🔴 Garlic Bulb - Red (contains root, dimmed if not tracked)")
        print("  🟡 Tracked Garlic - Yellow highlight (actively tracked)")
        print("  🟢 Gripper - Green")
        print("  ⚫ Garlic Root - Not drawn (only used for detection)")
        print("  🟣 Width Points - Purple dots on LEFT/RIGHT sides of tracked garlic")
        print("\n" + "="*60)
        print("TRACKING LOGIC:")
        print("="*60)
        print(f"  1. Wait {TRACKING_WAIT_INITIAL_SEC}s for garlic to appear")
        print(f"  2. Gather candidates for {TRACKING_GATHER_SEC}s")
        print(f"  3. Select ONE garlic:")
        print(f"     - Closest to bottom ({SELECTION_BOTTOM_WEIGHT*100:.0f}%)")
        print(f"     - Biggest size ({SELECTION_AREA_WEIGHT*100:.0f}%)")
        print(f"     - Highest confidence ({SELECTION_CONF_WEIGHT*100:.0f}%)")
        print("  4. Track until lost, then repeat")
        print("="*60)