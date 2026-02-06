from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from label_studio_tools.core.utils.io import get_local_path
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

MODEL_PATH = r"C:\Users\Roka\Dev\GitProjects\MyProjects\VipResearch-Zhou\Segmenter\scripts\runs\garlic_seg\final\weights\best.pt"

LABEL_MAP = {
    "garlic_bulb": "Garlic Bulb",
    "garlic_root": "Garlic Root",
}

# -------- TUNING --------
CONF_THRESHOLD = 0.3        # Minimum confidence to keep
IOU_THRESHOLD = 0.4         # NMS IoU threshold
SIMPLIFY_TOLERANCE = 0.3    # Polygon simplification (% space)
MIN_AREA = 0.3              # Minimum polygon area (% of image area)
MERGE_IOU_THRESHOLD = 0.3   # Merge overlapping same-class polygons above this IoU
NECK_EROSION = 1.0          # Erosion amount to sever thin necks (% space, higher = more aggressive)
SMALL_FRAGMENT_RATIO = 0.15 # After severing necks, drop fragments smaller than this ratio of the largest piece
# -------------------------


def safe_poly(points):
    """Create a valid shapely polygon from points."""
    if len(points) < 3:
        return None
    poly = ShapelyPolygon(points)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None
    # make_valid can return GeometryCollection, extract polygon
    if poly.geom_type == "Polygon":
        return poly
    elif poly.geom_type == "MultiPolygon":
        return max(poly.geoms, key=lambda g: g.area)
    elif poly.geom_type == "GeometryCollection":
        polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
        return max(polys, key=lambda g: g.area) if polys else None
    return None


def remove_thin_extensions(poly):
    """
    Morphological opening (erode then dilate) to sever thin necks/bridges.
    Then keep the largest fragment and any big siblings, discard small blobs.
    """
    if poly is None or poly.is_empty:
        return poly

    # Erode: thin necks disappear
    eroded = poly.buffer(-NECK_EROSION)
    if eroded.is_empty:
        # Entire polygon was thinner than the erosion — keep original
        return poly

    # Dilate back: restore the main body shape
    opened = eroded.buffer(NECK_EROSION)

    if opened.is_empty:
        return poly

    # If opening split it into pieces, keep the big ones
    if opened.geom_type == "MultiPolygon":
        pieces = sorted(opened.geoms, key=lambda g: g.area, reverse=True)
        largest_area = pieces[0].area
        # Keep pieces that are significant relative to the largest
        keepers = [p for p in pieces if p.area >= largest_area * SMALL_FRAGMENT_RATIO]
        opened = unary_union(keepers)

    # Intersect with original to avoid expanding beyond the original boundary
    result = poly.intersection(opened)

    if result.is_empty:
        return poly
    if result.geom_type == "MultiPolygon":
        result = max(result.geoms, key=lambda g: g.area)
    if result.geom_type != "Polygon":
        return poly

    return result


def polygon_iou(p1, p2):
    """Compute IoU between two shapely polygons."""
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def merge_overlapping(predictions):
    """Merge overlapping polygons of the same class."""
    if len(predictions) <= 1:
        return predictions

    groups = {}
    for pred in predictions:
        label = pred["value"]["polygonlabels"][0]
        groups.setdefault(label, []).append(pred)

    merged = []
    for label, preds in groups.items():
        polys = []
        for p in preds:
            pts = p["value"]["points"]
            sp = safe_poly(pts)
            if sp is not None:
                polys.append((sp, p))

        used = [False] * len(polys)
        for i in range(len(polys)):
            if used[i]:
                continue
            cluster = [i]
            for j in range(i + 1, len(polys)):
                if used[j]:
                    continue
                if polygon_iou(polys[i][0], polys[j][0]) > MERGE_IOU_THRESHOLD:
                    cluster.append(j)
                    used[j] = True
            used[i] = True

            if len(cluster) == 1:
                merged.append(polys[cluster[0]][1])
            else:
                shapes = [polys[k][0] for k in cluster]
                best_conf = max(polys[k][1]["score"] for k in cluster)
                union = unary_union(shapes)

                if union.geom_type == "MultiPolygon":
                    union = max(union.geoms, key=lambda g: g.area)

                coords = list(union.exterior.coords)[:-1]
                merged.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "value": {
                        "polygonlabels": [label],
                        "points": [[x, y] for x, y in coords],
                    },
                    "score": best_conf,
                })

    return merged


def extract_points(poly):
    """Extract Label Studio points from a shapely polygon."""
    if poly is None or poly.is_empty or poly.geom_type != "Polygon":
        return []
    coords = list(poly.exterior.coords)[:-1]
    return [[x, y] for x, y in coords]


class YOLOSegBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo = YOLO(MODEL_PATH)
        self.labels = self.yolo.names

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            image_path = get_local_path(task["data"]["image"], task_id=task["id"])
            preds = self.yolo(
                image_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
            )[0]

            predictions = []
            img_h, img_w = preds.orig_shape

            if preds.masks is not None:
                for mask, box in zip(preds.masks, preds.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    polygon = mask.xyn[0]

                    if len(polygon) < 3:
                        continue

                    points = [[float(x) * 100, float(y) * 100] for x, y in polygon]

                    # Build shapely polygon
                    poly = safe_poly(points)
                    if poly is None:
                        continue

                    # Remove thin necks and small dangling extensions
                    poly = remove_thin_extensions(poly)
                    if poly is None or poly.is_empty:
                        continue

                    # Simplify noisy vertices
                    poly = poly.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
                    if poly.is_empty or poly.geom_type != "Polygon":
                        continue

                    # Filter tiny detections
                    if poly.area < MIN_AREA * 100:
                        continue

                    points = extract_points(poly)
                    if len(points) < 3:
                        continue

                    label_name = self.labels[cls]
                    label_name = LABEL_MAP.get(label_name, label_name)

                    predictions.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "value": {
                            "polygonlabels": [label_name],
                            "points": points,
                        },
                        "score": conf,
                    })

            # Merge overlapping same-class polygons
            predictions = merge_overlapping(predictions)

            results.append({
                "result": predictions,
                "score": sum(p["score"] for p in predictions) / len(predictions) if predictions else 0,
            })

        return results