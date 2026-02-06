from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from label_studio_tools.core.utils.io import get_local_path

MODEL_PATH = r"C:\Users\Roka\Dev\GitProjects\MyProjects\VipResearch-Zhou\Segmenter\scripts\runs\garlic_seg\final\weights\best.pt"

LABEL_MAP = {
    "garlic_bulb": "Garlic Bulb",
    "garlic_root": "Garlic Root",
}

class YOLOSegBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo = YOLO(MODEL_PATH)
        self.labels = self.yolo.names

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            image_path = get_local_path(task["data"]["image"], task_id=task["id"])
            preds = self.yolo(image_path)[0]

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

            results.append({
                "result": predictions,
                "score": sum(p["score"] for p in predictions) / len(predictions) if predictions else 0,
            })

        return results