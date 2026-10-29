"""Helpers: crop a signature, identify who signed it, and draw labels."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (92, 127, 27)    # BGR
AMBER = (59, 169, 242)   # BGR
DARK_TEXT = (51, 41, 31)
WHITE = (255, 255, 255)


def crop_box(img, bbox, pad=0.08):
    """Crop a bounding box (x1, y1, x2, y2) with a small padding. Returns None if empty."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def identify(people_model, crop, threshold=0.6, imgsz=224):
    """Ask the classifier whose signature this is.

    Returns (name, confidence, candidates). The name is "Unknown" when the
    best match is below `threshold`.
    """
    result = people_model.predict(crop, imgsz=imgsz, verbose=False)[0]
    probs = result.probs
    candidates = [
        {"name": result.names[int(i)].replace("_", " "), "confidence": round(float(c), 3)}
        for i, c in zip(probs.top5[:3], probs.top5conf[:3])
    ]
    best = candidates[0]
    name = best["name"] if best["confidence"] >= threshold else "Unknown"
    return name, best["confidence"], candidates


def draw_detections(img, detections):
    """Draw boxes with a label: person name if identified, otherwise 'signature'."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    scale = max(0.5, min(h, w) / 900)

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        person = d.get("person")
        if person is None:
            label = f'{d["class_name"]} {d["confidence"] * 100:.0f}%'
            color, text_color = GREEN, WHITE
        elif person == "Unknown":
            label = f'Unknown {d["person_confidence"] * 100:.0f}%'
            color, text_color = AMBER, DARK_TEXT
        else:
            label = f'{person} {d["person_confidence"] * 100:.0f}%'
            color, text_color = GREEN, WHITE

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(label, FONT, scale, max(1, thick - 1))
        top = y1 - th - base - 8
        if top < 0:
            top = y1
        cv2.rectangle(out, (x1, top), (x1 + tw + 10, top + th + base + 8), color, -1)
        cv2.putText(out, label, (x1 + 5, top + th + 3), FONT, scale, text_color,
                    max(1, thick - 1), cv2.LINE_AA)
    return out
