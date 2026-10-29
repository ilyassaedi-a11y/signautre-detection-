"""Crop signatures with the detector and split them into train/val folders
for the identity classifier.

Input:   raw_signatures/<Person Name>/*.jpg        (a few photos per person)
Output:  signature_people/train/<Person Name>/*.jpg
         signature_people/val/<Person Name>/*.jpg
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from utils import crop_box  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGES = 4


def make_detector(weights, conf, imgsz):
    """Return a function: image -> best signature box (x1, y1, x2, y2) or None."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(img):
        r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        best = int(r.boxes.conf.argmax())
        return r.boxes.xyxy[best].tolist()

    return detect


def collect(person_dir, detect, pad):
    """Crop the signature from every image of one person."""
    crops, fallback = [], 0
    for f in sorted(person_dir.iterdir()):
        if f.suffix.lower() not in IMAGE_EXT:
            continue
        img = cv2.imread(str(f))
        if img is None:
            print(f"  skipped (cannot read): {f.name}")
            continue
        bbox = detect(img)
        crop = crop_box(img, bbox, pad) if bbox is not None else None
        if crop is None:  # detector found nothing: use the whole image
            crop, fallback = img, fallback + 1
        crops.append((f.stem, crop))
    return crops, fallback


def split(items, val_ratio, rng):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_val = max(2, round(n * val_ratio)) if n >= 8 else 1
    return items[n_val:], items[:n_val]


def build_dataset(raw, out, detect, val_ratio=0.25, pad=0.08, seed=42):
    raw, out = Path(raw), Path(out)
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(seed)
    summary = {}

    for person_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        crops, fallback = collect(person_dir, detect, pad)
        if len(crops) < MIN_IMAGES:
            print(f"- {person_dir.name}: only {len(crops)} usable images, need at least "
                  f"{MIN_IMAGES}. Skipped.")
            continue
        train, val = split(crops, val_ratio, rng)
        for subset, rows in (("train", train), ("val", val)):
            folder = out / subset / person_dir.name
            folder.mkdir(parents=True, exist_ok=True)
            for stem, crop in rows:
                cv2.imwrite(str(folder / f"{stem}.jpg"), crop)
        summary[person_dir.name] = (len(train), len(val), fallback)

    print(f"\n{'Person':30s} {'train':>6s} {'val':>5s}  no-box")
    for name, (t, v, fb) in summary.items():
        print(f"{name:30s} {t:6d} {v:5d}  {fb}")
    print(f"\nPeople in dataset: {len(summary)}   saved to: {out}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Prepare the signature identity dataset")
    p.add_argument("--raw", required=True, help="Folder with one sub-folder per person")
    p.add_argument("--detector", default="weights/best.pt", help="Signature detector weights")
    p.add_argument("--out", default="signature_people")
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--pad", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    detect = make_detector(args.detector, args.conf, args.imgsz)
    summary = build_dataset(args.raw, args.out, detect, args.val_ratio, args.pad, args.seed)
    if len(summary) < 2:
        raise SystemExit("Need at least 2 people with enough images to train a classifier.")


if __name__ == "__main__":
    main()
