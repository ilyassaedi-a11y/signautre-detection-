"""Train the identity classifier (whose signature is this?) and save it to
weights/people_cls.pt."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Train YOLO11 classifier on signatures")
    p.add_argument("--data", required=True, help="Folder with train/ and val/ sub-folders")
    p.add_argument("--model", default="yolo11n-cls.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="people")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, project="runs", name=args.name, exist_ok=True,
        fliplr=0.0, flipud=0.0,  # a mirrored signature is not the same signature
        plots=True,
    )

    metrics = model.val()
    print(f"\nTop-1 accuracy: {metrics.top1:.3f}")
    print(f"Top-5 accuracy: {metrics.top5:.3f}")

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = Path(__file__).resolve().parent.parent / "weights"
    out.mkdir(exist_ok=True)
    shutil.copy(best, out / "people_cls.pt")
    print("Saved:", out / "people_cls.pt")


if __name__ == "__main__":
    main()
