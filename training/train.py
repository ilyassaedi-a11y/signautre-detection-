"""Train YOLO11 and copy the best weights to weights/best.pt."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Train YOLO11")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m/l/x.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="detector")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, patience=args.patience,
                project="runs", name=args.name, exist_ok=True, plots=True)

    metrics = model.val()
    print(f"\nmAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = Path(__file__).resolve().parent.parent / "weights"
    out.mkdir(exist_ok=True)
    shutil.copy(best, out / "best.pt")
    print("Saved:", out / "best.pt")


if __name__ == "__main__":
    main()
