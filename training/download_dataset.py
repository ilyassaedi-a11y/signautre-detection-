"""Download a labeled dataset from Roboflow in YOLOv11 format."""
import argparse
import os

from roboflow import Roboflow


def main():
    p = argparse.ArgumentParser(description="Download Roboflow dataset")
    p.add_argument("--workspace", required=True, help="Roboflow workspace id")
    p.add_argument("--project", required=True, help="Roboflow project id")
    p.add_argument("--version", type=int, required=True, help="Dataset version number")
    p.add_argument("--out", default="dataset", help="Download folder")
    args = p.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set your key first: export ROBOFLOW_API_KEY=your_key")

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    dataset = version.download("yolov11", location=args.out)
    print("Dataset downloaded to:", dataset.location)
    print("Use this data.yaml for training:", os.path.join(dataset.location, "data.yaml"))


if __name__ == "__main__":
    main()
