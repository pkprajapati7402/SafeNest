"""Colab/T4 training entry point for a consented SafeNest hazard dataset.

Run this file in Google Colab after uploading or mounting a dataset. It is
deliberately not executed by the local application and does not download data.

Expected dataset layout:
  dataset/
    data.yaml
    images/train, images/val, images/test
    labels/train, labels/val, labels/test

The data.yaml names must describe only classes that are actually annotated.
Split by physical scene/household, not by adjacent video frames.
"""

from pathlib import Path
import os

from ultralytics import YOLO


# Set these in Colab or replace them with paths in your mounted Google Drive.
DATASET_YAML = os.environ.get("SAFENEST_DATASET_YAML", "/content/dataset/data.yaml")
OUTPUT_DIR = os.environ.get("SAFENEST_OUTPUT_DIR", "/content/safenest-runs")
BASE_MODEL = os.environ.get("SAFENEST_BASE_MODEL", "yolov8n.pt")


def main():
    dataset_yaml = Path(DATASET_YAML)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}. "
            "Upload a YOLO-format dataset or mount Google Drive first."
        )

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(dataset_yaml),
        epochs=80,
        imgsz=640,
        batch=-1,  # let Ultralytics estimate a safe batch for the T4
        device=0,
        workers=2,
        seed=42,
        patience=20,
        project=OUTPUT_DIR,
        name="safenest-hazard-yolov8n",
        pretrained=True,
        cos_lr=True,
        plots=True,
        verbose=True,
    )

    best_path = Path(OUTPUT_DIR) / "safenest-hazard-yolov8n" / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Training completed without best weights: {best_path}")

    trained = YOLO(str(best_path))
    metrics = trained.val(
        data=str(dataset_yaml),
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        split="test",
    )
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Best weights:", best_path)

    export_path = trained.export(format="onnx", imgsz=640, simplify=True)
    print("ONNX export:", export_path)
    print(
        "Copy best.pt to the SafeNest host and set "
        "SAFE_NEST_MODEL_PATH=best.pt before starting the server."
    )


if __name__ == "__main__":
    main()
