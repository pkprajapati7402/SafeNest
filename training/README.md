# SafeNest GPU training

`train_colab.py` is a lightweight Google Colab/Kaggle T4 entry point. It
expects a YOLO-format dataset and does not download or contain child footage.

## Dataset requirements

Use public data and/or photos/videos recorded with informed consent. Define the
classes before labeling. A sensible first custom taxonomy is:

```yaml
names: [child, adult, medicine, chemical, flame, outlet, stairs, open_window, safety_gate]
```

Only keep classes for which there are enough varied examples. Include empty
rooms and hard negatives such as ordinary bottles, toys, reflections and
posters. Split by household/scene before extracting frames so near-duplicate
frames cannot leak between train and test.

The YAML should use the standard sibling directories:

```text
dataset/
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
```

In Colab:

```python
!pip install -U ultralytics mlflow
%run /content/train_colab.py
```

Copy the resulting `best.pt` to the SafeNest host and set
`SAFE_NEST_MODEL_PATH=best.pt`. Do not replace the baseline until the test
split report includes per-class precision, recall, F1, mAP50, mAP50-95,
false-alerts-per-hour and representative false positives/missed events.
