"""Configuration for the lightweight COCO baseline.

The checked-in ``yolov8n.pt`` baseline can emit only the COCO entries listed
under ``BASELINE_DANGEROUS_CLASSES``. The additional custom entries are
reserved for a future trained SafeNest model and are never inferred by the
baseline weights.
"""

# Classes we treat as the "child/person" side of the risk equation
PERSON_CLASSES = {"person", "child", "adult"}

# Classes present in the COCO label set used by yolov8n.pt.
BASELINE_DANGEROUS_CLASSES = {
    "knife": 95,
    "scissors": 90,
    "fork": 45,
    "hair drier": 55,
    "toaster": 60,
    "oven": 50,
    "microwave": 45,
}

# Classes for the custom SafeNest model. They do not become active until
# SAFE_NEST_MODEL_PATH points to weights trained with matching labels.
CUSTOM_DANGEROUS_CLASSES = {
    "medicine": 90,
    "chemical": 85,
    "flame": 95,
    "open_flame": 95,
    "outlet": 90,
    "electrical_socket": 90,
    "stairs": 70,
    "open_window": 80,
}

DANGEROUS_CLASSES = {**BASELINE_DANGEROUS_CLASSES, **CUSTOM_DANGEROUS_CLASSES}

# Everything else we still want to see on screen but don't score as hazard
NEUTRAL_HIGHLIGHT_CLASSES = {
    "bottle", "cell phone", "teddy bear", "chair", "couch", "bed",
    "dining table", "cup", "bowl", "book", "clock", "sink", "refrigerator",
    "safety_gate", "crib", "play_mat",
}

# Minimum detection confidence to accept a box at all
CONF_THRESHOLD = 0.35

# Normalized distances are resolution-independent approximations. A calibrated
# zone/polygon should replace this heuristic when a camera is installed.
DIST_HIGH_RISK = 0.28
DIST_WARNING = 0.55


def danger_weight(class_name: str) -> int:
    return DANGEROUS_CLASSES.get(class_name.lower(), 0)


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def pixel_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def simple_risk_score(danger, distance):
    """
    Risk score calculation based on hazard danger weight and proximity.
    Close or medium range to a child/person escalates to full danger level.
    """
    if distance <= DIST_HIGH_RISK:
        proximity_factor = 1.0
    elif distance <= DIST_WARNING:
        proximity_factor = 0.85
    else:
        proximity_factor = 0.75
    return round(danger * proximity_factor, 1)


def risk_label(score: float) -> str:
    if score >= 70:
        return "CRITICAL"
    if score >= 45:
        return "HIGH RISK"
    if score >= 20:
        return "WARNING"
    return "SAFE"
