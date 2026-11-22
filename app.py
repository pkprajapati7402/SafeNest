"""
SafeNest AI - Phase 1 Web Dashboard (Flask backend)

Runs the YOLO model server-side. The browser captures webcam frames via
getUserMedia and posts them here; this returns detections + a risk score
as JSON, and the browser draws the overlay.

Run:
    python app.py
Then open:
    http://localhost:5000
"""

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from twilio.rest import Client


def load_env(env_path=None):
    """Load key-value pairs from .env file into os.environ."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^(?:(?:export|\$env:)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?(.*?)["\']?$', line)
            if m:
                key, val = m.group(1), m.group(2)
                os.environ[key] = val


load_env()


from config import (
    PERSON_CLASSES,
    DANGEROUS_CLASSES,
    NEUTRAL_HIGHLIGHT_CLASSES,
    CONF_THRESHOLD,
    danger_weight,
    box_center,
    simple_risk_score,
    risk_label,
)

app = Flask(__name__, template_folder=".", static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
ALERTS_DIR = os.path.join(app.root_path, "saved_alerts")
CRITICAL_THRESHOLD = 70
MODEL_PATH = os.environ.get("SAFE_NEST_MODEL_PATH", "yolov8n.pt")
MODEL_VERSION = os.path.basename(MODEL_PATH)
API_TOKEN = os.environ.get("SAFE_NEST_API_TOKEN", "").strip()

print("Loading YOLO model (first run downloads weights automatically)...")
model = YOLO(MODEL_PATH)
print("Model loaded.")


def decode_image(data_url: str):
    """Turn a 'data:image/jpeg;base64,...' string into a BGR numpy array."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ValueError("image must be a data:image URL")
    try:
        _, b64data = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("image is not valid base64") from error
    if len(img_bytes) > 1_500_000:
        raise ValueError("image is too large")
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def require_api_token():
    """Protect state-changing and inference endpoints in configured installs."""
    if not API_TOKEN:
        return None
    supplied = request.headers.get("Authorization", "")
    expected = f"Bearer {API_TOKEN}"
    if not hmac.compare_digest(supplied, expected):
        return jsonify({"error": "installation authentication required"}), 401
    return None


def sign_alert(risk):
    """Create a short-lived server-issued proof for an alert action."""
    payload = {
        "score": round(float(risk["score"]), 1),
        "label": risk["label"],
        "object": risk.get("object"),
        "issued_at": int(time.time()),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    signature = hmac.new(
        (API_TOKEN or "local-development-secret").encode(),
        encoded.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded}.{signature}"


def verify_alert_token(token):
    """Verify a server-issued alert proof and reject stale or altered data."""
    if not isinstance(token, str) or "." not in token:
        raise ValueError("server-issued alert token is required")
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(
        (API_TOKEN or "local-development-secret").encode(),
        encoded.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid alert token")
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, UnicodeDecodeError, binascii.Error) as error:
        raise ValueError("invalid alert token") from error
    if int(time.time()) - int(payload.get("issued_at", 0)) > 120:
        raise ValueError("alert token expired")
    if payload.get("label") != "CRITICAL" or float(payload.get("score", 0)) < CRITICAL_THRESHOLD:
        raise ValueError("alert token is not critical")
    return payload


def save_alert(data_url: str, risk: dict):
    """Save the alert image and metadata locally for later review."""
    _, b64data = data_url.split(",", 1)
    image_bytes = base64.b64decode(b64data, validate=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.makedirs(ALERTS_DIR, exist_ok=True)
    image_name = f"alert_{timestamp}.jpg"
    metadata_name = f"alert_{timestamp}.json"
    with open(os.path.join(ALERTS_DIR, image_name), "wb") as image_file:
        image_file.write(image_bytes)
    with open(os.path.join(ALERTS_DIR, metadata_name), "w", encoding="utf-8") as metadata_file:
        json.dump({"saved_at": timestamp, "risk": risk, "image": image_name}, metadata_file, indent=2)
    return image_name


def call_family_member(risk: dict):
    """Place an emergency call using backend-only Twilio configuration or local dev simulation."""
    dev_test_mode = os.environ.get("DEV_TEST_MODE", "false").strip().lower() in ("true", "1", "yes")
    object_name = risk.get("object") or "a dangerous item"

    if dev_test_mode:
        sim_sid = f"SIM_CALL_{int(time.time())}"
        app.logger.info(f"[DEV_TEST_MODE] Simulated emergency call initiated for {object_name} (SID: {sim_sid})")
        return {"sid": sim_sid, "simulated": True, "object": object_name}

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER")
    family_number = os.environ.get("FAMILY_PHONE_NUMBER")
    twiml_url = os.environ.get("TWILIO_TWIML_URL") or "http://demo.twilio.com/docs/voice.xml"

    if not all((account_sid, auth_token, from_number, family_number)):
        missing = []
        if not account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not from_number:
            missing.append("TWILIO_FROM_NUMBER")
        if not family_number:
            missing.append("FAMILY_PHONE_NUMBER")
        raise RuntimeError(f"Missing Twilio configuration in .env: {', '.join(missing)}")
    if float(risk.get("score", 0)) < CRITICAL_THRESHOLD:
        raise ValueError("emergency call requires a critical risk score")

    client = Client(account_sid, auth_token)
    # Using 'url' parameter to ensure 100% compatibility with Twilio Trial accounts
    call = client.calls.create(
        to=family_number,
        from_=from_number,
        url=twiml_url,
    )
    return {"sid": call.sid, "simulated": False}


@app.route("/static/<path:filename>", endpoint="static")
def static_files(filename):
    return send_from_directory(app.root_path, filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": MODEL_VERSION})


@app.route("/readyz")
def readyz():
    return jsonify({"ready": model is not None, "model": MODEL_VERSION}), 200 if model is not None else 503


@app.route("/detect", methods=["POST"])
def detect():
    auth_error = require_api_token()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "no image provided"}), 400

    try:
        frame = decode_image(data_url)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    t0 = time.time()
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    infer_ms = round((time.time() - t0) * 1000, 1)

    h, w = frame.shape[:2]
    detections = []
    persons, dangers = [], []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        center = box_center((x1, y1, x2, y2))

        if class_name in PERSON_CLASSES:
            category = "person"
            persons.append(center)
        elif class_name in DANGEROUS_CLASSES:
            category = "danger"
            dangers.append((class_name, center, danger_weight(class_name)))
        elif class_name in NEUTRAL_HIGHLIGHT_CLASSES:
            category = "neutral"
        else:
            continue

        detections.append({
            "class": class_name,
            "conf": round(conf, 3),
            "category": category,
            # normalized 0-1 coords so the frontend can scale to any canvas size
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
        })

    best_score, best_label, best_object = 0.0, "SAFE", None

    if persons and dangers:
        # Both person and dangerous hazard in frame -> proximity-weighted score
        for person_center in persons:
            for class_name, danger_center, weight in dangers:
                normalized_person = (person_center[0] / w, person_center[1] / h)
                normalized_danger = (danger_center[0] / w, danger_center[1] / h)
                dist = ((normalized_person[0] - normalized_danger[0]) ** 2
                        + (normalized_person[1] - normalized_danger[1]) ** 2) ** 0.5
                score = simple_risk_score(weight, dist)
                if score > best_score:
                    best_score, best_object = score, class_name
                    best_label = risk_label(score)
    elif dangers:
        # Dangerous hazard present in monitored area
        for class_name, danger_center, weight in dangers:
            score = round(weight * 0.8, 1)
            if score > best_score:
                best_score, best_object = score, class_name
                best_label = risk_label(score)

    risk = {
        "score": best_score,
        "label": best_label,
        "object": best_object,
        "has_person": len(persons) > 0,
    }
    response = {
        "detections": detections,
        "risk": risk,
        "infer_ms": infer_ms,
        "model": MODEL_VERSION,
    }
    if best_label == "CRITICAL":
        response["risk"]["alert_token"] = sign_alert(risk)
    return jsonify(response)


@app.route("/save-alert", methods=["POST"])
def save_alert_route():
    auth_error = require_api_token()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    risk = payload.get("risk") or {}
    try:
        verify_alert_token(risk.get("alert_token"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 403
    if not data_url or not data_url.startswith("data:image/"):
        return jsonify({"error": "valid alert image is required"}), 400
    try:
        image_name = save_alert(data_url, risk)
    except (ValueError, TypeError, base64.binascii.Error, OSError) as error:
        return jsonify({"error": f"could not save alert: {error}"}), 400
    return jsonify({"saved": True, "image": image_name})


@app.route("/call-family", methods=["POST"])
def call_family_route():
    auth_error = require_api_token()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    try:
        risk = payload.get("risk") or {}
        if payload.get("manual") is True:
            verified_risk = {
                "score": float(CRITICAL_THRESHOLD),
                "label": "CRITICAL",
                "object": "manual emergency alert",
            }
        else:
            verified_risk = verify_alert_token(risk.get("alert_token"))
        call_info = call_family_member(verified_risk)
    except (RuntimeError, ValueError) as error:
        return jsonify({"called": False, "error": str(error)}), 503
    except Exception as error:
        app.logger.exception("Twilio emergency call failed")
        return jsonify({"called": False, "error": str(error)}), 502
    return jsonify({
        "called": True,
        "call_sid": call_info.get("sid"),
        "simulated": call_info.get("simulated", False),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
