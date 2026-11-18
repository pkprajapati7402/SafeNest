# SafeNest AI Improvement Plan

## 1. Target project direction

The strongest version of SafeNest is not “YOLO detects an object and calls a
parent.” It is a **privacy-aware, event-driven child-safety perception system**
that:

1. observes a camera scene,
2. detects and tracks people and household hazards,
3. infers context over time (proximity, persistence, zones and activity),
4. explains why an incident was classified as risky,
5. escalates only after a debounced policy decision, and
6. records measurable evidence that the system is useful and safe.

Use the project title only if the implementation supports this broader claim.
Otherwise, use a narrower title such as “Real-Time Household Hazard Detection
and Caregiver Alerting Using YOLO.”

This is a safety aid, not a medical, emergency-response, or guaranteed
child-supervision system. The UI, README, demonstrations and evaluation must
say that clearly.

## 2. Current-state assessment

### What already works

- Flask service accepts webcam JPEG frames and runs an Ultralytics YOLO model.
- The browser renders normalized bounding boxes and a live risk banner.
- There is a basic hazard-weight plus person-proximity score.
- Critical events can save an image/JSON pair and initiate a Twilio call.
- Developer simulation mode, Docker packaging and a deployment manifest exist.
- The UI discloses that the current formula is a placeholder.

### Important limitations and defects to fix first

1. **The model/hazard taxonomy does not match.** `yolov8n.pt` is a COCO
   pretrained model. Most configured names (`fire`, `medicine`, `socket`,
   `chemical`, `stove`, `open flame`, etc.) are not COCO classes, so those rules
   cannot activate as intended. `bottle` is an unsafe proxy for medicine or
   cleaning chemicals and will create false alarms.
2. **There is no child detector.** A COCO `person` box cannot distinguish a
   child from an adult, and the system should not claim that it can.
3. **Risk is based on one frame.** A momentary false detection can become a
   critical event. The 25-second cooldown is browser state, not a server-side
   incident policy.
4. **Pixel distance is not physically meaningful.** It changes with resolution,
   camera position and perspective. Centroid-to-centroid distance also ignores
   whether the child can actually reach the object.
5. **The browser controls escalation.** `/call-family` trusts a client-provided
   risk object and `/save-alert` accepts arbitrary client-provided metadata.
   A caller can bypass the UI and repeatedly trigger calls.
6. **No authentication, authorization, rate limiting or request-size limit is
   present.** A public deployment would expose inference, disk writes and
   telephony endpoints.
7. **Alert storage is local disk only.** It has no retention policy, encryption,
   incident index, access control, cleanup job or durable-storage guarantee.
8. **Operational observability is minimal.** There is no health/readiness
   endpoint, structured incident ID, queue, model version in responses, or
   latency/error metrics.
9. **There are no focused tests or evaluation data.** The README describes
   accuracy and latency claims without a reproducible benchmark.
10. **Deployment is CPU-oriented.** The current Render/Docker setup should not
    be presented as a low-latency production GPU deployment; browser-to-cloud
    raw video also creates privacy and bandwidth concerns.
11. **The untracked `.env` must never be committed.** Keep secrets in the
    hosting secret manager, add `.env` to `.gitignore`, and rotate any
    credential that has ever been exposed.
12. **README drift exists.** Keep documentation synchronized with actual
    endpoints, model classes, deployment behavior and the roadmap.

## 3. Prioritized implementation roadmap

### Phase 0 — Make the existing prototype honest and safe (1–2 weeks)

- Add `.env` and runtime data directories to `.gitignore`; verify no secret is
  tracked.
- Add `GET /healthz` and `GET /readyz` with model-load status and model version.
- Add strict JSON/content-type checks, maximum body/image dimensions, base64
  validation, timeouts and useful error IDs.
- Add a server-side configuration flag that disables real calls by default.
- Require authentication for all state-changing endpoints. At minimum use a
  per-installation bearer token; for a multi-user product use caregiver
  accounts, hashed passwords/passkeys, sessions, CSRF protection and roles.
- Add rate limits and a circuit breaker for inference, snapshot writes and
  Twilio calls. Enforce cooldown and deduplication on the server.
- Never accept `score`, `label` or `object` as authoritative from the browser.
  The server must create an incident from its own inference result and policy
  decision; the client may request a manual alert through a separate endpoint.
- Replace the public TwiML demo URL with a project-owned, auditable voice
  message. Include incident ID, time, hazard, confidence and a callback
  instruction, without exposing a child image in the call.
- Add explicit consent, camera status, retention and “AI may be wrong” copy.
- Change the phone input so it is not misleading: the current value is only
  stored in local browser storage and does not configure the server call target.
- Add backend unit tests for decoding, malformed input, risk boundaries,
  missing Twilio configuration, manual-alert authorization and cooldown.

**Exit criteria:** no unauthenticated public call can place a phone call or
write arbitrary files; invalid/oversized frames are rejected; tests run in CI;
the README describes limitations accurately.

### Phase 1 — Build a real event/risk engine (2–4 weeks)

Create separate modules instead of keeping all behavior in `app.py`:

```text
safenest/
  api/                 # request validation and authentication
  vision/              # detector, tracker and model adapter
  risk/                # features, policy and state machine
  incidents/           # persistence, retention and notification jobs
  settings.py
tests/
```

- Introduce a versioned `Detection` schema containing class, confidence,
  bounding box, frame timestamp, model version and optional track ID.
- Use normalized geometry and camera calibration rather than fixed pixel
  thresholds. Prefer bottom-center/person-footpoint to hazard contact zones;
  document that this remains an approximation.
- Add temporal smoothing: confidence hysteresis, N-of-M frames, minimum dwell
  time, recovery time and a `SAFE -> OBSERVING -> WARNING -> CRITICAL ->
  RECOVERY` state machine.
- Track objects with ByteTrack or BoT-SORT so proximity and dwell time use
  stable identities rather than unrelated boxes between frames.
- Add configurable zones: crib/play area, kitchen, stairs, windows and
  restricted cabinets. A zone should be a polygon, not a hard-coded screen
  distance.
- Produce an explainable risk breakdown:
  `hazard_severity`, `child_or_person_presence`, `reachability`,
  `dwell_seconds`, `zone_rule`, `confidence`, `uncertainty`, and final action.
- Add “unknown/low-confidence” and “camera blocked/offline” states. A failed
  sensor must not appear safe.
- Emit one incident record per event with start/end time, evidence references,
  policy version, model version, notification status and caregiver
  acknowledgement.

**Exit criteria:** replaying the same recorded clip produces deterministic
incidents; a one-frame false positive does not call anyone; persistent risk
does not create duplicate incidents; every alert explains its evidence.

### Phase 2 — Train a domain-specific perception model (3–6 weeks)

Do not train on random internet images alone. Define a narrow v1 taxonomy and
collect consented, representative scenes:

- person/child/infant (only if the team can label and evaluate this responsibly)
- knife, scissors, lighter/open flame, hot pan/stove
- medicine blister/bottle, cleaning chemical bottle
- electrical outlet/cord, stairs, open window, swimming-pool/water edge
- choking-sized object as a **scene/object category**, not a claim that the
  model can infer choking
- optional context classes: crib, play mat, high chair, safety gate

Recommended workflow:

1. Write annotation definitions and “not enough evidence” rules first.
2. Capture varied lighting, distance, occlusion, camera heights, skin tones,
   household layouts and empty/background scenes.
3. Split by physical scene/household, not random near-duplicate frames.
4. Label with CVAT, Roboflow or Label Studio; preserve consent and provenance.
5. Train a small YOLO detector in Colab T4/Kaggle GPU or another student-pack
   GPU. Start with a baseline, then fine-tune; do not assume a larger model is
   better for real-time use.
6. Validate with precision, recall, F1, mAP50, mAP50-95, per-class confusion
   matrices, false alerts/hour, missed-event rate and end-to-end alert
   latency.
7. Maintain a hard-negative set (ordinary bottles, toys, reflections,
   posters, adult hands, empty rooms) and test it on every model version.
8. Export and benchmark the selected model (PyTorch, ONNX and optionally
   TensorRT) on the intended edge hardware.

Track runs, datasets, parameters and artifacts with MLflow; use DVC or an
equivalent data registry if the dataset becomes large. Keep model version and
dataset commit in every incident.

**Exit criteria:** a published evaluation report contains class-level metrics,
false-positive examples, missed-event examples, latency on target hardware and
known out-of-distribution cases. Never report only mAP.

### Phase 3 — Practical caregiver product features (3–5 weeks)

Implement only features that support a clear safety workflow:

- caregiver login and household/device enrollment
- multiple verified contacts with ordered escalation and acknowledgement
- push notification/SMS/voice fallback, with delivery status and retry policy
- “I’m aware / dismiss / false alarm” acknowledgement
- incident timeline with filters, evidence, explanation and export/delete controls
- daily/weekly summary: events, response time, false alarms and offline periods
- privacy zones and camera masking; process locally by default
- configurable quiet hours and notification policy, never silently disabling
  critical local warnings
- local buzzer/light and offline-safe local alert when the network is down
- camera health: FPS, inference latency, dropped frames, obstruction and
  last-seen timestamp
- a calibration screen for camera placement and zone drawing
- accessibility: readable contrast, keyboard use, screen-reader labels and
  caregiver-friendly language

Do not add face recognition, child identity recognition, emotion inference,
continuous cloud recording or “predictive behavior” unless there is a separate
necessity, consent, child-rights and security review. They add privacy risk
without being required for the core safety claim.

### Phase 4 — Audio and multimodal context (optional, after vision is measured)

Add audio only as an opt-in module with local processing and short-lived
buffers. Candidate events are baby crying, glass breaking, smoke-alarm sound
and a caregiver voice command. Fuse independent evidence conservatively:
audio should increase an “investigate” state, not independently call emergency
services. Evaluate noisy homes and clearly document that crying is not proof of
danger or distress.

## 4. Recommended technical architecture

```text
Camera/device
  -> local capture + privacy masking
  -> detector (GPU/CPU) + tracker
  -> feature extraction (zones, reachability, dwell, sensor health)
  -> server-side policy/state machine
  -> incident database + object storage (short retention)
  -> notification queue/workers
  -> caregiver dashboard and audit log
```

For a student demo, a modular Flask application with SQLite/PostgreSQL and a
background worker is sufficient. For an actual deployment, separate inference
from API/notification workers so a slow model cannot block health checks or
duplicate calls. WebSocket/SSE can stream state changes; do not stream raw
video to every dashboard by default.

Suggested persistence entities:

- `households`, `users`, `devices`, `contacts`
- `camera_zones`, `model_versions`, `policy_versions`
- `detections` (short retention or aggregates only)
- `incidents`, `incident_evidence`, `notifications`, `audit_events`

Use object storage with private, expiring URLs for evidence. Encrypt in transit
and at rest, apply least privilege, define deletion/retention jobs, and make
export and deletion testable.

## 5. Evaluation and demonstration plan

Create a reproducible `evaluation/` package with:

- 10–20 scripted scenarios: safe room, knife near adult, knife near child,
  bottle false positive, occlusion, low light, child leaves zone, camera
  blocked, network outage and repeated persistent hazard
- annotated clips/images separated into train/validation/test by scene
- a replay runner that produces incident JSON and metrics
- latency benchmark at CPU, Colab T4 and intended edge device
- load test for concurrent cameras and notification failures
- security tests for oversized payloads, spoofed risk, replayed calls,
  unauthorized evidence access and rate-limit bypass

Report these metrics on the project page and in the viva:

- detection precision/recall/F1 per hazard
- critical-event recall and missed-event rate
- false critical alerts per hour
- median and p95 end-to-end alert latency
- notification delivery success and acknowledgement time
- camera-offline detection time
- CPU/GPU memory and throughput per camera
- retention/deletion verification

A strong final demo is a recorded scenario replay plus a live camera, showing
the evidence timeline, explainable risk factors, state transitions, a
simulated notification, recovery, false-alarm acknowledgement and metrics.
Do not demonstrate real calls against unsuspecting recipients.

## 6. GPU and tooling plan

- **Colab T4/Kaggle GPU:** dataset experiments, ablations, training and
  validation. Save checkpoints and metrics outside the ephemeral runtime.
- **Student cloud credits:** use only for short-lived training or a private
  staging inference service; never upload identifiable child footage without
  consent and encryption.
- **Edge target:** benchmark Jetson Orin Nano, Raspberry Pi 5 plus accelerator,
  or a local laptop GPU. Choose the model from measured latency and recall,
  not from model size alone.
- **Experiment tracking:** MLflow for runs/model lineage; DVC or Git-LFS for
  approved dataset/model artifacts; GitHub Actions for tests and evaluation
  smoke checks.
- **Operations:** structured JSON logs, OpenTelemetry traces/metrics, health
  probes, alert delivery dashboards and error-budget style thresholds.

## 7. Privacy, security and child-rights requirements

Treat every frame containing a child as sensitive personal data:

- process on-device/local network by default and upload only incident evidence
  needed for the configured purpose;
- obtain informed, verifiable caregiver consent before collection;
- display what is captured, why, where it is stored and when it is deleted;
- minimize retention and provide deletion/export;
- encrypt transport and storage; keep Twilio/model credentials server-side;
- use private evidence access, audit logs, authentication and least privilege;
- document limitations, uncertainty, bias and out-of-distribution behavior;
- provide a local fallback and never represent “no detection” as “safe”;
- conduct a child-rights/privacy impact assessment before collecting a real
  child dataset or deploying outside a classroom demonstration.

For an India-based project, check the current Digital Personal Data Protection
requirements and applicable rules with an advisor; the plan should not claim
legal compliance merely because a consent checkbox exists.

## 8. Suggested final-year scope

If time is limited, implement these six deliverables rather than adding many
shallow features:

1. custom hazard dataset and reproducible training/evaluation pipeline;
2. tracking plus temporal risk state machine;
3. server-authoritative incident and notification workflow;
4. zones, caregiver acknowledgement and incident timeline;
5. authentication, privacy retention and security tests;
6. published metrics, ablation study and CPU/GPU/edge benchmark.

That scope demonstrates computer vision, ML evaluation, backend architecture,
distributed/async reliability, security, privacy-by-design and deployment
engineering—the qualities expected from a strong final-year B.Tech project.

## 9. Research references

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework):
  validity, reliability, safety, security, transparency, privacy and bias
  should be measured throughout the lifecycle.
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track): persistent
  object IDs and ByteTrack/BoT-SORT are appropriate foundations for temporal
  video events.
- [Ultralytics validation](https://docs.ultralytics.com/modes/val): use
  precision, recall, F1, mAP50 and mAP50-95, including per-image analysis.
- [Ultralytics COCO dataset](https://docs.ultralytics.com/datasets/detect/coco):
  COCO has 80 categories; missing domain classes require custom training.
- [UNICEF Guidance on AI and Children](https://www.unicef.org/innocenti/reports/policy-guidance-ai-children):
  child-rights-by-design, proportional data collection, transparency and
  safety testing.
- [OWASP Secure AI Model Ops Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secure_AI_Model_Ops_Cheat_Sheet.html):
  protect datasets/models, validate inputs, control access and make pipelines
  reproducible.
- [Flask web security guidance](https://flask.palletsprojects.com/en/stable/web-security/):
  request limits, secure cookies, CSRF, trusted hosts and deployment controls.
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/):
  model versions, lineage, metadata and promotion/rollback.
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/):
  traces and metrics for production observability.
- [India DPDP Act material](https://www.meity.gov.in/data-protection-framework):
  consult the current official framework and rules before processing child data.

