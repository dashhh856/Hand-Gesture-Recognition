import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import os
from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
CLF_PATH   = os.path.join(BASE_DIR, "modelf.pkl")  


classifier = joblib.load(CLF_PATH)


BaseOptions           = mp_python.BaseOptions
HandLandmarker        = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode     = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = HandLandmarker.create_from_options(options)


state = {
    "gesture": "—",
    "confidence": 0.0,
    "hand_detected": False,
    "lock": threading.Lock()
}

GESTURE_EMOJI = {
    "open_palm":  "🖐",
    "peace":      "✌",
    "thumbs_up":  "👍",
    "fist":       "✊",
    "point":      "☝",
}

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_img)

        gesture    = "—"
        confidence = 0.0
        detected   = False

        if results.hand_landmarks:
            detected = True
            lms = results.hand_landmarks[0]
            h, w, _ = frame.shape

            
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 5, (100, 200, 255), 2)

            
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]
            pts = [(int(lms[i].x*w), int(lms[i].y*h)) for i in range(21)]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (100, 200, 255), 2)

            
            row = []
            for lm in lms:
                row.append(round(lm.x, 5))
                row.append(round(lm.y, 5))

            X = np.array(row).reshape(1, -1)
            gesture    = classifier.predict(X)[0]
            proba      = classifier.predict_proba(X)[0]
            confidence = float(np.max(proba))

        with state["lock"]:
            state["gesture"]       = gesture
            state["confidence"]    = confidence
            state["hand_detected"] = detected

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()


@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture")
def gesture():
    with state["lock"]:
        g = state["gesture"]
        c = state["confidence"]
        d = state["hand_detected"]
    return jsonify({
        "gesture":      g,
        "emoji":        GESTURE_EMOJI.get(g, ""),
        "confidence":   round(c * 100, 1),
        "hand_detected": d
    })


if __name__ == "__main__":
    print("Starting gesture server at http://localhost:5000")
    print("Open index.html in your browser")
    app.run(debug=False, threaded=True)