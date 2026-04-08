import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pandas as pd
import os


GESTURES            = ["open_palm", "peace", "thumbs_up", "fist", "point"]
SAMPLES_PER_GESTURE = 150
DATA_FILE           = "data/gesture_data2.csv"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


header = ["label"]
for i in range(21):
    header += [f"x{i}", f"y{i}"]

BaseOptions          = mp_python.BaseOptions
HandLandmarker       = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode    = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = HandLandmarker.create_from_options(options)


os.makedirs("data", exist_ok=True)
pd.DataFrame(columns=header).to_csv(DATA_FILE, index=False)

cap = cv2.VideoCapture(0)
print("\n Starting collection\n")

# loop
for gesture in GESTURES:
    print(f" Get ready for: [{gesture.upper()}]")
    print(f" Press SPACE when ready. Collecting {SAMPLES_PER_GESTURE} samples.\n")

   
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Gesture: {gesture} | Press SPACE to start",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): break

   
    sample_count = 0
    while sample_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            # Draw 21 landmark dots 
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            #  data row
            row = [gesture]
            for lm in hand_landmarks:
                row.append(round(lm.x, 5))
                row.append(round(lm.y, 5))

            pd.DataFrame([row], columns=header).to_csv(
                DATA_FILE, mode='a', header=False, index=False)

            sample_count += 1
            cv2.putText(frame, f"[{gesture}] {sample_count}/{SAMPLES_PER_GESTURE}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print(f"{sample_count} samples saved for [{gesture}]\n")


cap.release()
cv2.destroyAllWindows()
detector.close()


df = pd.read_csv(DATA_FILE)
print(f"\n All done! Data saved to: {DATA_FILE}")
print(f" Total rows: {len(df)}")
print(f"\n Samples per gesture:\n{df['label'].value_counts()}")
print(f"\n First 3 rows:\n{df.head(3)}")