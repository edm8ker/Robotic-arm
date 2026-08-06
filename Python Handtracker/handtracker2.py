import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_opts
import serial
import time
import urllib.request
import os
import threading

# Download model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Download complete!")

# Change to your COM port
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

last_state    = None
current_state = 'O'
frame_lock    = threading.Lock()
latest_frame  = None
display_frame = None
running       = True

def is_fist(landmarks):
    finger_tips  = [8, 12, 16, 20]
    finger_bases = [6, 10, 14, 18]
    curled = 0
    for tip, base in zip(finger_tips, finger_bases):
        if landmarks[tip].y > landmarks[base].y:
            curled += 1
    return curled >= 3

# Setup landmarker
options    = vision.HandLandmarkerOptions(
    base_options=base_opts.BaseOptions(model_asset_path=model_path),
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

def detection_thread():
    global current_state, last_state, latest_frame, display_frame, running

    while running:
        # Wait until we have a frame
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect(mp_image)
        except Exception as e:
            print("Detection error:", e)
            continue

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            h, w, _ = frame.shape
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            if is_fist(landmarks):
                current_state = 'C'
                cv2.putText(frame, "FIST - CLOSING",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2)
            else:
                current_state = 'O'
                cv2.putText(frame, "OPEN - OPENING",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

            if current_state != last_state:
                try:
                    arduino.write(current_state.encode())
                    time.sleep(0.1)
                    last_state = current_state
                except:
                    print("Serial connection lost!")
                    running = False
        else:
            cv2.putText(frame, "No hand detected",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2)

        with frame_lock:
            display_frame = frame

# Open camera first and wait for it to be ready
cap = cv2.VideoCapture(0)
print("Waiting for camera...")
time.sleep(2)

# Grab a few frames to warm up camera
for i in range(5):
    cap.read()

print("Camera ready!")

# Now start detection thread
detector = threading.Thread(target=detection_thread, daemon=True)
detector.start()

print("Hand tracking started. Press Q to quit.")

while running:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    with frame_lock:
        latest_frame = frame

    # Show processed frame if available, otherwise raw frame
    with frame_lock:
        show = display_frame.copy() if display_frame is not None else frame

    cv2.imshow("Hand Tracking", show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()