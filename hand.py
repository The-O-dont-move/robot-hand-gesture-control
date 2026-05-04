import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from serial_servo import LSCSeriesServo

FINGER_CLOSE = 900
FINGER_OPEN  = 2200
PALM_CLOSE   = 500
PALM_OPEN    = 2500

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 180.0
    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

class MovingAverage:
    def __init__(self, size=3):
        self.buf = deque(maxlen=size)
    def update(self, val):
        self.buf.append(val)
        return sum(self.buf) / len(self.buf)

angle_filters = [MovingAverage(3) for _ in range(5)]
wrist_angle_filter = MovingAverage(5)
wrist_smooth_angle = None

def compute_finger_angles(landmarks):
    def get(idx): return np.array(landmarks[idx])
    angles = []
    # 拇指
    v1 = get(3) - get(2)
    v2 = get(4) - get(3)
    angles.append(angle_between(v1, v2))
    # 其他四指
    for (mcp, pip, tip) in [(5,6,8), (9,10,12), (13,14,16), (17,18,20)]:
        v1 = get(pip) - get(mcp)
        v2 = get(tip) - get(pip)
        angles.append(angle_between(v1, v2))
    return angles

def classify_fingers(landmarks):
    fingers = []
    is_right = landmarks[5].x < landmarks[0].x
    if is_right:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    tips  = [8, 12, 16, 20]
    roots = [6, 10, 14, 18]
    for t, r in zip(tips, roots):
        fingers.append(1 if landmarks[t].y < landmarks[r].y else 0)
    return fingers

def get_wrist_pulse(landmarks):
    global wrist_smooth_angle
    p5  = np.array([landmarks[5].x, landmarks[5].y])
    p17 = np.array([landmarks[17].x, landmarks[17].y])
    vec = p17 - p5
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    angle = (angle + 360) % 360
    angle = wrist_angle_filter.update(angle)
    if wrist_smooth_angle is None:
        wrist_smooth_angle = angle
    else:
        diff = angle - wrist_smooth_angle
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        wrist_smooth_angle += diff * 0.3
        wrist_smooth_angle %= 360
    mapped = (wrist_smooth_angle - 90) / 180.0 * (PALM_OPEN - PALM_CLOSE) + PALM_CLOSE
    return int(np.clip(mapped, PALM_CLOSE, PALM_OPEN))

def draw_rainbow_hand(frame, landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm[0]*w), int(lm[1]*h)) for lm in landmarks]
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for i,j in conns:
        x1,y1 = pts[i]; x2,y2 = pts[j]
        c = (int(128+127*np.sin(x1*0.01+time.time()*2)),
             int(128+127*np.sin(y1*0.01+time.time()*2)),
             int(128+127*np.sin((x1+y1)*0.01)))
        cv2.line(frame,(x1,y1),(x2,y2),c,2)
    for x,y in pts:
        cv2.circle(frame,(x,y),4,(0,255,255),-1)

#
model_path = './hand_landmarker.task'
base = mp_python.BaseOptions(model_asset_path=model_path)
opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1, running_mode=vision.RunningMode.IMAGE,
                                    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                                    min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(opts)

SERIAL_PORT = 'COM5'
servo = LSCSeriesServo(SERIAL_PORT, 9600)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 控制变量
previous_finger_state = None
last_send_time = 0
last_finger_pulses = [FINGER_CLOSE] * 5
last_wrist_pulse = PALM_CLOSE
SEND_INTERVAL = 0.15
PULSE_CHANGE_THRESH = 30

print("渐变跟随模式启动！手指弯曲时平滑跟随，按 'q' 退出, '1'全握, '2'全张")

while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try: result = detector.detect(mp_img)
    except: continue

    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        kps = [(lm.x, lm.y, lm.z) for lm in hand]
        draw_rainbow_hand(frame, kps)

        finger_state = classify_fingers(hand)
        raw_angles = compute_finger_angles(kps)
        smoothed_angles = [angle_filters[i].update(raw_angles[i]) for i in range(5)]
        wrist_pulse = get_wrist_pulse(hand)

        finger_pulses = []
        for i in range(5):
            if finger_state[i] == 1:
                pulse = FINGER_OPEN
            else:
                ang = np.clip(smoothed_angles[i], 90, 180)
                ratio = (180 - ang) / 90.0
                pulse = int(FINGER_OPEN - ratio * (FINGER_OPEN - FINGER_CLOSE))
            finger_pulses.append(pulse)


        status_color = (0,255,0) if servo.healthy() else (0,0,255)
        cv2.putText(frame, f"Serial: {'OK' if servo.healthy() else 'LOST'}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        state_str = f"T:{finger_state[0]} I:{finger_state[1]} M:{finger_state[2]} R:{finger_state[3]} P:{finger_state[4]}"
        cv2.putText(frame, state_str, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        angle_str = " ".join([f"{smoothed_angles[i]:.0f}" for i in range(5)])
        cv2.putText(frame, f"Deg: {angle_str}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        cv2.putText(frame, f"Wrist: {wrist_pulse}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        now = time.time()


        state_changed = (finger_state != previous_finger_state)

        finger_changed = any(abs(finger_pulses[i] - last_finger_pulses[i]) > PULSE_CHANGE_THRESH for i in range(5))
        wrist_changed = abs(wrist_pulse - last_wrist_pulse) > 30
        time_ok = (now - last_send_time) > SEND_INTERVAL

        if state_changed or (time_ok and (finger_changed or wrist_changed)):
            servos = [(i+1, finger_pulses[i]) for i in range(5)] + [(6, wrist_pulse)]
            print(f"发送 -> 手指:{finger_pulses}, 手腕:{wrist_pulse}")
            servo.move_servos(servos, time_ms=200)
            last_finger_pulses = finger_pulses.copy()
            last_wrist_pulse = wrist_pulse
            previous_finger_state = finger_state
            last_send_time = now

    else:
        cv2.putText(frame, "No hand", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'):
        servo.move_servos([(i, FINGER_CLOSE) for i in range(1,6)] + [(6, PALM_CLOSE)], time_ms=500)
        previous_finger_state = None
        last_finger_pulses = [FINGER_CLOSE]*5
    elif key == ord('2'):
        servo.move_servos([(i, FINGER_OPEN) for i in range(1,6)] + [(6, PALM_CLOSE)], time_ms=500)
        previous_finger_state = None
        last_finger_pulses = [FINGER_OPEN]*5

    cv2.imshow('Hand Control', frame)

cap.release()
cv2.destroyAllWindows()
servo.close()
detector.close()