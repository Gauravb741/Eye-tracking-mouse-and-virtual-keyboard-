import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
import os
import sys
from pynput.keyboard import Controller as KeyboardController, Key

current_mode = "m"
switch_flag = False

def mode_listener():
    global current_mode, switch_flag
    while True:
        mode = input("Enter 'm' for mouse mode or 'k' for keyboard mode: ").strip().lower()
        if mode in ['m', 'k'] and mode != current_mode:
            current_mode = mode
            switch_flag = True

def eye_tracking_mouse():
    global switch_flag
    print("[MODE] Eye Tracking Mouse & Gesture Control")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    screen_w, screen_h = pyautogui.size()
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
    smoothening = 3

    def get_eye_center(landmarks, eye_indices):
        x = [landmarks[i][0] for i in eye_indices]
        y = [landmarks[i][1] for i in eye_indices]
        return int(np.mean(x)), int(np.mean(y))

    def calibrate(cap, seconds=5):
        print("Calibrating eyes...")
        start_time = time.time()
        min_x, min_y = 9999, 9999
        max_x, max_y = -9999, -9999

        while time.time() - start_time < seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                le = get_eye_center(pts, LEFT_EYE_IDX)
                re = get_eye_center(pts, RIGHT_EYE_IDX)
                center = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2)
                min_x, max_x = min(min_x, center[0]), max(max_x, center[0])
                min_y, max_y = min(min_y, center[1]), max(max_y, center[1])
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.putText(frame, "Calibrating...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Calibrate", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow("Calibrate")
        return min_x, max_x, min_y, max_y

    def fingers_status(hand_landmarks):
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        s = []
        s.append(int(hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x))
        for tip, pip in zip(tips[1:], pips[1:]):
            s.append(int(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y))
        return s

    cap = cv2.VideoCapture(0)
    time.sleep(2)
    calibration = calibrate(cap)
    if not calibration:
        print("Calibration failed.")
        cap.release()
        return
    ex_min, ex_max, ey_min, ey_max = calibration
    prev_x = prev_y = 0
    last_click_time = 0
    cooldown = 1.0

    while cap.isOpened():
        if switch_flag:
            break

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = face_mesh.process(rgb)
        hand = hands.process(rgb)

        if face.multi_face_landmarks:
            lm = face.multi_face_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
            le = get_eye_center(pts, LEFT_EYE_IDX)
            re = get_eye_center(pts, RIGHT_EYE_IDX)
            cx = (le[0] + re[0]) // 2
            cy = (le[1] + re[1]) // 2

            screen_x = np.interp(cx, [ex_min, ex_max], [0, screen_w])
            screen_y = np.interp(cy, [ey_min, ey_max], [0, screen_h])
            screen_x = np.clip(screen_x, 10, screen_w - 10)
            screen_y = np.clip(screen_y, 10, screen_h - 10)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            cv2.circle(frame, le, 5, (255, 0, 0), -1)
            cv2.circle(frame, re, 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        if hand.multi_hand_landmarks:
            hlm = hand.multi_hand_landmarks[0]
            fingers = fingers_status(hlm)
            cv2.putText(frame, f"{fingers}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ct = time.time()

            if fingers == [0, 1, 0, 0, 0] and ct - last_click_time > cooldown:
                pyautogui.click()
                last_click_time = ct
            elif fingers == [0, 1, 1, 0, 0] and ct - last_click_time > cooldown:
                pyautogui.doubleClick()
                last_click_time = ct
            elif fingers == [1, 1, 1, 1, 1] and ct - last_click_time > cooldown:
                pyautogui.click(button='right')
                last_click_time = ct

        cv2.imshow("Eye Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def virtual_keyboard():
    global switch_flag
    print("[MODE] Virtual Keyboard (Pinch Gesture)")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_draw = mp.solutions.drawing_utils
    keyboard = KeyboardController()

    class Button:
        def __init__(self, x, y, w, h, text):
            self.x, self.y, self.w, self.h = x, y, w, h
            self.text = text
            self.pressed = False
            self.last_time = 0

        def draw(self, img):
            color = (200, 200, 200) if not self.pressed else (0, 255, 0)
            cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), color, -1)
            cv2.putText(img, self.text, (self.x+10, self.y+self.h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        def check(self, p1, p2, thres=30):
            cx, cy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            pinch = np.linalg.norm(np.array(p1)-np.array(p2))
            inside = self.x < cx < self.x+self.w and self.y < cy < self.y+self.h
            if inside and pinch < thres and time.time() - self.last_time > 0.6:
                self.last_time = time.time()
                self.pressed = True
                return True
            self.pressed = False
            return False

    keys = [
        list("QWERTYUIOP"),
        list("ASDFGHJKL"),
        ["Shift"] + list("ZXCVBNM") + ["Back"],
        ["Tab", "Space", "Enter"]
    ]
    buttons = []
    key_w, key_h, gap = 80, 80, 10
    screen_w = 1280
    start_y = 100
    for i, row in enumerate(keys):
        total_units = sum(5 if k == "Space" else 2 if k in ["Tab", "Enter", "Shift", "Back"] else 1 for k in row)
        start_x = (screen_w - (key_w + gap) * total_units + gap) // 2
        for k in row:
            w = key_w * (5 if k == "Space" else 2 if k in ["Tab", "Enter", "Shift", "Back"] else 1)
            buttons.append(Button(start_x, start_y + i*(key_h+gap), w, key_h, k))
            start_x += w + gap

    cap = cv2.VideoCapture(0)
    cap.set(3, screen_w)
    cap.set(4, 720)
    prev = 0

    while cap.isOpened():
        if switch_flag:
            break

        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            h, w, _ = img.shape
            lm = res.multi_hand_landmarks[0].landmark
            thumb = int(lm[mp_hands.HandLandmark.THUMB_TIP].x * w), int(lm[mp_hands.HandLandmark.THUMB_TIP].y * h)
            index = int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w), int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            cv2.circle(img, thumb, 8, (255, 0, 255), -1)
            cv2.circle(img, index, 8, (255, 0, 255), -1)
            cv2.line(img, thumb, index, (255, 0, 255), 2)

            for b in buttons:
                if b.check(index, thumb):
                    if b.text == "Back":
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                    elif b.text == "Space":
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                    elif b.text == "Enter":
                        keyboard.press(Key.enter)
                        keyboard.release(Key.enter)
                    elif b.text == "Tab":
                        keyboard.press(Key.tab)
                        keyboard.release(Key.tab)
                    elif b.text == "Shift":
                        keyboard.press(Key.shift)
                        keyboard.release(Key.shift)
                    else:
                        keyboard.press(b.text.lower())
                        keyboard.release(b.text.lower())
            mp_draw.draw_landmarks(img, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        for b in buttons:
            b.draw(img)

        fps = 1 / (time.time() - prev) if prev else 0
        prev = time.time()
        cv2.putText(img, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Virtual Keyboard", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    listener_thread = threading.Thread(target=mode_listener, daemon=True)
    listener_thread.start()

    while True:
        switch_flag = False
        if current_mode == "m":
            eye_tracking_mouse()
        elif current_mode == "k":
            virtual_keyboard()
