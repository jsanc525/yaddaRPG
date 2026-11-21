import time

import cv2
import numpy as np

from enum import Enum

stream_url = 'https://192.168.0.29:4444/video/mjpeg'
output_dir = 'captured-rolls'

diff_threshold = 25
motion_pixel_ratio = 0.015
settle_seconds = 1.0
cooldown_seconds = 1.0

roi = (0.15, 0.15, 0.85, 0.85)

class State(Enum):
    idle = 1
    rolling = 2
    settling = 3

def has_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
    delta = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(delta, diff_threshold, 255, cv2.THRESH_BINARY)
    changed_ratio = np.count_nonzero(thresh) / thresh.size
    return changed_ratio > motion_pixel_ratio

def extract_roi_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = int(roi[0] * w), int(roi[1] * h)
    x2, y2 = int(roi[2] * w), int(roi[3] * h)
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7,7), 0)

def main():
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "tls_verify;0"

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream: {stream_url}")

    state = State.idle
    prev_gray = None
    quiet_since = None
    last_trigger = 0.0
    roll_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            curr_gray = extract_roi_gray(frame)
            if prev_gray is None:
                prev_gray = curr_gray
                continue

            moving = has_motion(prev_gray, curr_gray)
            now = time.monotonic()

            if state == State.idle:
                if moving and (now - last_trigger) >= cooldown_seconds:
                    state = State.rolling
                    quiet_since = None
            elif state == State.rolling:
                if not moving:
                    state = State.settling
                    quiet_since = now
            elif state == state.settling:
                if moving:
                    state = state.rolling
                    quiet_since = None
                elif (
                    quiet_since is not None
                    and (now - quiet_since) >= settle_seconds
                ):
                    roll_count += 1
                    filename = (
                        f"{output_dir}/roll_{roll_count:04d}_"
                        f"{int(time.time())}.jpg"
                    )
                    cv2.imwrite(filename, frame)
                    print(f"[trigger] Saved {filename}")
                    last_trigger = now
                    state = State.idle
                    quiet_since = None

            prev_gray = curr_gray
    finally:
        cap.release()

if __name__ == "__main__":
    main()
