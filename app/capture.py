import time

import cv2
import numpy as np

from enum import Enum
from inputimeout import inputimeout

stream_url = 'https://192.168.0.29:4444/video/mjpeg'
output_dir = 'captured-rolls'

diff_threshold = 25.0
motion_pixel_ratio = 0.015
min_contour_area = 120.0     # Minimum connected blob area (px^2) to count as motion
dilate_iterations = 1        # Bridges split fragments of a small die
log_interval_seconds = 0.5   # How often to print live motion stats
settle_seconds = 2.0
cooldown_seconds = 60.0

roi = (0.15, 0.15, 0.85, 0.85)

class State(Enum):
    idle = 0
    rolling = 1
    settling = 2
    resetting = 3

def analyze_motion(prev_gray: np.ndarray, curr_gray: np.ndarray):
    delta = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(delta, diff_threshold, 255, cv2.THRESH_BINARY)
    if dilate_iterations > 0:
        thresh = cv2.dilate(thresh, None, iterations=dilate_iterations)

    pixel_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    areas = [cv2.contourArea(c) for c in contours]
    qualified_areas = [a for a in areas if a >= min_contour_area]
    max_area = max(areas) if areas else 0.0
    moving = len(qualified_areas) > 0

    return moving, max_area, len(qualified_areas), pixel_ratio

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
    roll_count = 0
    last_log = 0.0

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

            moving, max_area, qualified_count, pixel_ratio = analyze_motion(
                prev_gray, curr_gray
            )
            now = time.monotonic()

            if state == State.idle:
                if moving:
                    state = State.rolling
                    quiet_since = None
            elif state == State.rolling:
                if not moving:
                    state = State.settling
                    quiet_since = now
            elif state == State.settling:
                if moving:
                    state = State.rolling
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
                    state = State.resetting
                    quiet_since = None
            elif state == State.resetting:
                reset_msg = "Capture reset"
                try:
                    prompt = (
                        "Clear the tray and press any key to continue. "
                        "Capture will automatically restart in "
                        f"{int(cooldown_seconds)} seconds."
                    )
                    _ = inputimeout(prompt=prompt, timeout=cooldown_seconds)
                    state = State.idle
                    prev_gray = None
                    print(reset_msg)

                except Exception:
                    state = State.idle
                    prev_gray = None
                    print(reset_msg)
            if (now - last_log) >= log_interval_seconds:
                quiet_elapsed = (
                    (now - quiet_since) if quiet_since is not None else 0.0
                )
                print(
                    f"[{state.name:8s}] moving={int(moving)} "
                    f"max_area={max_area:7.1f} blobs={qualified_count:2d} "
                    f"ratio={pixel_ratio:.4f} quiet={quiet_elapsed:.2f}s"
                )
                last_log = now
            prev_gray = curr_gray
    finally:
        cap.release()

if __name__ == "__main__":
    main()
