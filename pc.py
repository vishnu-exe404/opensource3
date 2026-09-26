import time
import math
import cv2
import mediapipe as mp
import requests

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"
REQUEST_COOLDOWN = 15.0          # min seconds between Gemma reminders (avoid nagging)

SLOUCH_DURATION_THRESHOLD = 8.0  # seconds of continuous slouching before alerting
NECK_ANGLE_THRESHOLD = 25        # degrees from vertical - above this = forward head tilt
TORSO_ANGLE_THRESHOLD = 15       # degrees from vertical - above this = leaning/hunching

_last_request_time = 0.0
_last_response = ""


# --- Gemma / Ollama ---
def ask_gemma(slouch_duration):
    """
    Sends only the slouch duration (no images) to Gemma for a short,
    friendly posture reminder. Cooldown-limited so it doesn't nag every frame.
    """
    global _last_request_time, _last_response

    now = time.time()
    if now - _last_request_time < REQUEST_COOLDOWN:
        return _last_response

    prompt = (
        f"A person has been slouching at their desk for about {slouch_duration:.0f} seconds. "
        "In one short, warm, friendly sentence, remind them to sit up straight "
        "and adjust their posture. Do not mention that you are an AI or reference "
        "any image or camera."
    )

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=8)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        _last_response = text if text else "(No response from Gemma.)"
    except requests.exceptions.ConnectionError:
        _last_response = "[Ollama not reachable - is 'ollama serve' running?]"
    except requests.exceptions.Timeout:
        _last_response = "[Ollama request timed out.]"
    except Exception as e:
        _last_response = f"[Gemma error: {e}]"

    _last_request_time = now
    return _last_response


# --- Angle Calculation ---
def angle_from_vertical(top_point, bottom_point, w, h):
    """
    Returns the angle (degrees) between the line top_point->bottom_point
    and a perfectly vertical line. 0 degrees = perfectly upright.
    Points are MediaPipe normalized landmarks; w/h convert to pixel space.
    """
    dx = (top_point.x - bottom_point.x) * w
    dy = (top_point.y - bottom_point.y) * h
    angle_rad = math.atan2(abs(dx), abs(dy))  # angle off the vertical axis
    return math.degrees(angle_rad)


def midpoint(p1, p2):
    class _Mid:
        pass
    m = _Mid()
    m.x = (p1.x + p2.x) / 2
    m.y = (p1.y + p2.y) / 2
    return m


def draw_overlay(frame, status, slouch_duration, neck_angle, torso_angle, gemma_text):
    color = (0, 0, 255) if status == "Slouching" else (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (330, 110), (245, 117, 16), -1)
    cv2.putText(frame, f"Posture: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color, 2)
    cv2.putText(frame, f"Neck angle: {neck_angle:.0f} deg", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Torso angle: {torso_angle:.0f} deg", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if status == "Slouching":
        cv2.putText(frame, f"Duration: {slouch_duration:.0f}s", (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if gemma_text:
        max_chars = 60
        lines = [gemma_text[i:i + max_chars] for i in range(0, len(gemma_text), max_chars)]
        for i, line in enumerate(lines[:2]):
            cv2.putText(frame, line, (10, frame.shape[0] - 40 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_posture_checker():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. Check the camera connection/permissions.")
        return

    slouch_start_time = None
    last_gemma_text = ""
    status = "Good Posture"

    print("Starting Posture Checker. Sit naturally in frame. Press 'q' or ESC to quit.")

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            neck_angle = 0.0
            torso_angle = 0.0
            slouch_duration = 0.0

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = result.pose_landmarks.landmark
                PL = mp_pose.PoseLandmark

                try:
                    left_shoulder = lm[PL.LEFT_SHOULDER.value]
                    right_shoulder = lm[PL.RIGHT_SHOULDER.value]
                    left_ear = lm[PL.LEFT_EAR.value]
                    right_ear = lm[PL.RIGHT_EAR.value]
                    left_hip = lm[PL.LEFT_HIP.value]
                    right_hip = lm[PL.RIGHT_HIP.value]

                    shoulder_mid = midpoint(left_shoulder, right_shoulder)
                    ear_mid = midpoint(left_ear, right_ear)
                    hip_mid = midpoint(left_hip, right_hip)

                    # Neck angle: ear midpoint relative to shoulder midpoint.
                    # Large angle = head pushed forward relative to shoulders.
                    neck_angle = angle_from_vertical(ear_mid, shoulder_mid, w, h)

                    # Torso angle: shoulder midpoint relative to hip midpoint.
                    # Large angle = leaning/hunching forward from the waist.
                    torso_angle = angle_from_vertical(shoulder_mid, hip_mid, w, h)

                except (IndexError, AttributeError):
                    neck_angle = 0.0
                    torso_angle = 0.0

                is_slouching = (neck_angle > NECK_ANGLE_THRESHOLD) or (torso_angle > TORSO_ANGLE_THRESHOLD)

                now = time.time()
                if is_slouching:
                    status = "Slouching"
                    if slouch_start_time is None:
                        slouch_start_time = now
                    slouch_duration = now - slouch_start_time

                    # --- AI Reminder (only duration sent, no images) ---
                    if slouch_duration >= SLOUCH_DURATION_THRESHOLD:
                        last_gemma_text = ask_gemma(slouch_duration)
                else:
                    status = "Good Posture"
                    slouch_start_time = None
                    slouch_duration = 0.0
                    last_gemma_text = ""  # clear reminder once posture is corrected

            else:
                cv2.putText(frame, "No person detected", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                slouch_start_time = None

            draw_overlay(frame, status, slouch_duration, neck_angle, torso_angle, last_gemma_text)

            cv2.imshow("Posture Checker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_posture_checker()
