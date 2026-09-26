# opensource3# AI Posture Checker

A real-time webcam application that monitors your sitting posture using pose detection, tracks how long you've been slouching, and delivers short, friendly reminders from a local Gemma model via Ollama — no images ever leave your machine.

## Features

- **Pose-Based Posture Tracking** — Uses MediaPipe Pose to measure neck angle (forward head tilt) and torso angle (leaning/hunching) relative to vertical.
- **Slouch Duration Timer** — Tracks continuous slouching time and only alerts after a configurable threshold, avoiding false alarms from brief movements.
- **Live Angle & Status Overlay** — On-screen display of current posture status ("Good Posture" / "Slouching"), neck angle, torso angle, and slouch duration.
- **Local AI Reminders** — Once slouching persists past the threshold, sends only the slouch duration — never images — to a local [Ollama](https://ollama.ai/) instance running `gemma2:2b`, and displays a short, warm reminder on-screen. Reminders are cooldown-limited so they don't nag every frame.
- **Auto-Clearing Feedback** — The reminder disappears as soon as good posture is restored.

## Requirements

- Python 3.8+
- A webcam
- [Ollama](https://ollama.ai/) installed and running locally with the `gemma2:2b` model pulled (optional — the app runs without it, just without AI reminders)

### Python dependencies

```bash
pip install opencv-python mediapipe requests
```

## Project Structure

```
.
├── main.py              # (this script)
```

No local data files are created — this app doesn't store any images, video, or history; everything runs in memory for the current session.

## Usage

Run the script:

```bash
python main.py
```

Sit naturally in view of the webcam. The live feed opens with:
- Pose skeleton overlay
- Posture status (green = good, red = slouching)
- Neck angle and torso angle in degrees
- Slouch duration once you start slouching
- Periodic friendly reminders to sit up straight

Press `q` or `ESC` to quit.

## Configuration

Key parameters can be adjusted near the top of the script:

| Variable | Description |
|---|---|
| `OLLAMA_URL` | Ollama API endpoint |
| `MODEL_NAME` | Ollama model used for reminders (default: `gemma2:2b`) |
| `REQUEST_COOLDOWN` | Minimum seconds between Gemma reminder requests (avoids nagging) |
| `SLOUCH_DURATION_THRESHOLD` | Seconds of continuous slouching before triggering an alert |
| `NECK_ANGLE_THRESHOLD` | Degrees off vertical (ear vs. shoulder midpoint) considered forward head tilt |
| `TORSO_ANGLE_THRESHOLD` | Degrees off vertical (shoulder vs. hip midpoint) considered leaning/hunching |

## How It Works

1. MediaPipe Pose detects shoulder, ear, and hip landmarks each frame.
2. The midpoints of each pair (left/right) are computed to reduce noise from head/body rotation.
3. Two angles are measured from vertical:
   - **Neck angle** — ear midpoint relative to shoulder midpoint (forward head posture)
   - **Torso angle** — shoulder midpoint relative to hip midpoint (leaning/hunching)
4. If either angle exceeds its threshold, the user is flagged as slouching and a timer starts.
5. Once the slouch duration passes `SLOUCH_DURATION_THRESHOLD`, a coaching reminder is requested from Gemma (cooldown-limited) and shown on-screen.

## Privacy Note

No video, images, or posture data are ever saved to disk or sent externally. Only the slouch duration (a number, in seconds) is sent to your **local** Ollama instance to generate reminder text.

## License

Add a license of your choice (e.g. MIT) here.
