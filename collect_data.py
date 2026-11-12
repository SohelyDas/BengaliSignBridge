import os
import sys
import time
from pathlib import Path

# Enable UTF-8 encoding for standard output and error on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labels import CLASSES, LABELS

# Ensure working directory is the script folder
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

SEQ_LEN = 30            # frames per sample
SAMPLES_PER_CLASS = 30  # default target samples per class per signer
SIGNER_ID = "s01"       # identifier for the person recording (e.g. s01, s02)
DATA_DIR = BASE_DIR / "data"

# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load Bengali font if available on Windows, with fallback
BENGALI_FONT = None
for font_path in [
    r"C:\Windows\Fonts\vrinda.ttf",
    r"C:\Windows\Fonts\vrindab.ttf",
    r"C:\Windows\Fonts\kalpurush.ttf",
    str(BASE_DIR / "fonts" / "kalpurush.ttf"),
]:
    if Path(font_path).exists():
        try:
            BENGALI_FONT = ImageFont.truetype(font_path, 28)
            break
        except Exception:
            pass


def overlay_text(
    frame: np.ndarray,
    text: str,
    pos: tuple,
    font_scale: float = 0.7,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    bengali_text: str = "",
) -> np.ndarray:
    """Draw text on an OpenCV frame with optional Bengali Unicode rendering."""
    x, y = pos
    if bengali_text and BENGALI_FONT is not None:
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            full_text = f"{text} ({bengali_text})" if text else bengali_text
            draw.text((x, y), full_text, font=BENGALI_FONT, fill=(color[2], color[1], color[0]))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    display_text = f"{text} [{bengali_text}]" if (text and bengali_text) else (text or bengali_text)
    cv2.putText(
        frame,
        display_text,
        (x, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def extract(results) -> np.ndarray:
    """Return a 126-length feature vector: [left hand 63 | right hand 63]."""
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            vec = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            label = hd.classification[0].label  # "Left" or "Right"
            if label == "Left":
                left = vec
            else:
                right = vec

    return np.concatenate([left, right])


def open_camera(max_attempts: int = 2) -> cv2.VideoCapture:
    """Attempt to open camera using CAP_DSHOW on Windows, with standard fallback."""
    for idx in range(max_attempts):
        # On Windows, DirectShow backend is fast and reliable
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()

        # Fallback to default backend
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()

    raise RuntimeError(
        "Could not open any webcam. Please ensure a camera is connected, "
        "not in use by another app, and that camera permissions are enabled."
    )


def get_next_sample_index(class_dir: Path, signer_id: str) -> int:
    """Find the next available sample index for this signer in the class directory."""
    existing_files = list(class_dir.glob(f"{signer_id}_*.npy"))
    if not existing_files:
        return 0

    indices = []
    for f in existing_files:
        try:
            idx_str = f.stem.split("_")[-1]
            indices.append(int(idx_str))
        except (ValueError, IndexError):
            continue

    return max(indices) + 1 if indices else len(existing_files)


def run_countdown(cap, hands, cls_name: str, bengali_label: str, window_name: str) -> bool:
    """Show a short 3-2-1 countdown with live hand tracking so user can position hands."""
    for count in [3, 2, 1]:
        start_time = time.time()
        while time.time() - start_time < 0.5:
            ok, frame = cap.read()
            if not ok or frame is None:
                return False
            frame = cv2.flip(frame, 1)

            # Process hands live
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Overlay countdown
            frame = overlay_text(
                frame,
                f"Sign: {cls_name}",
                (20, 20),
                font_scale=0.8,
                color=(255, 255, 0),
                bengali_text=bengali_label,
            )
            cv2.putText(
                frame,
                f"GET READY: {count}",
                (int(frame.shape[1] / 2) - 150, int(frame.shape[0] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                return False
    return True


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    window_name = "Bengali Sign Data Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"Starting data collection for signer: '{SIGNER_ID}'")
    print(f"Data directory: {DATA_DIR}")
    print(f"Total classes: {len(CLASSES)} -> {CLASSES}")

    cap = open_camera()

    try:
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            for cls_idx, cls in enumerate(CLASSES):
                class_dir = DATA_DIR / cls
                class_dir.mkdir(parents=True, exist_ok=True)
                bengali_label = LABELS.get(cls, "")

                start_idx = get_next_sample_index(class_dir, SIGNER_ID)
                total_for_signer = start_idx

                print(f"\n--- Class [{cls_idx + 1}/{len(CLASSES)}]: '{cls}' ({bengali_label}) ---")
                print(f"Already collected for {SIGNER_ID}: {total_for_signer} samples.")

                sample_count = 0
                while sample_count < SAMPLES_PER_CLASS:
                    current_sample_num = start_idx + sample_count

                    # ---------- Preview & Wait Loop ----------
                    skip_class = False
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            print("Warning: Failed to capture camera frame.")
                            continue

                        frame = cv2.flip(frame, 1)

                        # Run hand detection so the user sees live feedback
                        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        hand_count = 0
                        hand_types = []
                        if results.multi_hand_landmarks:
                            hand_count = len(results.multi_hand_landmarks)
                            for hand_lms, hd in zip(
                                results.multi_hand_landmarks, results.multi_handedness or []
                            ):
                                mp_draw.draw_landmarks(
                                    frame,
                                    hand_lms,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style(),
                                )
                                label = hd.classification[0].label
                                hand_types.append(label)

                        # UI Header
                        frame = overlay_text(
                            frame,
                            f"Class: {cls} ({cls_idx + 1}/{len(CLASSES)})",
                            (20, 15),
                            font_scale=0.75,
                            color=(0, 255, 0),
                            bengali_text=bengali_label,
                        )

                        sample_info = f"Sample: {sample_count + 1}/{SAMPLES_PER_CLASS} (Signer: {SIGNER_ID})"
                        cv2.putText(
                            frame, sample_info, (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

                        hand_color = (0, 255, 0) if hand_count > 0 else (0, 165, 255)
                        hand_str = f"Hands: {hand_count} detected ({', '.join(hand_types) if hand_types else 'None'})"
                        cv2.putText(
                            frame, hand_str, (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 2
                        )

                        cv2.putText(
                            frame,
                            "[SPACE] Record  |  [N] Next Class  |  [Q] Quit",
                            (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(10) & 0xFF

                        # Check if user clicked window close button
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("Window closed by user.")
                            return

                        if key == ord(" "):
                            break  # Start countdown and recording
                        if key == ord("n"):
                            print(f"Skipping remainder of class '{cls}'.")
                            skip_class = True
                            break
                        if key == ord("q"):
                            print("Quitting data collection.")
                            return

                    if skip_class:
                        break

                    # ---------- Countdown ----------
                    countdown_ok = run_countdown(cap, hands, cls, bengali_label, window_name)
                    if not countdown_ok:
                        return

                    # ---------- Recording Sequence ----------
                    sequence = []
                    for frame_idx in range(SEQ_LEN):
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            print("Warning: Camera dropped a frame during recording.")
                            continue

                        frame = cv2.flip(frame, 1)
                        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        if results.multi_hand_landmarks:
                            for hand_lms in results.multi_hand_landmarks:
                                mp_draw.draw_landmarks(
                                    frame,
                                    hand_lms,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style(),
                                )

                        sequence.append(extract(results))

                        # Red recording indicator
                        cv2.circle(frame, (35, 45), 12, (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            f"RECORDING... [{frame_idx + 1}/{SEQ_LEN}]",
                            (60, 52),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow(window_name, frame)
                        cv2.waitKey(1)

                    # Ensure sequence is complete
                    if len(sequence) == SEQ_LEN:
                        out_path = class_dir / f"{SIGNER_ID}_{current_sample_num:03d}.npy"
                        np.save(out_path, np.asarray(sequence, dtype=np.float32))
                        print(f"Saved: {out_path.name} (samples collected for '{cls}': {sample_count + 1})")
                        sample_count += 1
                    else:
                        print(f"Warning: Discarded incomplete recording ({len(sequence)}/{SEQ_LEN} frames).")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection session finished. Camera released.")


if __name__ == "__main__":
    main()
