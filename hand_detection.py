import cv2
import mediapipe as mp
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

csv_file = "hand_joints.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as f:
    csv_writer = csv.writer(f)
    if not file_exists:
        header = ["frame", "hand", "joint_id", "x", "y", "z"]
        csv_writer.writerow(header)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca kamera")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                label = handedness.classification[0].label  # "Right" atau "Left"

                # Gambar landmark & koneksi
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                )

                # Gambar nomor per sendi (0–20)
                for j, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(frame, str(j), (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                # Hitung bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
                ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

                # Gambar bounding box (warna hijau tebal)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

                # Tambahkan background untuk teks biar jelas
                label_text = f"{label} Hand"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (xmin, ymin - th - 6), (xmin + tw, ymin), (0, 255, 0), -1)  # kotak isi
                cv2.putText(frame, label_text, (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # teks hitam

                # Simpan ke CSV (joint_id = 0–20)
                with open(csv_file, mode="a", newline="") as f:
                    csv_writer = csv.writer(f)
                    for j, lm in enumerate(hand_landmarks.landmark):
                        csv_writer.writerow([frame_count, label, j, lm.x, lm.y, lm.z])

        frame_count += 1
        cv2.imshow("Hand Detection with Bounding Box", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
