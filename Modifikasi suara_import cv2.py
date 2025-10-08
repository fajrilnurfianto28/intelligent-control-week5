import cv2
import mediapipe as mp
from gtts import gTTS
import playsound
import os
import time
import threading

# === Inisialisasi MediaPipe ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# === Buat folder penyimpanan suara ===
os.makedirs("sounds", exist_ok=True)

# === Fungsi deteksi jari terbuka ===
def jari_terbuka(hand_landmarks):
    jari = []
    tips = [4, 8, 12, 16, 20]
    for i, tip in enumerate(tips):
        if i == 0:  # jempol arah x
            jari.append(hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x)
        else:       # jari lain arah y
            jari.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)
    return jari  # hasil: [jempol, telunjuk, tengah, manis, kelingking]

# === Fungsi mainkan suara di thread terpisah (non-blocking) ===
def play_audio_async(text):
    def worker():
        try:
            audio_file = os.path.join("sounds", f"{text}_{int(time.time())}.mp3")
            tts = gTTS(text, lang="id")
            tts.save(audio_file)
            playsound.playsound(audio_file)
        except Exception as e:
            print("Gagal memainkan suara:", e)

    threading.Thread(target=worker, daemon=True).start()

# === Kamera ===
cap = cv2.VideoCapture(0)
played_phrase = None
last_play_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca kamera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w, _ = frame.shape

        phrase = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                jari_status = jari_terbuka(hand_landmarks)

                # === Logika gesture ===
                if jari_status == [False, True, False, False, False]:
                    phrase = "Perkenalkan"
                elif jari_status == [False, True, True, False, False]:
                    phrase = "Saya"
                elif jari_status == [True, True, False, False, True]:
                    phrase = "Fajril"
                elif all(jari_status):  # semua jari terbuka
                    phrase = "Terima kasih"

                # === Tampilkan di layar ===
                if phrase:
                    cv2.putText(frame, phrase, (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

                # === Suara non-blocking ===
                if phrase and phrase != played_phrase:
                    current_time = time.time()
                    if current_time - last_play_time > 2:  # jeda 2 detik
                        print(f"Gesture terdeteksi: {phrase}")
                        play_audio_async(phrase)
                        played_phrase = phrase
                        last_play_time = current_time

        cv2.imshow("Gesture Voice (Perkenalkan Saya Fajril + Terima Kasih)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
