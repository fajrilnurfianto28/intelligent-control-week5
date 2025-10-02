import numpy as np

# Patch untuk kompatibilitas Gymnasium/Gym lama dengan NumPy baru
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import time
# Import Matplotlib untuk visualisasi grafik
import matplotlib.pyplot as plt 

# --- 1. Inisialisasi Lingkungan dan Parameter DRL ---

# Menggunakan CartPole-v1
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)
TARGET_UPDATE_FREQ = 10 # Frekuensi pembaruan Jaringan Target

# --- 2. Membangun Model Deep Q-Network (DQN) dan Jaringan Target ---

def build_dqn_model():
    """Membangun model jaringan saraf untuk Q-function (Q-Network)."""
    model = keras.Sequential([
        # Menggunakan inisialisasi He/uniform untuk performa yang lebih baik
        keras.layers.Dense(24, input_shape=(state_size,), activation="relu", kernel_initializer='he_uniform'),
        keras.layers.Dense(24, activation="relu", kernel_initializer='he_uniform'),
        keras.layers.Dense(action_size, activation="linear")
    ])
    # Menggunakan Huber loss untuk stabilitas yang lebih baik dibanding MSE pada DRL
    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model

model = build_dqn_model()
# Target Model digunakan untuk menghitung target Q-value (Q-hat) demi stabilitas.
target_model = build_dqn_model()
target_model.set_weights(model.get_weights()) # Inisialisasi bobot sama

print("Model DQN dan Target Model berhasil dibuat.")
model.summary()
print("-" * 50)

# --- 3. Fungsi Pembaruan Jaringan Target ---

def update_target_model(main_model, target_model):
    """Menyalin bobot dari Main Model ke Target Model."""
    target_model.set_weights(main_model.get_weights())
    # Ini adalah 'Hard Update'

# --- 4. Fungsi Pemilihan Aksi (Epsilon-Greedy) ---

def select_action(state, epsilon_val):
    """Memilih aksi menggunakan strategi epsilon-greedy."""
    if np.random.rand() <= epsilon_val:
        return env.action_space.sample() # Memilih aksi acak
    
    # Perluasan dimensi state jika hanya satu sampel (reshape sudah dilakukan di loop utama)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# --- 5. Fungsi Pelatihan Model (Experience Replay Vectorized) ---

def train_model(memory, batch_size, main_model, target_model, gamma_val):
    """
    Pelatihan model menggunakan Experience Replay yang Tervektor.
    Ini jauh lebih cepat daripada iterasi satu per satu karena memanfaatkan NumPy dan GPU.
    """
    if len(memory) < batch_size:
        return

    # Ambil minibatch acak
    minibatch = random.sample(memory, batch_size)

    # Pisahkan data ke dalam array NumPy
    states = np.array([t[0][0] for t in minibatch])
    actions = np.array([t[1] for t in minibatch])
    rewards = np.array([t[2] for t in minibatch])
    next_states = np.array([t[3][0] for t in minibatch])
    # Mengkonversi boolean done ke integer (0 atau 1) untuk perhitungan
    dones = np.array([t[4] for t in minibatch], dtype=np.int32) 

    # 1. Prediksi Q-value untuk next_state menggunakan Target Model (Stabilitas)
    future_q_values = target_model.predict(next_states, verbose=0)
    max_future_q = np.amax(future_q_values, axis=1)

    # 2. Hitung Target Q-value secara vektor
    # (1 - dones) akan menjadi 0 jika done=True, sehingga target Q-value hanya = reward
    targets = rewards + gamma_val * max_future_q * (1 - dones)
    
    # 3. Prediksi Q-value untuk state saat ini menggunakan Main Model (target_f)
    target_f = main_model.predict(states, verbose=0)
    
    # 4. Update Q-value hanya untuk aksi yang dipilih
    for i in range(batch_size):
        target_f[i][actions[i]] = targets[i]

    # 5. Lakukan pelatihan (fit) untuk seluruh batch sekaligus
    main_model.fit(states, target_f, epochs=1, verbose=0)


# --- 6. Proses Training Utama ---

print("Memulai proses training...")
MAX_EPISODES = 1000
episode_scores = []
start_time = time.time()

for episode in range(MAX_EPISODES):
    state, _ = env.reset() 
    state = np.reshape(state, [1, state_size])
    
    # Inisialisasi 'time' di setiap episode (digunakan untuk skor/durasi)
    for time in range(500):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        
        if done:
            break

    episode_scores.append(time + 1) 

    # Panggil fungsi pelatihan yang telah divektorisasi
    train_model(memory, batch_size, model, target_model, gamma)

    # Update Epsilon (pengurangan eksplorasi)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # --- PENTING: Update Jaringan Target secara periodik ---
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        update_target_model(model, target_model)

    # Tampilkan kemajuan
    avg_score = np.mean(episode_scores[-10:]) if len(episode_scores) >= 10 else np.mean(episode_scores)
    print(f"Episode: {episode + 1}/{MAX_EPISODES}, Score: {time + 1}, Avg (10): {avg_score:.2f}, Epsilon: {epsilon:.4f}")

    # Kondisi Early Stopping
    if len(episode_scores) >= 10 and avg_score >= 195:
        end_time = time.time()
        print("\n" + "=" * 60)
        print(f"!!! Selesai Lebih Cepat: Agen dianggap berhasil menguasai CartPole !!!")
        print(f"Berhasil dicapai dalam {episode + 1} episode. Total waktu: {end_time - start_time:.2f} detik.")
        print("=" * 60)
        break

print("-" * 50)
print("Training selesai!")

# --- 7. Visualisasi Hasil ---

# Hitung rata-rata skor bergulir
rolling_mean = np.array(episode_scores)
rolling_mean = [np.mean(rolling_mean[max(0, i-10):i+1]) for i in range(len(rolling_mean))]


plt.figure(figsize=(10, 6))
plt.plot(episode_scores, label='Skor Tiap Episode', alpha=0.5)
plt.plot(rolling_mean, label='Rata-rata 10 Episode', color='red', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Skor (Durasi Hidup Tiang)')
plt.title('Optimized Deep Q-Network (DQN) - Performa CartPole')
plt.axhline(y=195, color='green', linestyle='--', label='Target Penyelesaian (195)')
plt.legend()
plt.show() # Tampilkan plot
