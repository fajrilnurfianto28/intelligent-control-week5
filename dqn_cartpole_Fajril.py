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

# --- 1. Inisialisasi Lingkungan dan Parameter DRL ---

# Menggunakan CartPole-v1
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameter yang Disesuaikan (Epsilon Decay lebih cepat: 0.99 -> 0.995)
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99 # Dipercepat dari 0.995
batch_size = 32
memory = deque(maxlen=2000)
TARGET_UPDATE_FREQ = 10 # Frekuensi pembaruan Jaringan Target

# --- 2. Membangun Model Deep Q-Network (DQN) dan Jaringan Target ---

def build_dqn_model():
    """Membangun model jaringan saraf untuk Q-function (Q-Network)."""
    model = keras.Sequential([
        keras.layers.Dense(24, input_shape=(state_size,), activation="relu", kernel_initializer='he_uniform'),
        keras.layers.Dense(24, activation="relu", kernel_initializer='he_uniform'),
        keras.layers.Dense(action_size, activation="linear")
    ])
    # Menggunakan Huber loss lebih stabil daripada MSE untuk DRL
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
    # Anda juga dapat menggunakan soft update (tau=0.01), tetapi hard update lebih umum untuk DQN dasar.

# --- 4. Fungsi Pemilihan Aksi (Epsilon-Greedy) ---

def select_action(state, epsilon_val, main_model):
    """Memilih aksi menggunakan strategi epsilon-greedy."""
    if np.random.rand() <= epsilon_val:
        return env.action_space.sample() # Memilih aksi acak
    
    # Perluasan dimensi state jika hanya satu sampel (reshape sudah dilakukan di loop utama)
    q_values = main_model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# --- 5. Fungsi Pelatihan Model (Experience Replay Vectorized) ---

def train_model(memory, batch_size, main_model, target_model, gamma_val):
    """
    Pelatihan model menggunakan Experience Replay yang tervektor.
    Ini jauh lebih cepat daripada iterasi satu per satu.
    """
    if len(memory) < batch_size:
        return

    # Ambil minibatch acak
    minibatch = random.sample(memory, batch_size)

    # Pisahkan state, action, reward, next_state, done ke dalam array NumPy
    states = np.array([t[0][0] for t in minibatch])
    actions = np.array([t[1] for t in minibatch])
    rewards = np.array([t[2] for t in minibatch])
    next_states = np.array([t[3][0] for t in minibatch])
    dones = np.array([t[4] for t in minibatch])

    # 1. Prediksi Q-value untuk next_state menggunakan Target Model (Stabilitas)
    # Target Model adalah yang tidak sering diperbarui.
    future_q_values = target_model.predict(next_states, verbose=0)
    
    # Q-value maksimum berikutnya
    max_future_q = np.amax(future_q_values, axis=1)

    # 2. Hitung Target Q-value (target = reward + gamma * max_future_q)
    # Jika episode selesai (done=True), future Q-value adalah 0.
    targets = rewards + gamma_val * max_future_q * (1 - dones)
    
    # 3. Prediksi Q-value untuk state saat ini menggunakan Main Model
    # Ini akan menjadi basis untuk pembaruan (target_f)
    target_f = main_model.predict(states, verbose=0)
    
    # 4. Update Q-value hanya untuk aksi yang dipilih (aksi[i])
    # Q-value dari aksi yang dipilih diganti dengan target yang telah dihitung
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
    # State harus dalam bentuk (1, 4) agar cocok dengan input model
    state = np.reshape(state, [1, state_size])
    score = 0

    for time_step in range(500):
        # Pilih aksi menggunakan Epsilon-Greedy
        action = select_action(state, epsilon, model)
        
        # Lakukan langkah di lingkungan
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Bentuk ulang next_state untuk memori dan input model
        next_state = np.reshape(next_state, [1, state_size])
        
        # Simpan pengalaman ke memori
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        score += reward
        
        if done:
            break

    episode_scores.append(time_step + 1)

    # Lakukan pelatihan jika memori sudah mencukupi
    train_model(memory, batch_size, model, target_model, gamma)

    # Update Epsilon (pengurangan eksplorasi)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update Jaringan Target secara periodik
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        update_target_model(model, target_model)

    # Tampilkan kemajuan
    avg_score = np.mean(episode_scores[-10:]) if len(episode_scores) >= 10 else np.mean(episode_scores)
    print(f"Episode: {episode + 1}/{MAX_EPISODES}, Score: {time_step + 1}, Avg (10): {avg_score:.2f}, Epsilon: {epsilon:.4f}")

    # Kriteria Keberhasilan (Rata-rata skor 195 dalam 10 episode terakhir)
    if len(episode_scores) >= 10 and avg_score >= 195:
        end_time = time.time()
        print("\n" + "=" * 60)
        print(f"!!! Selesai Lebih Cepat: Agen dianggap berhasil menguasai CartPole !!!")
        print(f"Berhasil dicapai dalam {episode + 1} episode. Total waktu: {end_time - start_time:.2f} detik.")
        print("=" * 60)
        break

print("-" * 50)
print("Training selesai!")
# model.save("cartpole_dqn_optimized_weights.h5")