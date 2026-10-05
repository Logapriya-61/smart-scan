import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rf_env.rf_receiver_env import RFReceiverEnv
from agents.q_learning_agent import ThompsonQLearningAgent

def run_training():
    env = RFReceiverEnv()
    agent = ThompsonQLearningAgent(action_space_size=env.action_space.n)
    
    num_episodes = 150
    episode_rewards = []
    early_hits, later_hits = [], []
    
    print("--- Starting Closed-Loop RF Receiver Training ---")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        episode_hit_track = []
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
            episode_hit_track.append(info["current_hit"])
            
        episode_rewards.append(total_reward)
        
        # Capture stages for evaluation comparison
        if episode == 5:
            early_hits = episode_hit_track
        elif episode == num_episodes - 1:
            later_hits = episode_hit_track
            
        print(f"Completed Episode [{episode+1}/{num_episodes}] - Total Reward: {total_reward:.2f}")

    print("Training Complete! Generating and saving plots...")
    
    # 1. Save Spectrum & Grid Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    freqs = np.linspace(env.freq_min, env.freq_max, 500)
    spectrum = np.zeros_like(freqs)
    for sig in env.signals:
        spectrum += sig["power"] * np.exp(-((freqs - sig["freq"]) / (sig["bw"] / 2))**2)
    spectrum += np.random.normal(0, 0.02, size=freqs.shape)
    
    axes[0].plot(freqs, spectrum, color='cyan', lw=2)
    axes[0].set_facecolor('#0b0f19')
    axes[0].set_title("Synthetic RF Spectrum Ground Truth", color='white', fontsize=12)
    axes[0].set_xlabel("Frequency (MHz)", color='white')
    axes[0].set_ylabel("Power Level", color='white')
    axes[0].tick_params(colors='white')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    im = axes[1].imshow(env.history_grid, aspect='auto', cmap='inferno', extent=[env.freq_min, env.freq_max, env.max_steps, 0])
    axes[1].set_facecolor('#0b0f19')
    axes[1].set_title("Band-Time Transmission Grid (Scan History)", color='white', fontsize=12)
    axes[1].set_xlabel("Frequency (MHz)", color='white')
    axes[1].set_ylabel("Time Steps", color='white')
    axes[1].tick_params(colors='white')
    plt.colorbar(im, ax=axes[1], label='Detection Intensity')
    
    fig.patch.set_facecolor('#0b0f19')
    plt.tight_layout()
    plt.savefig("rf_spectrum_and_grid.png", dpi=300)
    plt.close()
    
    # 2. Save Thompson Sampling Behavior Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(early_hits, color='coral', alpha=0.7)
    axes[0].set_title("Early Episodes (High Exploration)", color='white')
    axes[0].set_xlabel("Episode Steps", color='white')
    axes[0].set_ylabel("Hit Success", color='white')
    axes[0].set_facecolor('#0b0f19')
    axes[0].tick_params(colors='white')
    
    axes[1].plot(later_hits, color='lime', alpha=0.9)
    axes[1].set_title("Later Episodes (Exploitation via Thompson Sampling)", color='white')
    axes[1].set_xlabel("Episode Steps", color='white')
    axes[1].set_ylabel("Hit Success", color='white')
    axes[1].set_facecolor('#0b0f19')
    axes[1].tick_params(colors='white')
    
    fig.patch.set_facecolor('#0b0f19')
    plt.tight_layout()
    plt.savefig("thompson_sampling_behavior.png", dpi=300)
    plt.close()
    
    print("Plots saved successfully as 'rf_spectrum_and_grid.png' and 'thompson_sampling_behavior.png'!")

if __name__ == "__main__":
    run_training()