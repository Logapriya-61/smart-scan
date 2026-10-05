import matplotlib.pyplot as plt
import numpy as np

def plot_rf_spectrum_and_grid(env):
    """Plots the synthetic RF spectrum ground truth and the band-time grid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Frequency Spectrum Ground Truth
    freqs = np.linspace(env.freq_min, env.freq_max, 500)
    spectrum = np.zeros_like(freqs)
    for sig in env.signals:
        spectrum += sig["power"] * np.exp(-((freqs - sig["freq"]) / (sig["bw"] / 2))**2)
    spectrum += np.random.normal(0, 0.02, size=freqs.shape) # Noise floor
    
    axes[0].plot(freqs, spectrum, color='cyan', lw=2)
    axes[0].set_facecolor('#0b0f19')
    axes[0].set_title("Synthetic RF Spectrum Ground Truth", color='white', fontsize=12)
    axes[0].set_xlabel("Frequency (MHz)", color='white')
    axes[0].set_ylabel("Power Level", color='white')
    axes[0].tick_params(colors='white')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # 2. Band-Time Grid (Transmission & Non-transmission)
    im = axes[1].imshow(env.history_grid, aspect='auto', cmap='inferno', extent=[env.freq_min, env.freq_max, env.max_steps, 0])
    axes[1].set_facecolor('#0b0f19')
    axes[1].set_title("Band-Time Transmission Grid (Scan History)", color='white', fontsize=12)
    axes[1].set_xlabel("Frequency (MHz)", color='white')
    axes[1].set_ylabel("Time Steps", color='white')
    axes[1].tick_params(colors='white')
    plt.colorbar(im, ax=axes[1], label='Detection Intensity')
    
    fig.patch.set_facecolor('#0b0f19')
    plt.tight_layout()
    plt.show()

def plot_thompson_sampling_behavior(before_hits, after_hits):
    """Shows behavior changes of receiver before and after learning via Thompson Sampling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(before_hits, color='coral', alpha=0.7)
    axes[0].set_title("Early Episodes (High Exploration / Random Hits)", color='white')
    axes[0].set_xlabel("Episode Steps", color='white')
    axes[0].set_ylabel("Hit Success", color='white')
    axes[0].set_facecolor('#0b0f19')
    axes[0].tick_params(colors='white')
    
    axes[1].plot(after_hits, color='lime', alpha=0.9)
    axes[1].set_title("Later Episodes (Exploitation via Thompson Sampling)", color='white')
    axes[1].set_xlabel("Episode Steps", color='white')
    axes[1].set_ylabel("Hit Success", color='white')
    axes[1].set_facecolor('#0b0f19')
    axes[1].tick_params(colors='white')
    
    fig.patch.set_facecolor('#0b0f19')
    plt.tight_layout()
    plt.show()