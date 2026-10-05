import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RFReceiverEnv(gym.Env):
    """
    Custom Gymnasium Environment for an RF Spectrum Receiver Simulator.
    Simulates spatial/agile emitters, frequency-time grids, and receiver tuning.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, freq_range=(1000.0, 2000.0), num_signals=5, max_steps=100):
        super().__init__()
        
        self.freq_min, self.freq_max = freq_range
        self.num_signals = num_signals
        self.max_steps = max_steps
        
        # Action Space: Discrete tuning choices (e.g., jump or step frequency/bandwidth)
        # 0: Shift Down, 1: Shift Up, 2: Widen Bandwidth, 3: Narrow Bandwidth, 4: Dwell/Scan
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: [current_freq, current_bandwidth, detected_energy, hit_flag, time_step_norm]
        self.observation_space = spaces.Box(
            low=np.array([self.freq_min, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.freq_max, 100.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Internal states
        self.current_freq = (self.freq_min + self.freq_max) / 2.0
        self.current_bandwidth = 10.0
        self.step_count = 0
        self.signals = []
        self.history_grid = [] # Band-time transmission history
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_freq = np.random.uniform(self.freq_min, self.freq_max)
        self.current_bandwidth = 10.0
        
        # Generate ground-truth agile/spatial emitters (frequency, bandwidth, power, agility profile)
        self.signals = []
        for _ in range(self.num_signals):
            sig_freq = np.random.uniform(self.freq_min + 50, self.freq_max - 50)
            sig_bw = np.random.uniform(5.0, 25.0)
            sig_power = np.random.uniform(0.6, 1.0)
            self.signals.append({"freq": sig_freq, "bw": sig_bw, "power": sig_power})
            
        self.history_grid = np.zeros((self.max_steps, 50)) # 50 frequency bins
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1
        
        # Action-to-effect mapping
        if action == 0:
            self.current_freq = max(self.freq_min, self.current_freq - 15.0)
        elif action == 1:
            self.current_freq = min(self.freq_max, self.current_freq + 15.0)
        elif action == 2:
            self.current_bandwidth = min(50.0, self.current_bandwidth + 5.0)
        elif action == 3:
            self.current_bandwidth = max(2.0, self.current_bandwidth - 5.0)
        elif action == 4:
            pass # Dwell / Stay tuned
            
        # Receiver catches signal check (Hit or Miss evaluation)
        detected_energy = 0.0
        hit = 0
        for sig in self.signals:
            # Check if receiver window overlaps with emitter
            freq_overlap = abs(self.current_freq - sig["freq"]) <= (self.current_bandwidth + sig["bw"]) / 2.0
            if freq_overlap:
                detected_energy += sig["power"]
                hit = 1
                
        # Add noise
        detected_energy += np.random.normal(0, 0.05)
        detected_energy = np.clip(detected_energy, 0.0, 1.0)
        
        # Populate band-time grid record
        bin_idx = int((self.current_freq - self.freq_min) / (self.freq_max - self.freq_min) * 49)
        self.history_grid[min(self.step_count - 1, self.max_steps - 1), bin_idx] = 1 if hit else 0.2
        
        # Reward function: Reward for hits, penalties for false alarms / time wasted
        reward = 10.0 if hit else -1.0
        if action == 4 and hit:
            reward += 5.0 # bonus for smart dwelling
            
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        obs = self._get_obs(detected_energy, hit)
        info = self._get_info(hit=hit)
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self, energy=0.0, hit=0.0):
        return np.array([
            self.current_freq,
            self.current_bandwidth,
            energy,
            float(hit),
            self.step_count / self.max_steps
        ], dtype=np.float32)

    def _get_info(self, hit=0):
        return {
            "ground_truth_signals": self.signals,
            "current_hit": hit,
            "step_count": self.step_count
        }

    _get_info.__doc__ = "Helper that returns extra debug data, ground truth locations, and hit metrics."
    _get_obs.__doc__ = "Helper that converts internal state into observation format."