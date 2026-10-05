import numpy as np

class ThompsonQLearningAgent:
    """
    Q-Learning Agent integrated with Thompson Sampling for balancing 
    exploration and exploitation in frequency band selection.
    """
    def __init__(self, action_space_size, num_bins=10):
        self.action_space_size = action_space_size
        self.num_bins = num_bins
        # Beta distribution parameters (Alpha: successes + 1, Beta: failures + 1)
        self.alpha_params = np.ones((num_bins, action_space_size))
        self.beta_params = np.ones((num_bins, action_space_size))
        
    def _get_bin_idx(self, freq, freq_min=1000.0, freq_max=2000.0):
        bin_edge = (freq_max - freq_min) / self.num_bins
        idx = int((freq - freq_min) / bin_edge)
        return min(max(idx, 0), self.num_bins - 1)

    def select_action(self, state):
        freq = state[0]
        bin_idx = self._get_bin_idx(freq)
        
        # Thompson Sampling: sample from Beta distribution for each action
        samples = [np.random.beta(self.alpha_params[bin_idx, a], self.beta_params[bin_idx, a]) 
                   for a in range(self.action_space_size)]
        
        return int(np.argmax(samples))

    def update(self, state, action, reward, next_state):
        bin_idx = self._get_bin_idx(state[0])
        if reward > 0:
            self.alpha_params[bin_idx, action] += 1.0 # Hit success update
        else:
            self.beta_params[bin_idx, action] += 1.0 # Miss failure update