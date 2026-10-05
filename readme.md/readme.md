# 📡 Gymnasium Custom RF Spectrum & Receiver Simulator

> An advanced reinforcement learning closed-loop framework pairing a custom **Gymnasium Environment** with **Thompson Sampling** and **Q-Learning** to optimize cognitive radio receiver detection across complex agile and spatial RF spectrums.

---

## 🎯 Executive Summary & Judge Overview

When evaluating cognitive radios and autonomous electronic warfare/receiver systems, traditional scanning is inefficient against agile, hopping, or low-probability-of-intercept (LPI) emitters. This project builds a **fully compliant Gymnasium custom environment (`RFReceiverEnv`)** that simulates a realistic RF landscape. 

The repository demonstrates a **complete closed-loop learning cycle**:
1. **The Ground Truth Environment:** Generates dynamic, spatial, and agile RF emitter profiles alongside noise floors.
2. **The Perception & Observation Layer:** Maps spectrum tuning actions to energy detection, returning observation vectors and hit/miss metrics.
3. **Exploration vs. Exploitation (Thompson Sampling):** Utilizes Beta-distribution sampling over frequency bins to dynamically balance exploration and exploitation.
4. **Adaptive Q-Learning Control:** Trains an agent across episodes (progressing from early exploration chaos to late-stage optimized tracking) to maximize hit rates and minimize detection latency.

---

## 🏗️ Repository Architecture

```text
rf_simulator_project/
│
├── rf_env/
│   ├── __init__.py          # Gymnasium registry integration
│   └── rf_receiver_env.py   # Core Custom Gymnasium Environment (Dynamics, Physics, Reward)
│
├── agents/
│   ├── q_learning_agent.py  # Thompson Sampling & Q-Learning Agent logic
│   └── train.py             # Closed-loop multi-episode training pipeline
│
├── utils/
│   └── visualization.py     # High-fidelity spectrum, grid, and behavior plotting
│
├── requirements.txt         # Project dependencies
├── rf_spectrum_and_grid.png # Generated Ground Truth & Band-Time Grid visualization
└── thompson_sampling_behavior.png # Receiver behavior evolution visualization
```

---

## 🔬 Core Components Breakdown

| Component | Implementation File | What It Does in the Simulator |
| :--- | :--- | :--- |
| **Gymnasium Compliance** | `rf_env/rf_receiver_env.py` | Implements standard `gym.Env` with `reset()`, `step()`, `observation_space` (Box), and `action_space` (Discrete). |
| **RF Emitter Physics** | `rf_env/rf_receiver_env.py` | Simulates multi-signal ground truth emitters with frequency, bandwidth, power, and spatial agility characteristics. |
| **Band-Time Transmission Grid** | `rf_env/rf_receiver_env.py` & `train.py` | Logs historical scan bins over time steps, distinguishing active signal transmissions from noise floors. |
| **Thompson Sampling Engine** | `agents/q_learning_agent.py` | Maintains Beta-distribution ($\alpha, \beta$) parameters per frequency bin, sampling success probabilities to balance exploration/exploitation. |
| **Closed-Loop Training Loop** | `agents/train.py` | Manages episode resets, action selections, state transitions, experience updates, and performance tracking across 150 episodes. |

---

## 📊 Visualizations & Results

The training pipeline automatically generates diagnostic visualizations that highlight the receiver's adaptation:

### 1. Synthetic RF Spectrum & Band-Time Transmission Grid
* **Left Plot:** The ground truth power spectral density showing overlapping spatial/agile emitter peaks over noise.
* **Right Plot:** The band-time history grid tracking receiver sweeps, illuminating transmission windows vs. non-transmission zones.

![RF Spectrum and Grid](rf_spectrum_and_grid.png)

### 2. Receiver Behavior Evolution (Thompson Sampling Adaptation)
* **Early Episodes (Exploration):** High variance, random frequency jumps, and frequent missed signals as the agent maps the environment.
* **Later Episodes (Exploitation):** Converged, high-success hit patterns where the Thompson sampling agent directly targets high-probability signal corridors.

![Thompson Sampling Behavior](thompson_sampling_behavior.png)

---

## 🛠️ Installation & Getting Started

Follow these steps to run the simulation and verify training locally in Visual Studio Code:

### 1. Clone & Open Repository
Open your terminal inside VS Code at the project root folder:
```bash
cd rf_simulator_project
```

### 2. Set Up Virtual Environment
Create and activate a Python virtual environment:
* **Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training Pipeline
Execute the full training loop, which trains the agent across 150 episodes and outputs high-resolution diagnostic plots:
```bash
python agents/train.py
```

---

## 💡 Key Takeaways for Judges
* **Modular Design:** Clean separation of concerns between environment physics (`rf_env`), decision intelligence (`agents`), and data rendering (`utils`).
* **Advanced Decision Theory:** Standard Q-learning is augmented with **Thompson Sampling**, providing a probabilistic handle on uncertain RF bands superior to naive epsilon-greedy strategies.
* **Production-Ready Structure:** Fully compatible with standard reinforcement learning pipelines and ready for scaling to vectorised multi-environment setups.