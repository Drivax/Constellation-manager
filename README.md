# Constellation Manager

For readers who are not familiar with space systems, a satellite constellation is a group of satellites that work together in orbit.
They are used for communication, Earth observation, navigation, and scientific missions.

Managing a constellation is difficult because each satellite moves fast, has limited control authority, and still affects the quality of the global system.
Even a small error in phase, spacing, or control usage can accumulate over time.

This project studies that problem in a simplified but still physically grounded setting.
It uses real orbital data, real propagation equations, and a cooperative learning setup to explore how a set of agents could help maintain an organized constellation state.

## Visual Context

The images below are real spacecraft examples.
They are here to give visual context before getting into the technical details.

![Hubble Space Telescope](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/HST-SM4.jpeg/640px-HST-SM4.jpeg)

![International Space Station](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/International_Space_Station_after_undocking_of_STS-132.jpg/640px-International_Space_Station_after_undocking_of_STS-132.jpg)

## Project Objective

This project trains a small multi-agent reinforcement learning system to manage a real satellite constellation sample.
It uses 100 Starlink satellites, real TLE data, SGP4 orbit propagation, anomaly detection, and cooperative control.

The main idea is simple.
Each satellite is treated as an agent.
Each agent observes a compact description of its orbital situation and chooses a discrete control action.
All agents together try to keep the constellation organized while avoiding unnecessary control effort and unusual orbital states.

The project objective is to connect several pieces that are often studied separately.
It combines real TLE ingestion, physical propagation, anomaly scoring, multi-agent policy learning, checkpointing, inference, and visual analysis in one runnable pipeline.

It is not meant to replace real flight dynamics software.
It is meant to be a careful experimental baseline.
The value of the project is that the whole workflow can be inspected, reproduced, and extended from one repository.

## Dataset

The orbital data comes from Celestrak.
The project downloads the Starlink group in TLE format from this source:

- https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle

For this minimal version, the code keeps the first 100 satellites.
Each record contains a satellite name, TLE line 1, and TLE line 2.

These TLEs give compact orbital parameters.
They are converted into SGP4 satellite objects and propagated step by step during training and evaluation.

Key features used in the learning pipeline are based on propagated orbital state.
They include radius, speed, inclination, eccentricity, and phase features built from sine and cosine terms.

## Methodology

### 1. Real orbit simulation

At each step, the environment propagates every satellite with SGP4.
This gives a physically grounded position and velocity instead of an artificial toy orbit.

The environment then derives compact control features from the propagated state.
This keeps the learning problem manageable while still tied to real orbital data.

### 2. Multi-agent control with MAPPO

Each satellite is treated as one agent.
All agents share the same actor network.
This reduces the number of parameters and makes training more stable.

The critic uses a global summary of the constellation state.
This is the usual centralized-training, decentralized-execution idea.
During execution, each satellite still acts from its local observation.

The reward penalizes phase error, altitude error, control effort, and anomaly score.
The agents therefore learn a compromise between coordination, stability, and cautious control.

### 3. Anomaly detection and analysis

A small autoencoder is trained on nominal orbital feature vectors.
Its reconstruction error is used as an anomaly score.

This score is added to the observation and also used in the reward.
In simple terms, the policy is encouraged to avoid states that look unusual compared with the nominal orbital patterns seen during autoencoder training.

After training, the project exports metrics, checkpoints, a lightweight actor policy, a 3D snapshot, and an animated GIF.
This makes the behavior easier to inspect and reuse.

## Key Equations

### Orbit propagation

$$
\mathbf{x}_{t+1} = \mathrm{SGP4}(\mathrm{TLE}, t)
$$

This means the satellite state at time $t+1$ comes from the SGP4 propagation model applied to the TLE data.
Here, $\mathbf{x}$ contains the propagated position and velocity.

### Autoencoder reconstruction loss

$$
\mathcal{L}_{\mathrm{AE}} = \frac{1}{d} \sum_{i=1}^{d} \left(f_i - \hat{f}_i\right)^2
$$

This is the mean squared reconstruction error.
If the autoencoder reconstructs a feature vector badly, the state is probably less typical.
That is why this value is used as an anomaly score.

### PPO clipped objective

$$
\mathcal{L}_{\mathrm{clip}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

This is the core PPO objective.
It limits how much the new policy can move away from the old one in one update.
That usually makes training more stable.

### Generalized Advantage Estimation

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

These equations define the temporal-difference residual and the advantage estimate.
In practice, this helps the agent learn from delayed effects without making the updates too noisy.

## Evaluation

The values below come from the latest generated training and evaluation artifacts in the repository.

- Evaluation episode reward: `-192.5973`
- Mean phase error: `1.5635`
- Mean altitude error: `0.006316`
- Mean anomaly score: `0.9569`

Final training values from the last saved iteration:

- Mean reward: `-2.2510`
- Phase error: `1.6836`
- Altitude error: `0.006254`
- Anomaly score: `0.8952`
- Actor loss: `0.0118`
- Critic loss: `297.1541`
- Entropy: `1.0544`

These numbers should be read as a compact baseline.
They are useful for checking that the pipeline is running correctly and producing consistent artifacts.

## Results and Graphs

### Final constellation snapshot

This image shows the final propagated 3D constellation state produced by the current pipeline.

![Final constellation state](outputs/constellation_final.png)

### Training history

This graph shows how reward, phase error, and anomaly score evolve during training.

![Training metrics](outputs/training_metrics.png)

### Animated rollout

This GIF shows the constellation trajectory over time.

![Constellation animation](outputs/constellation_animation.gif)

## Repository Structure

```text
Constellation-manager/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── inference.py
├── environment.py
├── train.py
├── data/
├── models/
│   └── agent.py
├── utils/
│   ├── tle_loader.py
│   └── visualization.py
└── outputs/
```

## Installation and Execution

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run training and evaluation:

```bash
python main.py
```

Resume from checkpoints from the command line:

```bash
python main.py --resume-mode latest
python main.py --resume-mode best
python main.py --resume-checkpoint-path outputs/checkpoints/mappo_iter_012.pt
```

Run deterministic inference from the exported actor policy:

```bash
python inference.py
python inference.py --policy-path outputs/policy_actor.pt --steps 60
```

Main output files:

- `outputs/constellation_final.png`
- `outputs/constellation_animation.gif`
- `outputs/training_metrics.csv`
- `outputs/training_metrics.json`
- `outputs/training_metrics.png`
- `outputs/evaluation_metrics.json`
- `outputs/inference_metrics.json`
- `outputs/policy_actor.pt`
- `outputs/checkpoints/`
