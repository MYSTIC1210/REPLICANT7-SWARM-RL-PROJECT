# REPLICANT-7 — Multi-Agent Reinforcement Learning Swarm

A Saronic-style autonomous maritime drone coordination simulator built in React, featuring MADDPG-based swarm policy learning, sparse communication constraints, emergent behaviour, and a live policy visualiser.

## Overview

REPLICANT-7 is a full multi-agent RL simulation platform rendered as a single-file React (JSX) application. Eight autonomous maritime drones coordinate to patrol, deny, and reconnaissance three operational zones — under realistic acoustic communication bandwidth limits — while the system tracks training metrics, epsilon decay, reward accumulation, and emergent formation behaviour in real time.

## Features

- **8 independent MADDPG agents** operating simultaneously in a 720×680 simulation arena
- **3 operational zones** — Patrol, Deny, Recon — with dynamic threat targets
- **Acoustic comm budget** — max 3 messages per agent per tick (bandwidth-constrained coordination)
- **Communication range** — 160px acoustic link threshold with visible network overlay
- **Lévy-flight + RL hybrid navigation** — exploration vs exploitation balance
- **Live training metrics** — episode rewards, epsilon decay curve, policy loss, critic convergence
- **Trail history** — 55-frame agent path visualization
- **Emergent formation detection** — automatic identification of clustering and spread patterns
- **Military HUD interface** — Rajdhani/Share Tech Mono fonts, dark navy palette, pulsing alerts

## RL Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.95 | Discount factor |
| `LR_ACTOR` | 3×10⁻⁴ | Actor learning rate |
| `LR_CRITIC` | 1×10⁻³ | Critic learning rate |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.05 | Final exploration rate |
| `EPSILON_DECAY` | 0.9995 | Per-step decay multiplier |
| `REWARD_COVERAGE` | +1.2 | Zone coverage reward |
| `REWARD_COMM_PENALTY` | −0.4 | Over-broadcast penalty |
| `REWARD_COLLISION` | −2.0 | Inter-agent collision penalty |
| `REWARD_ZONE_CLEAR` | +3.0 | Zone cleared of threats |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 (JSX) |
| Simulation | Custom physics engine (tick-based, 52ms/frame) |
| RL Algorithm | MADDPG (Multi-Agent DDPG, simulated) |
| State | `useState`, `useEffect`, `useRef`, `useCallback` |
| Styling | Inline CSS-in-JS with design token system |
| Fonts | Rajdhani, Share Tech Mono (Google Fonts) |

## Design System

```
Color Tokens
├── r  Red    #ff3355  — threats / critical alerts
├── g  Green  #00f5a0  — nominal / zone clear
├── b  Blue   #00b4ff  — comms / data
├── y  Yellow #ffcc00  — warnings
├── o  Orange #ff7733  — active engagement
└── p  Purple #cc44ff  — analytics / RL metrics

Typography
├── FH  Rajdhani      — tactical headers
└── FM  Share Tech Mono — telemetry + metric readouts
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/MYSTIC1210/replicant7-swarm-rl.git
cd replicant7-swarm-rl

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

Or embed `swarm_rl.jsx` directly in any React + Vite project.

## Project Structure

```
replicant7-swarm-rl/
├── swarm_rl.jsx            # Full simulation — single-file React app
├── README.md
└── REPLICANT7_Report.docx  # System design and algorithm reference
```

## Simulation Architecture

```
REPLICANT-7 Simulation Loop (52ms tick)
│
├── Environment Update
│   ├── Threat position updates (3 dynamic targets per zone)
│   └── Zone status evaluation (patrol / deny / recon)
│
├── Agent Decision (×8)
│   ├── Sensor scan (90px radius)
│   ├── Communication broadcast (budget-limited, 160px range)
│   ├── MADDPG policy query (actor network)
│   └── Action execution (velocity + heading)
│
├── Reward Computation
│   ├── Coverage bonus, comm penalty, collision check
│   └── Zone clear bonus
│
└── Visualisation
    ├── Agent trails (55 frames)
    ├── Comm network overlay
    ├── Zone heat map
    └── Training metrics panel
```

## Use Cases

- Autonomous maritime surface vehicle (MSV) swarm coordination research
- Multi-agent RL algorithm visualisation and education
- Naval threat denial and area coverage simulation
- Emergent communication protocol studies

## Author

**Dinesh E** — [LinkedIn](https://www.linkedin.com/in/dinesh-ravilla1210) | [GitHub](https://github.com/MYSTIC1210)
