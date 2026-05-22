# Sample Output — REPLICANT-7 Multi-Agent Swarm RL

## Browser Console (Dev Mode)

```
[REPLICANT-7] Simulation initialised — 8 agents, 3 zones
[REPLICANT-7] Episode 1 started — ε=1.000 (full exploration)
[REPLICANT-7] Episode 47 — ε=0.794  avg_reward=+1.24  zone_clears=2
[REPLICANT-7] Episode 312 — ε=0.213  avg_reward=+4.87  zone_clears=11
[REPLICANT-7] Episode 800 — ε=0.059  avg_reward=+7.31  zone_clears=24
[REPLICANT-7] Emergent behaviour: CLUSTER formation detected (agents 2,5,7)
```

## Agent Status (Episode 800, Tick 1,240)

```
AGENT  │  ZONE    │  STATUS    │  COMM MSG  │  REWARD
───────┼──────────┼────────────┼────────────┼────────
AGT-0  │  PATROL  │  ACTIVE    │  2/3 used  │  +1.2
AGT-1  │  DENY    │  ENGAGING  │  3/3 used  │  +3.0 ← zone clear
AGT-2  │  RECON   │  ACTIVE    │  1/3 used  │  +1.2
AGT-3  │  PATROL  │  ACTIVE    │  0/3 used  │  +1.2
AGT-4  │  DENY    │  ENGAGING  │  3/3 used  │  +3.0 ← zone clear
AGT-5  │  RECON   │  CLUSTER   │  2/3 used  │  +1.2
AGT-6  │  PATROL  │  ACTIVE    │  1/3 used  │  +1.2
AGT-7  │  RECON   │  CLUSTER   │  2/3 used  │  +1.2
```

## Training Metrics

```
Episode   │  ε (Epsilon)  │  Avg Reward  │  Zone Clears
──────────┼───────────────┼──────────────┼─────────────
   1      │   1.000       │    -0.41     │     0
  50      │   0.779       │    +1.18     │     3
 100      │   0.607       │    +2.34     │     7
 200      │   0.368       │    +4.91     │    15
 400      │   0.135       │    +6.44     │    22
 800      │   0.059       │    +7.31     │    24
```

## Communication Budget Analysis

```
Avg messages sent per agent per tick: 1.8 / 3.0 budget
Over-budget violations: 0 (penalty avoidance learned by ep 150)
Comm range utilisation: 62% of links within 160px threshold
```

## Emergent Behaviour Detected

```
Episode 800+ — CLUSTER formation (AGT-2, AGT-5, AGT-7)
  → Agents self-organised into recon triangle covering zone C
  → No explicit clustering reward — emergent from shared KM feedback
  → Formation stability: 87% of ticks within 40px of centroid
```
