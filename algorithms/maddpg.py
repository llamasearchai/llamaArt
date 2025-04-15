import torch
import torch.nn as nn


class MADDPG:
    def __init__(self, state_dim, action_dim, num_agents):
        # ... existing code ...
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
