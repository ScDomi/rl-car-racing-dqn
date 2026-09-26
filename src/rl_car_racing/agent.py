"""DQN agent implementation for Gymnasium CarRacing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .model import CarRacingDQN
from .preprocessing import preprocess_frame


@dataclass
class AgentConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99
    batch_size: int = 64
    replay_memory_size: int = 50_000
    train_start: int = 1_000
    target_update_steps: int = 5_000


class DQNAgent:
    """Deep Q-learning agent with replay memory and a target network."""

    def __init__(self, config: AgentConfig | None = None, action_size: int = 5) -> None:
        self.config = config or AgentConfig()
        self.action_size = action_size
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.config.replay_memory_size
        )
        self.frame_stack: deque[np.ndarray] = deque(maxlen=4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = CarRacingDQN(action_size).to(self.device)
        self.target_network = CarRacingDQN(action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        # Discretized controls for the continuous CarRacing action space: steer, gas, brake.
        self.actions = [
            np.array([-0.8, 0.0, 0.0], dtype=np.float32),
            np.array([0.8, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.8, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.8], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        ]

    def reset_frames(self) -> None:
        self.frame_stack.clear()

    def push_frame(self, frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(frame)
        self.frame_stack.append(processed)
        while len(self.frame_stack) < 4:
            self.frame_stack.append(np.zeros_like(processed))
        return self.stacked_state()

    def stacked_state(self) -> np.ndarray:
        return np.stack(self.frame_stack, axis=0)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def replay(self) -> float | None:
        if len(self.memory) < self.config.train_start:
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        states = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.bool, device=self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.q_network(next_states).argmax(dim=1)
        next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + self.config.gamma * next_q * (~dones)

        loss = F.mse_loss(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.config.epsilon_min:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
