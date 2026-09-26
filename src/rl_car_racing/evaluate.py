"""Run a trained CarRacing DQN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

from .agent import DQNAgent


def evaluate(args: argparse.Namespace) -> None:
    env = gym.make("CarRacing-v3", render_mode="human" if args.render else "rgb_array")
    agent = DQNAgent()
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    rewards: list[float] = []
    for episode in range(args.episodes):
        frame, _ = env.reset()
        agent.reset_frames()
        state = agent.push_frame(frame)
        total_reward = 0.0

        for _ in range(args.max_steps):
            action_index = agent.act(state, training=False)
            frame, reward, terminated, truncated, _ = env.step(agent.actions[action_index])
            state = agent.push_frame(frame)
            total_reward += float(reward)
            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"episode={episode:03d} reward={total_reward:.2f}")

    env.close()
    print(f"average_reward={sum(rewards) / len(rewards):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CarRacing DQN checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/car-racing-dqn-best.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
