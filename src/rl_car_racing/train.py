"""Training entry point for the CarRacing DQN showcase."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from .agent import AgentConfig, DQNAgent


def shape_reward(reward: float) -> float:
    """Small reward-shaping helper used in the original university prototype."""
    if reward < 0:
        return -2.0
    if reward > 0:
        return float(reward * 1.5)
    return float(reward)


def plot_results(rewards: list[float], epsilons: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(len(rewards))
    moving = np.convolve(rewards, np.ones(10) / 10, mode="valid") if len(rewards) >= 10 else []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(episodes, rewards, alpha=0.45, label="episode reward")
    if len(moving):
        axes[0].plot(np.arange(len(moving)) + 9, moving, linewidth=2, label="10-episode moving avg")
    axes[0].set_title("Training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(episodes, epsilons, color="tab:orange")
    axes[1].set_title("Exploration schedule")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Epsilon")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    env = gym.make("CarRacing-v3", render_mode="human" if args.render else "rgb_array")
    agent = DQNAgent(
        AgentConfig(
            learning_rate=args.learning_rate,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            train_start=args.train_start,
        )
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rewards: list[float] = []
    epsilons: list[float] = []
    reward_window: deque[float] = deque(maxlen=10)
    best_average = -float("inf")

    print(f"device={agent.device} episodes={args.episodes} render={args.render}")

    for episode in range(args.episodes):
        frame, _ = env.reset()
        agent.reset_frames()
        state = agent.push_frame(frame)
        total_reward = 0.0
        losses: list[float] = []

        for step in range(args.max_steps):
            action_index = agent.act(state, training=True)
            next_frame, reward, terminated, truncated, _ = env.step(agent.actions[action_index])
            next_state = agent.push_frame(next_frame)
            done = terminated or truncated

            agent.remember(state, action_index, shape_reward(float(reward)), next_state, done)
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += float(reward)
            if done:
                break

        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        reward_window.append(total_reward)
        rolling_average = float(np.mean(reward_window))
        average_loss = float(np.mean(losses)) if losses else 0.0

        print(
            f"episode={episode:04d} reward={total_reward:8.2f} "
            f"avg10={rolling_average:8.2f} epsilon={agent.epsilon:.3f} loss={average_loss:.4f}"
        )

        if rolling_average > best_average:
            best_average = rolling_average
            agent.save(str(checkpoint_dir / "car-racing-dqn-best.pth"))

    env.close()
    agent.save(str(checkpoint_dir / "car-racing-dqn-final.pth"))
    plot_results(rewards, epsilons, Path(args.output_plot))
    print(f"saved checkpoints to {checkpoint_dir}")
    print(f"saved plot to {args.output_plot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Gymnasium CarRacing.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--render", action="store_true", help="Show the live CarRacing window during training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--train-start", type=int, default=1_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-plot", default="assets/training-results.png")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
