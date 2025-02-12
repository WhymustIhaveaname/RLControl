#! /usr/bin/env python3

"""
Try gymnasium package with a simple inverted pendulum.

Author: Youran
Date: 2024-12-08
"""

import types
import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play
from gymnasium.utils.save_video import save_video

def try_cartpole():
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    observation, info = env.reset(seed=42)
    step_starting_index = 0
    episode_index = 0
    step_index = 0
    simulation_count = 0
    live_steps = []
    while simulation_count < 100:
        # action = env.action_space.sample() # 24.25, 1.4
        # action = 0 if observation[2] > 0 else 1 # 8.75, 0.05
        action = 0 if observation[2] < 0 else 1 # 42.09, 0.93
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"end at step {step_index}")
            frames = env.render()
            save_video(
                frames=frames,
                video_folder="videos",
                fps=env.metadata["render_fps"],
                step_starting_index=step_starting_index,
                episode_index=episode_index
            )
            live_steps.append(step_index - step_starting_index + 1)
            step_starting_index = step_index + 1
            episode_index += 1
            observation, info = env.reset()
            simulation_count += 1

        step_index += 1

    avg_steps = step_index / simulation_count
    std_steps = np.std(live_steps)/np.sqrt(simulation_count)
    print(f"avg steps per simulation: {avg_steps}, {std_steps}")
    env.close()

def user_play():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    play(env, keys_to_action={"a": 0, "d": 1})

if __name__ == "__main__":
    # user_play()
    try_cartpole()
