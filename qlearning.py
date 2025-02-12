#! /usr/bin/env python3

"""
Adapted from https://github.com/RJBrooker/Q-learning-demo-Cartpole-V1/
"""

import math
from typing import Tuple
from tqdm import tqdm

import numpy as np
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from sklearn.preprocessing import KBinsDiscretizer

def qlearning_cartpole():
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")

    n_bins = ( 6 , 12 )
    Q_table = np.zeros(n_bins + (env.action_space.n,))

    lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
    upper_bounds = [env.observation_space.high[2],  math.radians(50) ]
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])

    def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
        state = [[angle, pole_velocity]]
        state = est.transform(state)
        return tuple(map(int,state[0]))

    policy = lambda state: np.argmax(Q_table[state])
    exploration_rate = lambda n, min_rate=0.1: max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))
    learning_rate = lambda n, min_rate=0.01: max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))

    def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
        future_optimal_value = np.max(Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    n_episodes = 200
    best_sofar = 0
    for e in range(n_episodes):
        obs, _ = env.reset()
        current_state = discretizer(*obs)
        done = False

        while done==False:
            # exploit
            action = policy(current_state)

            # explore
            if np.random.random() < exploration_rate(e) :
                action = env.action_space.sample()

            # increment enviroment
            obs, reward, done, *_ = env.step(action)
            new_state = discretizer(*obs)

            # Update Q-Table
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward , new_state)
            old_value = Q_table[current_state][action]
            Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value

            current_state = new_state
        else:
            frames = env.render()
            if len(frames) > 1.1 * best_sofar:
                best_sofar = len(frames)
                save_video(
                    frames=frames,
                    episode_trigger=lambda x: True,
                    video_folder="videos",
                    name_prefix="qlearning",
                    episode_index=e,
                    fps=env.metadata["render_fps"]
                )
                print("new best at episode %d: %d frames, saved" % (e, best_sofar))

    print("Q mean std min max", Q_table.mean(),Q_table.std(),Q_table.min(),Q_table.max())

if __name__ == "__main__":
    qlearning_cartpole()
