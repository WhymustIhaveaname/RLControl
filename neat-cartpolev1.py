#! /usr/bin/env python3

"""
[1] https://github.com/CodeReclaimers/neat-python/
[2] https://github.com/mvoelk/neat
[3] https://github.com/NirajSawant136/Simple-AI-using-NEAT uses [1]
[4] https://github.com/techwithtim/NEAT-Flappy-Bird/ uses [1]
"""

import os
import neat
import gymnasium as gym
from gymnasium.utils.save_video import save_video

env = gym.make("CartPole-v1", render_mode="rgb_array_list")

def eval_one_genome(genome, config):
    genome.fitness = 0  # start with fitness level of 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    obs, _ = env.reset()
    done = False

    while not done and genome.fitness < 1500:
        action = net.activate(obs)
        action = 1 if action[0] > 0 else 0
        obs, reward, done, *_ = env.step(action)
        genome.fitness += reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_one_genome(genome, config)
        # print(genome_id, genome.fitness)


def main(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(eval_genomes, 50)

    print("Best fitness -> {}".format(winner))

if __name__ == "__main__":
    main("config-cartpolev1.ini")
