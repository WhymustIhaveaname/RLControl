#! /usr/bin/env python3

"""
Github: https://github.com/EMI-Group/tensorneat
Paper: https://arxiv.org/pdf/2404.01817
"""

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv,GymNaxEnv
from tensorneat.common import ACT, AGG

# problem = BraxEnv(
#     env_name="hopper",
#     max_step=1000,
# )
# algorithm = NEAT(
#     pop_size=1000,
#     species_size=20,
#     survival_threshold=0.1,
#     compatibility_threshold=1.0,
#     genome=DefaultGenome(
#         # num_inputs=11,
#         # num_outputs=3,
#         init_hidden_layers=(),
#         node_gene=BiasNode(
#             activation_options=ACT.tanh,
#             aggregation_options=AGG.sum,
#         ),
#         output_transform=ACT.tanh,
#     ),
# )

problem = BraxEnv(
    env_name="inverted_pendulum",
    max_step=1000,
)

# problem = GymNaxEnv( # not working
#     env_name="CartPole-v1",
#     repeat_times=5,
# )

algorithm = NEAT(
    pop_size=50,
    species_size=10,
    survival_threshold=0.1,
    compatibility_threshold=1.0,
    genome=DefaultGenome(
        num_inputs=4,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
        init_hidden_layers=(),
        node_gene=BiasNode(
            activation_options=ACT.tanh,
            aggregation_options=AGG.sum,
        ),
        output_transform=ACT.tanh,
    )
)

pipeline = Pipeline(
    algorithm=algorithm,
    problem=problem,
    seed=42,
    generation_limit=2,
    fitness_target=1000,
)

# Initialize state
state = pipeline.setup()

# Run until termination
state, best = pipeline.auto_run(state)

# show result
pipeline.show(state, best, output_type="gif")

# print(state.__class__) # tensorneat.common.state.State
# print(state.registered_keys()) # dict_keys(['randkey', 'pop_nodes', 'pop_conns', 'generation', 'species'])
# print(state.species.registered_keys()) # 'species_keys', 'best_fitness', 'last_improved', 'member_count', 'idx2species', 'center_nodes', 'center_conns', 'next_species_key'
# print(best)  # tuple

# visualize the best individual
# genome = pipeline.algorithm.genome
# network = genome.network_dict(None, *best)  # Transform the network from JAX arrays to a Python dict
# genome.visualize(network, save_path="./network.png",with_labels=True)
# print(algorithm.genome.repr(state, *best))
# print(*best)

# randkey_, randkey = jax.random.split(state.randkey)
# state = state.update(randkey=randkey)
# problem.show(
#     state,
#     randkey_,
#     algorithm.forward,
#     params
# )
