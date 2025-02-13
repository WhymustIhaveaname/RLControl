#! /usr/bin/env python3

"""
Github: https://github.com/EMI-Group/tensorneat
Paper: https://arxiv.org/pdf/2404.01817
"""

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv
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

algorithm = NEAT(
    pop_size=50,
    species_size=10,
    survival_threshold=0.1,
    compatibility_threshold=1.0,
    genome=DefaultGenome(
        num_inputs=4,
        num_outputs=1,
        init_hidden_layers=(),
        node_gene=BiasNode(
            activation_options=ACT.tanh,
            aggregation_options=AGG.sum,
        ),
        output_transform=ACT.tanh,
    ),
)

pipeline = Pipeline(
    algorithm=algorithm,
    problem=problem,
    seed=42,
    generation_limit=100,
    fitness_target=1000,
)

# Initialize state
state = pipeline.setup()

# Run until termination
state, best = pipeline.auto_run(state)
