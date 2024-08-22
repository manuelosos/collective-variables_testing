import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from sponet import CNVMParameters, CNVM
from sponet.collective_variables import OpinionShares
from sponet import sample_many_runs, calc_rre_traj

num_opinions = 3  # opinion 1 represents 'S', opinion 2 represents 'I'
num_agents = 1000
infection_rate = 0.5

r = np.array(
    [[0, 0.8, 0.2],
     [0.2, 0, 0.8],
     [0.8, 0.2, 0]]
)
r_tilde = np.array(
    [[0, 0.01, 0.01],
     [0.01, 0, 0.01],
     [0.01, 0.01, 0]]
)

network = nx.erdos_renyi_graph(num_agents, p=0.01)

params = CNVMParameters(
    num_opinions=num_opinions,
    network=network,
    r=r,
    r_tilde=r_tilde,
)

model = CNVM(params)

cv = OpinionShares(num_opinions, normalize=True)  # for measuring the percentage of infectious nodes

t_max = 200

x_init = np.concatenate([0 * np.ones(int(20 * num_agents / 100)),
                         1 * np.ones(int(50 * num_agents / 100)),
                         2 * np.ones(int(30 * num_agents / 100))])

opinion_shares = OpinionShares(num_opinions, normalize=True)


initial_states = np.array([x_init])
t, c = sample_many_runs(params,
                        initial_states,
                        t_max,
                        num_timesteps=200,
                        num_runs=200,
                        collective_variable=opinion_shares,
                        n_jobs=-1)
print(t.shape)
print(c.shape)


for i in range(c.shape[3]):
    plt.plot(t, np.mean(c[0, :, :, i], axis=0), label=f"$c_{i+1}$")
plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("counts")
plt.show()