import numpy as np
import seaborn as sns
import sponet_cv_testing.datamanagement as dm

file_path = "../data/results/CNVM2_ab2_n500_r098-100_rt001-002_l400_a1000_s150/"
xi = np.load(file_path + "transition_manifold.npy")
x_anchor = np.load(file_path + "x_data.npz")["x_anchor"]
#params = load_params(file_path + "params.pkl")
network = dm.open_network(file_path, "network")

print(xi.shape)