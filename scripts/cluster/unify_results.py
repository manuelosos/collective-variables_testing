import argparse
import numpy as np
import os
import re

parser = argparse.ArgumentParser()

parser.add_argument("path", help="Path to the result directory.")

def main():


    args = parser.parse_args()

    path = args.path

    rel_path = "../../test_space/tmp_results/CNVM2_ab2_n1000_r100-100_rt001-001_l200_20ts_a2000_s1000_ts0/"

    names_tm = ["diffusion_maps_eigenvalues",
                "transition_manifolds"]

    names_cv = ["cv", "cv_coefficients",
             "cv_coefficients_weighted",
             "cv_samples",
             "cv_samples_weighted",
             "cv_weighted"]

    for name in names_tm:
        res = unify_results(path + "transition_manifolds/", name)

        np.save(f"{path}transition_manifolds/{name}", res)



    for name in names_cv:
        res = unify_results(path + "collective_variables/", name)
        np.save(f"{path}collective_variables/{name}", res)

    return


def unify_results(path, name):

    pattern = re.compile(f"{name}_*\d.npy$")

    target = []
    file_list = os.listdir(path)
    for file in file_list:
        if pattern.match(file):
            target.append(file)

    target = sorted(target)

    res = []
    for file in target:

        res.append(np.load(f"{path}{file}", allow_pickle=True))

    return np.concatenate(res)






if __name__ == "__main__":
    main()