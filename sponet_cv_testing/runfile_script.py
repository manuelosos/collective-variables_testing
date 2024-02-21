import json
import datetime
from itertools import product
import datamanagement as dm
import numpy as np


def fstr(x: float) -> str:
    """Converts float to str without '.' with min 3 digits."""
    res = str(x).replace(".", "")
    while len(res) < 3:
        res += "0"
    return res


def create_runfiles(path: str, save=True) -> list[dict]:

    dynamic = "CNVM"
    num_states = 2
    r_ab_l = [0.98, 1, 1.02, 1.04]
    r_ba_l = [1]

    r_tilde_ab_l = [0.01, 0.02, 0.03]
    r_tilde_ba_l = [0.02]

    network_model = "albert-barabasi"
    num_nodes = 500
    num_attachments = 2

    lag_time_l = [4]
    num_anchor_points_l = [1000]
    num_samples_per_anchor_l = [150]

    #TODO checken ob eine der Raten größer als 10 ist. In diesem Fall ist die ID nicht mehr eindeutig

    runfiles = []
    for r_ab, r_ba, r_tilde_ab, r_tilde_ba, lag_time, num_anchor_points, num_samples_per_anchor in (
            product(r_ab_l, r_ba_l, r_tilde_ab_l, r_tilde_ba_l, lag_time_l, num_anchor_points_l, num_samples_per_anchor_l)):

        network_abbr: dict = {"albert-barabasi": "ab"}

        run_id = (f"{dynamic}{num_states}_"
                  f"{network_abbr[network_model]}{num_attachments}_n{num_nodes}_"
                  f"r{fstr(r_ab)}-{fstr(r_ba)}_rt{fstr(r_tilde_ab)}-{fstr(r_tilde_ba)}"
                  f"_l{fstr(lag_time)}_a{num_anchor_points}_s{num_samples_per_anchor}")

        if not dm.unique_run_id(run_id):
            print(f"{run_id} already in the dataframe")
            print("No File produced\n")
            continue
        print(run_id)

        run = {
            "run_id": run_id,
            "dynamic": {
                "model": dynamic,
                "num_states": num_states,
                "rates": {
                    "r": [[0, r_ab],
                          [r_ba, 0]],
                    "r_tilde": [[0, r_tilde_ab],
                                [r_tilde_ba, 0]]
                }

            },
            "network": {
                "generate_new": True,
                "model": "albert-barabasi",
                "num_nodes": num_nodes,
                "num_attachments": num_attachments
            },
            "simulation": {
                "sampling": {
                    "method": "local_cluster",
                    "lag_time": lag_time,
                    "num_anchor_points": num_anchor_points,
                    "num_samples_per_anchor": num_samples_per_anchor
                },
                "num_coordinates": 10
            }
        }

        if save:
            with open(f"{path}{run_id}.json", "w") as file:
                json.dump(run, file, indent=3)

        runfiles.append(run)

    return runfiles


def make_cluster_jobarray(path:str, runfiles: list[dict]):

    with open(f"{path}param_array.txt", "a") as file:
        for run in runfiles:
            run_id = run["run_id"]
            file.write(f"runfiles/{run_id}.json results\n")


if __name__ == "__main__":
    files = create_runfiles("../tests", save=False)
    make_cluster_jobarray("../tests", files)