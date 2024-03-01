import json
import datetime
from itertools import product
import datamanagement as dm
import pandas as pd
import numpy as np
from numpy import isclose


def _change_run(run: dict, equiv_run: pd.Series, type: str) -> dict:

    raise NotImplementedError


def _rates_multiple(row: dict, r_ab, r_ba, rt_ab, rt_ba, lag_time) -> bool:
    """
    Checks if the rates and lag time specified in row are a multiple of the given rates.
    """
    r_ab_ratio = row["r_ab"] / r_ab
    r_ba_ration = row["r_ba"] / r_ba
    rt_ab_ratio = row["rt_ab"] / rt_ab
    rt_ba_ratio = row["rt_ba"] / rt_ba
    lag_time_ratio = row["lag_time"] / lag_time

    if (isclose(r_ab_ratio, r_ba_ration) and
            isclose(r_ab_ratio, rt_ab_ratio) and
            isclose(r_ab_ratio, rt_ba_ratio) and
            isclose(r_ab_ratio, lag_time_ratio)):
        return True
    return False


def _create_dummy_entry(parameters) -> list:
    run_id: str = str(parameters["run_id"])

    dynamic_parameters: dict = parameters["dynamic"]
    dynamic_model: str = dynamic_parameters["model"]
    dynamic_rates: tuple = dm._get_run_rates(dynamic_parameters["rates"])

    network_parameters: dict = parameters["network"]
    network_model = network_parameters["model"]
    num_nodes = network_parameters["num_nodes"]

    sampling_parameters: dict = parameters["simulation"]["sampling"]
    lag_time: float = sampling_parameters["lag_time"]
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    num_coordinates: int = parameters["simulation"]["num_coordinates"]

    new_result: list = [
        dynamic_model, *dynamic_rates, "", network_model, num_nodes, lag_time, num_anchor_points,
        num_samples_per_anchor, num_coordinates, "", ""]
    return new_result


def run_reasonable(df, run: dict, change_run: bool = False) -> bool:
    """
    Checks if a run has been done with mathematically equivalent parameters.
    Parameters
    ----------
    run
        Parameters of the run
    change_run
    Returns
    -------
    bool checks
    """


    run_id = run["run_id"]
    r_ab, r_ba, rt_ab, rt_ba = dm._get_run_rates(run["dynamic"]["rates"])
    lag_time = run["simulation"]["sampling"]["lag_time"]
    df = df[(df["dynamic_model"] == run["dynamic"]["model"])
            & (df["network_model"] == run["network"]["model"])
            & (df["num_nodes"] == run["network"]["num_nodes"])]

    for index, row in df.iterrows():

        equivalency: str = "none"

        # swapped ab and ba rates
        if (isclose(row["r_ab"], r_ba) and
            isclose(row["r_ba"], r_ab) and
                isclose(row["rt_ab"], rt_ba) and
                isclose(row["rt_ba"], rt_ab) and
                isclose(row["lag_time"], lag_time)):
            print(f"Run {run_id}: rates are swapped with run: {index}")
            equivalency = "swap"

        # rates and lag_time multiple of another
        if _rates_multiple(row, r_ab, r_ba, rt_ab, rt_ba, lag_time):
            print(f"Run {run_id}: Rates are multiple of another run: {index}")
            equivalency = "multiple"

        # both cases above combined
        if _rates_multiple(row, r_ba, r_ab, rt_ba, rt_ab, lag_time):
            print(f"Run {run_id}: Rates are swapped multiple of another run: {index}")
            equivalency = "swap multiple"

        if equivalency != "none":
            if change_run:
                run = _change_run(run, row, equivalency)
                return run
            return False
    return True


def _fstr(x: float) -> str:
    """Converts float to str without '.' with min 3 digits."""
    res = str(x).replace(".", "")
    while len(res) < 3:
        res += "0"
    return res


def create_runfiles(path: str, save=True) -> list[dict]:

    dynamic = "CNVM"
    num_states = 2
    r_ab_l = [1, 2, 3, 4, 5]
    r_ba_l = [1]

    r_tilde_ab_l = [0.01, 0.02, 0.03]
    r_tilde_ba_l = [0.01, 0.02, 0.03]

    network_model = "albert-barabasi"
    num_nodes = 500
    num_attachments = 2

    lag_time_l = [1, 2, 3, 4, 5]
    num_anchor_points_l = [1000]
    num_samples_per_anchor_l = [150]

    df = dm.read_data_csv()

    #TODO checken ob eine der Raten größer als 10 ist. In diesem Fall ist die ID nicht mehr eindeutig

    runfiles = []
    for r_ab, r_ba, r_tilde_ab, r_tilde_ba, lag_time, num_anchor_points, num_samples_per_anchor in (
            product(r_ab_l, r_ba_l, r_tilde_ab_l, r_tilde_ba_l, lag_time_l, num_anchor_points_l, num_samples_per_anchor_l)):

        network_abbr: dict = {"albert-barabasi": "ab"}

        run_id = (f"{dynamic}{num_states}_"
                  f"{network_abbr[network_model]}{num_attachments}_n{num_nodes}_"
                  f"r{_fstr(r_ab)}-{_fstr(r_ba)}_rt{_fstr(r_tilde_ab)}-{_fstr(r_tilde_ba)}"
                  f"_l{_fstr(lag_time)}_a{num_anchor_points}_s{num_samples_per_anchor}")

        if run_id in df.index:
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

        if not run_reasonable(df, run, change_run=False):
            print("No file produced")
            continue

        df.loc[run_id] = _create_dummy_entry(run)

        if save:
            with open(f"{path}{run_id}.json", "w") as file:
                json.dump(run, file, indent=3)

        runfiles.append(run)

    print(f"{len(runfiles)} files produced")

    return runfiles


def make_cluster_jobarray(path: str, runfiles: list[dict]):

    with open(f"{path}param_array.txt", "a") as file:
        for run in runfiles:
            run_id = run["run_id"]
            file.write(f"runfiles/{run_id}.json results\n")
    return


if __name__ == "__main__":
    files = create_runfiles("../tests/cluster_runfiles/", save=True)
    make_cluster_jobarray("../tests/cluster_runfiles/", files)
