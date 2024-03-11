import json
from itertools import product
import sponet_cv_testing.datamanagement as dm
import pandas as pd
from numpy import isclose
import logging
import numpy as np
import datetime

logger = logging.getLogger("runfile_script")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
compact_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(compact_formatter)
logger.addHandler(console_handler)

network_abbr: dict = {"albert-barabasi": "ab"}


def _change_run(run: dict, equiv_run: pd.Series, type: str) -> dict:

    raise NotImplementedError


def _determine_rerun_number(rerun_df: pd.DataFrame) -> int:
    index = rerun_df.index
    numbered_index = index[index.str.match(".*r\d\d$")]
    if len(numbered_index) == 0:
        return 1

    return max([int(item) for item in numbered_index.str[-2:]]) + 1


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


def run_valid(
        df: pd.DataFrame,
        run: dict,
        allow_reruns: bool = False,
        allow_failed_reruns: bool = False,
        change_run: bool = False
) -> tuple[bool, dict | None]:
    """
    Checks if a run has been done with mathematically equivalent parameters.
    Parameters
    ----------
    df : DataFrame
        The dataframe of the result csv
    run
        Parameters of the run
    allow_reruns: (bool)
        If set to True reruns will be allowed. The ID will be appended with a corresponding number
        i.e. id = id+f"{number_of_rerun}". The first run has no number and is implicitly numbered with 0.
        Breaks if more than 100 runs are made.
    allow_failed_reruns: (bool)
        Set to True if a run should be repeated even if it has previously failed.
    change_run: (bool)
        If set to True the run will if a mathematically equivalent run is found. In the latter case the rates of the
        proposed run are changed to match those of the already existing one.
        Only relevant if reruns are allowed. Since this will result in a rerun.
    Returns
    -------
    bool True if the run is valid
    """

    run_id = run["run_id"]
    r_ab, r_ba, rt_ab, rt_ba = dm._get_run_rates(run["dynamic"]["rates"])
    lag_time = run["simulation"]["sampling"]["lag_time"]
    df = df[(df["dynamic_model"] == run["dynamic"]["model"])
            & (df["network_model"] == run["network"]["model"])
            & (df["num_nodes"] == run["network"]["num_nodes"])]

    for index, row in df.iterrows():

        equivalency: str | None = None

        # run already existing
        rerun_df = df[df.index.str.match("^" + run_id + ".*$") == True]
        if len(rerun_df) > 0:
            logger.debug(f"{run_id} already in the dataframe")
            equivalency = "rerun"

        # swapped ab and ba rates
        if (isclose(row["r_ab"], r_ba) and
            isclose(row["r_ba"], r_ab) and
                isclose(row["rt_ab"], rt_ba) and
                isclose(row["rt_ba"], rt_ab) and
                isclose(row["lag_time"], lag_time)):
            logger.debug(f"Run {run_id}: rates are swapped with run: {index}")
            equivalency = "swap"

        # rates and lag_time multiple of another
        if _rates_multiple(row, r_ab, r_ba, rt_ab, rt_ba, lag_time):
            logger.debug(f"Run {run_id}: Rates are multiple of another run: {index}")
            equivalency = "multiple"

        # both cases above combined
        if _rates_multiple(row, r_ba, r_ab, rt_ba, rt_ab, lag_time):
            logger.debug(f"Run {run_id}: Rates are swapped multiple of another run: {index}")
            equivalency = "swap multiple"

        if equivalency is None:
            return True, run

        if equivalency == "rerun":
            if not allow_reruns:
                logger.info("Reruns not allowed")
                return False, None

            if not rerun_df["finished"].all() and not allow_failed_reruns:
                logger.info("Previous run with same parameters has failed and no reruns of failed runs allowed.")
                return False, None

            # first run has no appended number and is thus implicitly numbered with 0
            run_number_str = str(_determine_rerun_number(rerun_df)).rjust(2, "0")
            run["run_id"] += f"_r{run_number_str}"
            return True, run

        if equivalency is not None:
            if change_run:
                raise NotImplementedError
                run = _change_run(run, row, equivalency)
                return True, run
            return False, None
    return True, run


def _create_dummy_entry(parameters) -> list:

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
        num_samples_per_anchor, num_coordinates, "", True]
    # Dummy entries always have "finished" set to True to avoid conflicts while generating multiple reruns
    return new_result


def _fstr(x: float) -> str:
    """Converts float to str without '.' with min 3 digits."""
    res = str(x).replace(".", "")
    while len(res) < 3:
        res += "0"
    return res


def create_runfiles(
        allow_reruns: bool = False,
        allow_failed_reruns = False,
        change_run: bool = False
) -> list[dict]:

    dynamic = "CNVM"
    num_states = 2
    r_ab_l = [0.99, 1, 1.01]
    r_ba_l = [1]

    r_tilde_ab_l = [0.01, 0.02, 0.03]
    r_tilde_ba_l = [0.02, 0.03, 0.04]

    network_model = "albert-barabasi"
    num_nodes = 500
    num_attachments = 2

    lag_time_l = [4.8, 4.9, 5, 5.1, 5.2]
    #np.arange(4.5, 5.6, 0.1)
    num_anchor_points_l = [1000]
    num_samples_per_anchor_l = [150]

    num_runs_per_set: list[int] = list(range(4))

    df = dm.read_data_csv()

    # TODO checken ob eine der Raten größer als 10 ist. In diesem Fall ist die ID nicht mehr eindeutig

    runfiles = []
    for r_ab, r_ba, r_tilde_ab, r_tilde_ba, lag_time, num_anchor_points, num_samples_per_anchor, _ in \
            (
            product(r_ab_l, r_ba_l, r_tilde_ab_l, r_tilde_ba_l,
                    lag_time_l, num_anchor_points_l, num_samples_per_anchor_l, num_runs_per_set)
            ):


        run_id = (f"{dynamic}{num_states}_"
                  f"{network_abbr[network_model]}{num_attachments}_n{num_nodes}_"
                  f"r{_fstr(r_ab)}-{_fstr(r_ba)}_rt{_fstr(r_tilde_ab)}-{_fstr(r_tilde_ba)}"
                  f"_l{_fstr(lag_time)}_a{num_anchor_points}_s{num_samples_per_anchor}")

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

        valid, tmp_run = run_valid(df, run, allow_reruns, allow_failed_reruns, change_run)
        if not valid:
            logger.info(f"no file produced of run {run['run_id']}")
            continue
        if change_run:
            run = tmp_run

        print(run["run_id"])

        # In order to compare with runs created in the same batch,
        # every run will be added as a dummy entry which will not be saved.
        df.loc[run["run_id"]] = _create_dummy_entry(run)

        runfiles.append(run)

    print(f"{len(runfiles)} files produced")

    return runfiles


def save_runfiles(save_path: str, runfiles: dict | list[dict]) -> None:
    """Saves the runfiles to json files in the specified path."""
    if isinstance(runfiles, dict):
        runfiles = [runfiles]

    if not save_path.endswith("/"):
        save_path += "/"

    for runfile in runfiles:
        with open(f"{save_path}{runfile['run_id']}.json", "w") as file:
            json.dump(runfile, file, indent=3)
    return


def get_runfiles(run_ids: str | list[str], data_path: str = "data/results/") -> list[dict]:
    """Fetches the runfile from one or multiple runs by their run_id."""
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    runs: list[dict] = []
    for run_id in run_ids:
        with open(f"{data_path}{run_id}/parameters.json", "r") as file:
            run: dict = json.load(file)
        runs.append(run)
    return runs


def get_unfinished_runfiles() -> list[dict]:
    data = dm.read_data_csv()
    unfinished_runs = data[data["finished"] == False]
    return get_runfiles(unfinished_runs.index)


def get_timeout_runs() -> list[dict]:
    runs = get_unfinished_runfiles()
    timeout_runs = []
    data_path: str = "data/results/"
    for run in runs:
        with open(f"{data_path}{run['run_id']}/runlog.log", "r") as file:
            last_log = file.readlines()[-1]
        if "ARPACK" in last_log:
            continue
        timeout_runs.append(run)
    return timeout_runs


def make_cluster_jobarray(path: str, runfiles: list[dict]) -> None:
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d")
    with open(f"{path}{timestamp}_param_array.txt", "a") as file:
        for run in runfiles:
            run_id = run["run_id"]
            file.write(f"runfiles/{run_id}.json results\n")
    return


if __name__ == "__main__":
    """files = create_runfiles(
        allow_reruns=True,
        allow_failed_reruns=True,
        change_run=False)"""
    # save_runfiles("tests/cluster_runfiles/", files)
    # make_cluster_jobarray("tests/cluster_runfiles/", files)
    res = get_timeout_runs()
    print(len(res))
    save_runfiles("tests/cluster_runfiles/", res)
    make_cluster_jobarray("tests/cluster_runfiles/", res)





