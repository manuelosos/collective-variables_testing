import argparse
from sponet_cv_testing.testpipeline import get_runfiles

parser = argparse.ArgumentParser()

parser.add_argument("runfile_path",
                    help="Path to the runfile.")
parser.add_argument("result_path",
                    help="Path to the directory where the results will be saved.")
parser.add_argument("network_path",
                    help="Path to the directory where the networks are located.")
parser.add_argument("save_path",
                    help="path of the created file.")


def main():

    args = parser.parse_args()

    runfile_path = args.runfile_path
    save_path = args.save_path
    result_path = args.result_path
    network_path = args.network_path


    run_parameters = get_runfiles(runfile_path)[0]
    num_time_steps = run_parameters["simulation"]["sampling"]["num_timesteps"]


    with open(save_path, "w") as file:

        for i in range(num_time_steps+1):
            line = f"{i} {runfile_path} {result_path} {network_path}\n"
            file.write(line)

    return


if __name__ == "__main__":
    main()