import argparse

from sponet_cv_testing.testpipeline import generate_network, get_runfiles
from sponet_cv_testing.resultmanagement import save_network


parser = argparse.ArgumentParser()

parser.add_argument("runfile_path",
                    help="Path to the runfile.")
parser.add_argument("save_path",
                    help="Path to the directory where the network will be saved.")
parser.add_argument("name",
                    help="Name under which the network will be saved.")

def main():

    args = parser.parse_args()

    runfile_path = args.runfile_path
    save_path = args.save_path
    name = args.name

    run_parameters = get_runfiles(runfile_path)[0]
    network = generate_network(run_parameters["network"])

    save_network(network, save_path, name)

    return


if __name__ == "__main__":
    main()


