import glob
import os
import argparse


def param_maker(weight_list, shuffle_list):
    """
    Return a list of tuples where each inner list has the weight and shuffle
    seed, given the list of weight and shuffle seeds fully crossed.
    """
    # Preallocate
    params = [(-1, -1)] * len(weight_list) * len(shuffle_list)
    paramComboIdx = 0

    # Loop through lists
    for s in shuffle_list:
        for w in weight_list:
            params[paramComboIdx] = (w, s)
            paramComboIdx += 1

    return params


def log_checker(directory, file_str, param_list):
    """
    Return success, failure, and missing seed combinations given a directory of
    logs and a file_str to search for. Uses param_list as a checklist.
    """
    logList = glob.glob(os.path.join(directory, file_str))

    success = []
    failiure = []
    missing = param_list.copy()
    # Loop through log matched list
    for log in logList:
        with open(log, "r") as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]

        # Go next if not finished.
        if not "final validation acc" in lines[-1]:
            continue

        # Find seeds
        snapshotLine = [line for line in lines if "Snapshot" in line][0]
        snapshotLine = snapshotLine.split()
        weightIndex = snapshotLine.index("weight") + 1
        shuffleIndex = snapshotLine.index("shuffle") + 1
        params = (
            int(snapshotLine[weightIndex]),
            int(snapshotLine[shuffleIndex]),
        )

        # Pop from list
        missing.pop(missing.index(params))

        # Remember it
        if "Saving" in lines[-1]:
            success += [log]
        else:
            failiure += [log]

    return success, failiure, missing


if __name__ == "__main__":
    # Deal with arguments
    parser = argparse.ArgumentParser(
        description="Checks through the logs to find if they have completed."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        help="directory where the logs are",
        required=True,
    )
    parser.add_argument(
        "--file_str",
        "-f",
        type=str,
        help="string to match log files with",
        required=True,
    )
    parser.add_argument(
        "--shuffle_seeds", help="list of seeds for shuffle, e.g., 2,4,6"
    )
    parser.add_argument(
        "--weight_seeds", help="list of seeds for weights, e.g., 2,4,6"
    )
    parser.add_argument("--shuffle_min", type=int, help="minimum shuffle seed")
    parser.add_argument(
        "--shuffle_max", type=int, help="maximum shuffle seed, not inclusive"
    )
    parser.add_argument("--weight_min", type=int, help="mininmum weight seed")
    parser.add_argument(
        "--weight_max", type=int, help="maximum weight seed, not inclusive"
    )

    args = parser.parse_args()

    if not args.shuffle_seeds in None and not args.weight_seeds is None:
        shuffleList = [seed for seed in args.shuffle_seeds.split()]
        weightList = [seed for seed in args.weight_seeds.split()]
    elif (
        args.weight_min is not None
        and args.weight_max is not None
        and args.shuffle_min is not None
        and args.shuffle_max is not None
    ):
        shuffleList = range(args.shuffle_min, args.shuffle_max)
        weightList = range(args.weight_min, args.weight_max)
    else:
        raise ValueError("incomplete seed arguments")

    param_list = param_maker(weightList, shuffleList)
    success, failiure, missing = log_checker(
        "../logs/master/", "train_slurm_*.out", param_list
    )

    print("Successes")
    print("\n".join(success))
    print("Failiure")
    print("\n".join(failiure))
    print("Missing")
    print(missing)
