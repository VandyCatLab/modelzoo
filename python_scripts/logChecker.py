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
        with open(log, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]

        # Go next if not finished.
        if not 'final validation acc' in lines[-1]:
            next

        # Find seeds
        snapshotLine = [line for line in lines if 'Snapshot' in line][0]
        snapshotLine = snapshotLine.split()
        weightIndex = snapshotLine.index('weight') + 1
        shuffleIndex = snapshotLine.index('shuffle') + 1
        params = (int(snapshotLine[weightIndex]), int(snapshotLine[shuffleIndex]))

        # Pop from list
        missing.pop(missing.index(params))

        # Remember it
        if 'Saving' in lines[-1]:
            success += log
        else:
            failiure += log

    
    return success, failiure, missing


if __name__ == '__main__':
    param_list = param_maker(range(10), range(10))
    success, failiure, missing = log_checker('../logs/master/', 'train_slurm_*.out', param_list)