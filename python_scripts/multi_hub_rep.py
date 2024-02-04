import datasets
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import os
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import hubReps
import csv
import math
from itertools import chain
import pandas as pd
from scipy.spatial.distance import cdist
import utilities as utils

# from csv_data_maker import make_csv_normal


# get basics working, threshhld, does it work for multiple models
# look into new tasks when finished
# standalone, the 3 tasks
# standalone repository, archive link to paper in review, describes the 3 tasks
# not doing the matching task anymore for humans, hard for human data


# truncated, just use logical indexing to find all zeros and make them 0 again
def apply_noise_nonzero(array, noise_factor=0.01):

    non_zero_count = np.count_nonzero(array)
    noise = np.random.normal(
        loc=0, scale=noise_factor * non_zero_count, size=array.shape
    )
    noisy_array = array + noise

    return noisy_array


def apply_noise_size(array, noise_factor=0.01):

    array_size = np.size(array)
    noise = np.random.normal(loc=0, scale=noise_factor * array_size, size=array.shape)
    noisy_array = array + noise

    return noisy_array


def apply_uniform_noise(array, noise_level=0.01):

    noise = np.random.normal(loc=0, scale=noise_level, size=array.shape)
    noisy_array = array + noise

    return noisy_array


# fix this
def apply_noise_three_std(rep_arr, paths, fixed_factor=0.5):

    arrays = [np.load(path) for path in paths]
    combined_arrays = np.concatenate(arrays)
    std_dev = np.std(combined_arrays)
    zero_indices = np.where(rep_arr == 0)
    noise = fixed_factor * std_dev * np.random.randn(*rep_arr.shape)
    rep_arr_with_noise = rep_arr + noise
    rep_arr_with_noise[zero_indices] = 0

    return rep_arr_with_noise


def apply_std_noise(array, scale, include_zeros=True, relu=True):
    if not include_zeros:
        # Find all non-zero elements then calculate std of every element
        non_zero = array[array != 0]
        std = np.std(non_zero)
    else:
        std = np.std(array)

    # Add noise
    noise = np.random.normal(loc=0, scale=std * scale, size=array.shape)
    noisy_array = array + noise

    # Apply relu
    if relu:
        noisy_array[noisy_array < 0] = 0

    return noisy_array, std


def image_list(data_dir):
    img_files = os.listdir(data_dir)
    img_files.sort()
    image_names = []
    for file in img_files:
        image_names.append(file)
    return image_names


def rep_maker(data_dir, modelFile, model_name, modelData, model, batch_size):
    if "origin" in modelData.keys() and modelData["origin"] == "keras":
        # Override model input shape for keras models if input_shape isn't none
        inputShape = model.input_shape[1:]
        if not any([dim is None for dim in inputShape]):
            modelData["shape"] = model.input_shape[1:]

    # get dataset from datasets.py
    dataset = get_dataset(data_dir, modelFile, model_name, modelData, model, batch_size)
    # print('\nDataset size:', dataset, '\n')

    # get reps from hubReps.py
    # could have a pointer error with iterations.
    return get_reps(modelFile, model, dataset, modelData, batch_size)


def learning_exemplar(
    model_name, image_names, reps, csv_file, noise=0.0, memory_decay=0.0
):
    # If noise is not 0, apply noise to the representations
    if noise != 0:
        reps, _ = apply_std_noise(reps, noise, include_zeros=False, relu=True)

    # Get the target representations (they're just the nz* ones)
    targetIdxs = [
        i for i, name in enumerate(image_names) if name.split("_")[0] == "nz1"
    ]
    targetReps = reps[targetIdxs, :]

    # Load csv
    trials = pd.read_csv(csv_file)

    # Loop through each row
    results = pd.DataFrame(
        columns=["ModelName", "Response", "Corr"] + list(trials.columns)
    )
    for index, row in trials.iterrows():
        # Get trial number
        trial = row["Trial"]

        # Get the index of the choices in this trial and their reps
        idxs = [
            i for i, name in enumerate(image_names) if name.split("_")[0] == str(trial)
        ]
        choiceReps = reps[idxs]

        dists = cdist(choiceReps, targetReps, "euclidean")
        chosenIdx, _ = np.unravel_index(np.argmin(dists), dists.shape)
        response = chosenIdx + 1

        # Copy row and add new info to it
        newRow = row.copy()
        newRow["ModelName"] = model_name
        newRow["Response"] = response
        newRow["Corr"] = int(response == newRow["CorrRes"])

        # Save new row to results
        results = pd.concat([results, pd.DataFrame(newRow).T])

    # Print model accuracy
    print(f"Model: {model_name}, Accuracy: {results['Corr'].mean()}")

    return results


def many_oddball(
    model_name, image_names, reps, csv_file, noise=0.0, encoding_noise=0.0
):
    # If noise is not 0, apply noise to the representations
    if encoding_noise != 0.0:
        repStd = np.std(reps[reps != 0])

    if noise != 0.0:
        reps, repStd = apply_std_noise(reps, noise, include_zeros=False, relu=True)

    # Load csv
    trials = pd.read_csv(csv_file)

    # Loop through each row
    results = pd.DataFrame(
        columns=["ModelName", "Response", "Corr"] + list(trials.columns)
    )
    choices = [0, 1, 2]
    for index, row in trials.iterrows():
        # Cast Feedback row to int
        row["Feedback"] = int(row["Feedback"])

        # Get trial number
        trial = row["Trial"]

        idxs = [
            i
            for i, name in enumerate(image_names)
            if name.split("-")[0].replace("trial", "") == str(trial)
        ]
        choiceReps = reps[idxs]

        dists = cdist(choiceReps, choiceReps, "euclidean")
        np.fill_diagonal(dists, np.inf)

        # Find the smallest distance
        chosenIdx = np.unravel_index(np.argmin(dists), dists.shape)

        # Find the response that isn't in the choices
        response = list(set(choices) - set(chosenIdx))[0] + 1

        # Copy row and add new info to it
        newRow = row.copy()
        newRow["ModelName"] = model_name
        newRow["Response"] = response
        newRow["Corr"] = int(response == newRow["CorrRes"])

        # Save new row to results
        results = pd.concat([results, pd.DataFrame(newRow).T])

    # Print model accuracy
    print(f"Model: {model_name}, Accuracy: {results['Corr'].mean()}")

    return results


def three_afc(model_name, image_names, reps, csv_file, noise=0.0, encoding_noise=0.0):
    # Calculate std noise for encoding noise
    if encoding_noise != 0.0:
        repStd = np.std(reps[reps != 0])

    # If noise is not 0, apply noise to the representations
    if noise != 0.0:
        reps, repStd = apply_std_noise(reps, noise, include_zeros=False, relu=True)

    # Load csv
    trials = pd.read_csv(csv_file)

    # Loop through each row
    results = pd.DataFrame(
        columns=["ModelName", "Response", "Corr"] + list(trials.columns)
    )
    for index, row in trials.iterrows():
        # Cast Feedback row to int
        row["Feedback"] = int(row["Feedback"])

        # Get trial number
        trial = row["Trial"]

        # Get the index of the target and its reps
        targetIdxs = [
            i
            for i, name in enumerate(image_names)
            if name.split("-")[0] == f"trial{trial}" and "target" in name
        ]
        targetRep = reps[targetIdxs]

        # Get the choice indices and their reps
        choiceIdxs = [
            i
            for i, name in enumerate(image_names)
            if name.split("-")[0] == f"trial{trial}" and "target" not in name
        ]
        choiceReps = reps[choiceIdxs]

        # Calculate distance and find the choice with smallest distance
        dists = cdist(choiceReps, targetRep, "euclidean")
        response = np.argmin(dists) + 1

        # Copy row and add new info to it
        newRow = row.copy()
        newRow["ModelName"] = model_name
        newRow["Response"] = response
        newRow["Corr"] = int(response == newRow["CorrRes"])

        # Save new row to results
        results = pd.concat([results, pd.DataFrame(newRow).T])

    # Print model accuracy
    print(f"Model: {model_name}, Accuracy: {results['Corr'].mean()}")

    return results


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def LBA(dist, c=1, ß=None, b=1, A=1, k=0, t0=0):

    n = math.exp((-c * dist))
    if ß is None:
        ß = n
    v = n / (n + ß)
    diff_score = 1 - v
    # a = random.randrange(0, A)
    # k = b - A
    drift_rate_same = np.random.normal(v, 1)
    drift_rate_diff = np.random.normal(diff_score, 1)

    rt_same = ((b - k) / drift_rate_same) + t0
    rt_diff = ((b - k) / drift_rate_diff) + t0

    if rt_same < rt_diff:
        return ["same", 1, rt_same]
    else:
        return ["diff", 2, rt_diff]


def same_diff(image_names, reps, csv_file):
    # get sets of images with correct result and reps
    image_sets = image_sets_maker(csv_file, image_names, reps, 2)

    euc_correct_total = 0
    # LBA_correct_total = 0
    euc_correct_factor_total = 0
    # LBA_correct_factor_total = 0
    euc_incorrect_factor_total = 0
    # LBA_incorrect_factor_total = 0
    num_total = 0
    all_euc_comparisons = []
    # all_LBA_comparisons = []
    answer_set = []
    for comparison in image_sets:
        euc_factor = comparison[-2]  # / (comparison[-2] - 0.002)
        all_euc_comparisons.append(euc_factor)
        # all_LBA_comparisons.append(comparison[-1][1])
    normed_eucs = normalize(all_euc_comparisons, 0, 1)
    # normed_LBAs = normalize(all_euc_comparisons, 0, 1)

    idx = 0
    for euc_score in normed_eucs:
        if euc_score < 0.5:
            euc_result = ["same", 1]
        else:
            euc_result = ["diff", 2]

        if (image_sets[idx][-3] == euc_result[0]) or (
            int(image_sets[idx][-3]) == euc_result[1]
        ):
            euc_ans = "correct"
            euc_correct_total += 1

            euc_correct_factor_total += euc_score
        else:
            euc_ans = "incorrect"
            euc_incorrect_factor_total += euc_score
        """
        if (comparison[-3] == comparison[-1][0]) or (int(comparison[-3]) == comparison[-1][1]):
            LBA_ans = 'correct'  
        else:
            LBA_ans = 'not correct'
        
        print('Correct Answer:', image_sets[idx][-3], '| Given:', euc_result, f'({euc_ans})', '| Euc Factor:', euc_score)
        
        if euc_ans == 'correct':
            euc_correct_total += 1
            euc_correct_factor_total += euc_score #comparison[-2]
        else:
            euc_incorrect_factor_total += euc_score #comparison[-2]
        
        if LBA_ans == 'correct':
            LBA_correct_total += 1
            LBA_correct_factor_total += comparison[-1][1]
        else:
            LBA_incorrect_factor_total += comparison[-1][1]
        """
        answer_set.append([euc_score, image_sets[idx][-3], euc_ans])
        num_total += 1
        idx += 1

    # print('normed_eucs:\n', normed_eucs)
    # print('normed_LBAs:', normed_LBAs)
    """''
    LBA_factor_total = LBA_correct_factor_total + LBA_incorrect_factor_total
    euc_factor_total = euc_correct_factor_total + euc_incorrect_factor_total

    if euc_correct_total != 0:
        euc_correct_avg = euc_correct_factor_total / euc_correct_total
    else:
        euc_correct_avg = "No Euc was correct"
    if LBA_correct_total != 0:
        LBA_correct_avg = LBA_correct_factor_total / LBA_correct_total
    else:
        LBA_correct_avg = "No LBA was correct"
    euc_incorrect_avg = euc_incorrect_factor_total / (num_total - euc_correct_total)
    LBA_incorrect_avg = LBA_incorrect_factor_total / (num_total - LBA_correct_total)
    

    print(f'\n\n\nUsing Model: {args.model_name}\n-------------\n\
Euc Correct: {euc_correct_total} / {num_total}\n\
LBA Correct: {LBA_correct_total} / {num_total}\n\
Average Factor for correct Euc: {euc_correct_avg}\n\
Average Factor for correct LBA: {LBA_correct_avg}\n\
Average Factor for incorrect Euc: {euc_incorrect_avg}\n\
Average Factor for incorrect LBA: {LBA_incorrect_avg}\n')
    """ ""
    return [euc_correct_total, num_total, answer_set]


def image_sets_maker(csv_file, image_names, reps, num_sets):

    with open(csv_file, newline="") as c:
        csv_data = csv.reader(c)
        image_sets = []
        row_idx = 0
        for row in csv_data:
            if row_idx > 0:
                if num_sets == 2:
                    image1_name = row[5].replace(".jpg", ".tif")
                    image2_name = row[6].replace(".jpg", ".tif")
                    CorrRes = row[2]
                    image1_idx = image_names.index(image1_name)
                    image2_idx = image_names.index(image2_name)
                    euc_dist = euclidean_distance(reps[image1_idx], reps[image2_idx])
                    # LBA_results = LBA(euc_dist)
                    LBA_results = None
                    image_sets.append(
                        [
                            row_idx,
                            image1_idx,
                            image2_idx,
                            image1_name,
                            image2_name,
                            CorrRes,
                            euc_dist,
                            LBA_results,
                        ]
                    )
                elif num_sets == 3:
                    image1_name = f"trial{row_idx}-1.jpg"
                    image2_name = f"trial{row_idx}-2.jpg"
                    image3_name = f"trial{row_idx}-3.jpg"
                    target_image = f"trial{row_idx}-target.jpg"
                    image1_idx = image_names.index(image1_name)
                    image2_idx = image_names.index(image2_name)
                    image3_idx = image_names.index(image3_name)
                    target_idx = image_names.index(target_image)
                    euc_dist_1 = euclidean_distance(reps[target_idx], reps[image1_idx])
                    euc_dist_2 = euclidean_distance(reps[target_idx], reps[image2_idx])
                    euc_dist_3 = euclidean_distance(reps[target_idx], reps[image3_idx])
                    CorrRes = row[1]
                    image_sets.append(
                        [
                            row_idx,
                            image1_idx,
                            image2_idx,
                            image3_idx,
                            target_idx,
                            image1_name,
                            image2_name,
                            image3_name,
                            target_image,
                            euc_dist_1,
                            euc_dist_2,
                            euc_dist_3,
                            CorrRes,
                        ]
                    )
                elif num_sets == "odd":
                    image1_name = f"trial{row_idx}-1.jpg"
                    image2_name = f"trial{row_idx}-2.jpg"
                    image3_name = f"trial{row_idx}-3.jpg"
                    image1_idx = image_names.index(image1_name)
                    image2_idx = image_names.index(image2_name)
                    image3_idx = image_names.index(image3_name)
                    euc_dist_1_2 = euclidean_distance(
                        reps[image1_idx], reps[image2_idx]
                    )
                    euc_dist_1_3 = euclidean_distance(
                        reps[image1_idx], reps[image3_idx]
                    )
                    euc_dist_2_3 = euclidean_distance(
                        reps[image2_idx], reps[image3_idx]
                    )
                    CorrRes = row[1]
                    image_sets.append(
                        [
                            row_idx,
                            image1_idx,
                            image2_idx,
                            image3_idx,
                            image1_name,
                            image2_name,
                            image3_name,
                            euc_dist_1_2,
                            euc_dist_1_3,
                            euc_dist_2_3,
                            CorrRes,
                        ]
                    )
                elif num_sets == "le":
                    euc_dist_1 = []
                    euc_dist_2 = []
                    euc_dist_target = []
                    set6 = [
                        "nz1_4_a.tif",
                        "nz1_12_b.tif",
                        "nz1_30_a.tif",
                        "nz1_75_a.tif",
                        "nz1_70_b.tif",
                        "nz1_77_b.tif",
                    ]
                    foil1_name = row[-2] + ".tif"
                    foil2_name = row[-1] + ".tif"
                    target_name = row[2] + ".tif"
                    foil1_idx = image_names.index(foil1_name)
                    foil2_idx = image_names.index(foil2_name)
                    target_idx = image_names.index(target_name)

                    for img6 in set6:
                        img6_idx = image_names.index(img6)
                        euc_dist_target.append(
                            euclidean_distance(reps[img6_idx], reps[target_idx])
                        )
                        euc_dist_1.append(
                            euclidean_distance(reps[img6_idx], reps[foil1_idx])
                        )
                        euc_dist_2.append(
                            euclidean_distance(reps[img6_idx], reps[foil2_idx])
                        )

                    CorrRes = row[3]
                    image_sets.append(
                        [
                            row_idx,
                            target_idx,
                            foil1_idx,
                            foil2_idx,
                            target_name,
                            foil1_name,
                            foil2_name,
                            euc_dist_target,
                            euc_dist_1,
                            euc_dist_2,
                            CorrRes,
                        ]
                    )
                else:
                    raise ValueError(f"Number of sets ({num_sets}) is not available")

            row_idx += 1

    return image_sets


def euclidean_distance(x1, x2):
    temp = x1 - x2
    squared_value = np.dot(temp.T, temp)
    euc_dist = np.sqrt(squared_value)
    return euc_dist


def load_image_set(data_dir, row, row_idx):
    # img_files = os.listdir(data_dir)
    # csv_file.read_line()
    image1_name = row[5].replace(".jpg", ".tif")
    image2_name = row[6].replace(".jpg", ".tif")
    # image1 = Image.open(os.path.join(data_dir, image1_name))
    # image2 = Image.open(os.path.join(data_dir, image2_name))
    CorrRes = row[2]
    image_set = [row_idx, image1_name, image2_name, CorrRes]

    return image_set


def find_model(fileList, modelName):
    fileList = fileList.split(", ")
    for file in fileList:
        with open(file, "r") as f:
            hubModels = json.loads(f.read())
            if modelName in hubModels:
                modelFile = file
                modelData = hubModels[modelName]

    return modelFile, modelData


def get_model(modelName, modelFile, hubModels):
    if modelFile == "../data_storage/hubModel_storage/hubModels.json":
        # Creating models
        info = hubModels[modelName]
        shape = info["shape"] if "shape" in info.keys() else [224, 224, 3]
        # Create model from tfhub
        inp = tf.keras.Input(shape=shape)
        out = hub.KerasLayer(info["url"])(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
    elif modelFile == "../data_storage/hubModel_storage/hubModels_keras.json":
        # Create model from keras function
        model = hubReps.get_keras_model(hubModels, modelName)[0]
    elif (
        (modelFile == "../data_storage/hubModel_storage/hubModels_pytorch.json")
        or (modelFile == "../data_storage/hubModel_storage/hubModels_timm.json")
        or (modelFile == "../data_storage/hubModel_storage/hubModels_transformers.json")
        or (
            modelFile
            == "../data_storage/hubModel_storage/hubModels_pretrainedmodels.json"
        )
    ):
        # Create model from pytorch hub
        model = hubReps.get_pytorch_model(hubModels, modelFile, modelName)
        model.eval()
    else:
        raise ValueError(f"Unknown models file {modelFile}")

    return model


def get_dataset(data_dir, modelFile, modelName, modelData, model, batch_size=64):
    if (
        modelFile == "../data_storage/hubModel_storage/hubModels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_keras.json"
    ):
        preprocFun = datasets.preproc(
            **modelData,
            labels=False,
        )
        dataset = datasets.get_flat_dataset(data_dir, preprocFun, batch_size=batch_size)

    elif (
        modelFile == "../data_storage/hubModel_storage/hubModels_pytorch.json"
        or modelFile
        == "../data_storage/hubModel_storage/hubModels_pretrainedmodels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_timm.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_transformers.json"
    ):

        # using pytorch model

        dataset = datasets.get_pytorch_dataset(
            data_dir, modelData, model, batch_size, modelName
        )

    else:
        raise ValueError(f"Unknown models file {modelFile}")

    return dataset


def get_reps(modelFile, model, dataset, modelData, batch_size=64):

    if (
        modelFile == "../data_storage/hubModel_storage/hubModels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_keras.json"
    ):
        reps = hubReps.get_reps(model, dataset, modelData, batch_size)

        utils.clear_model()

    elif (
        modelFile == "../data_storage/hubModel_storage/hubModels_pytorch.json"
        or modelFile
        == "../data_storage/hubModel_storage/hubModels_pretrainedmodels.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_timm.json"
        or modelFile == "../data_storage/hubModel_storage/hubModels_transformers.json"
    ):

        reps = hubReps.get_pytorch_reps(model, dataset, modelData, batch_size)

    return reps


if __name__ == "__main__":
    import argparse

    print("\n\n/////////////////////////////////////")
    os.environ["TORCH_HOME"] = "/data/modelnet/torch"

    parser = argparse.ArgumentParser(
        description="Get representations from Tensorflow Hub networks using the validation set of ImageNet, intended to be used in HPC"
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help="name of the model to get representations from",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        help="index of the model to get representations from",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=64,
        help="batch size for the images passing through networks",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        help="directory of the image dataset",
        default="../novset",
    )
    parser.add_argument(
        "--comp_dir",
        "-cd",
        type=str,
        help="directory of the compared images",
        default="./data_storage/temp_images",
    )
    parser.add_argument(
        "--model_files",
        "-f",
        type=str,
        help=".json file with the hub model info",
        default="../data_storage/hubModel_storage/hubModels.json, ../data_storage/hubModel_storage/hubModels_timm.json, ../data_storage/hubModel_storage/hubModels_keras.json, ../data_storage/hubModel_storage/hubModels_pytorch.json",
    )
    parser.add_argument(
        "--feature_limit",
        "-l",
        type=int,
        help="the maximum number of features a representation can have",
        default=2048,
    )
    parser.add_argument(
        "--reps_name",
        type=str,
        help="name of the representations to save",
        default="temp_reps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of the dataset to use, use 'test' for testing a model",
        default="test",
    )
    parser.add_argument(
        "--simSet",
        type=str,
        default="all",
        help="which set of similarity functions to use",
    )
    parser.add_argument(
        "--csv_file",
        "-cf",
        type=str,
        default="./data_storage/ziggerins_trials.csv",
        help="which csv of trial to use",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        help="which trial type to use",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=float,
        default=0.0,
        help="amount of noise to add",
    )
    parser.add_argument(
        "--encoding_noise",
        type=float,
        default=0.0,
        help="amount of encoding noise to add, based on durations",
    )
    parser.add_argument(
        "--memory_decay",
        type=float,
        default=0.0,
        help="amount of memory decay to add",
    )
    parser.add_argument(
        "--use_gpu", default=False, action="store_true", help="uses GPU"
    )
    args = parser.parse_args()

    if args.use_gpu:
        import torch

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    badModels = ["nts-net"]
    # Get all model files and information
    fileList = args.model_files.split(", ")

    # Read each json file as a dictionary
    hubModels = {}
    for file in fileList:
        with open(file, "r") as f:
            tmp = json.loads(f.read())

            # Add modelFile to each model
            for model in tmp.keys():
                tmp[model]["modelFile"] = file

            hubModels.update(tmp)

    if args.model_name is not None:
        # Fill a list with just that first model
        modelList = [args.model_name]
    else:
        # Fill a list with all the models
        modelList = list(hubModels.keys())

    if args.test == "threeAFC":
        # Setup results file
        # Check if results already exists
        resultsPath = f"../data_storage/results/results_{args.test}"
        if args.noise != 0:
            resultsPath += f"_noise-{args.noise}"

        if args.encoding_noise != 0:
            raise NotImplementedError("Encoding not implemented")
            resultsPath += f"_noise-{args.encoding_noise}"

        resultsPath += ".csv"
        if os.path.exists(resultsPath):
            results = pd.read_csv(resultsPath)
        else:
            results = pd.DataFrame(
                columns=[
                    "ModelName",
                    "Response",
                    "Corr",
                    "Trial",
                    "CorrRes",
                    "Duration",
                    "Feedback",
                ]
            )

        # Load model
        missingModels = []
        for modelName in modelList:
            modelData = hubModels[modelName]
            modelFile = modelData["modelFile"]

            rep_path = f"../data_storage/results/3afc_reps/{modelName.replace('/', '-')}-3afc.npy"
            image_name_path = f"../data_storage/results/3afc_reps/{modelName.replace('/', '-')}-3afc.txt"
            ddir = "../data_storage/standalone/3AFC_set"
            if os.path.exists(rep_path):
                print(f"Already have reps for {args.test} from {modelName}")

                if modelName in results["ModelName"].values:
                    print(f"Already have results for 3afc: {modelName}")
                    # Get the accuracy for this model
                    acc = results[results["ModelName"] == modelName]["Corr"].mean()
                    print(f"Accuracy for {modelName}: {acc}")
                    continue
                else:
                    reps = np.load(rep_path)
                    img_names = []
                    with open(image_name_path, "r") as f:
                        for line in f:
                            img_names.append(line.strip())
            else:
                if modelName in badModels:
                    print(f"Skipping {modelName}")
                    missingModels.append(modelName)

                    continue

                try:
                    model = get_model(modelName, modelFile, hubModels)
                except Exception as e:
                    # Echo exception
                    print(f"Error loading model {modelName}: {e}")
                    missingModels.append(modelName)

                    continue

                print(f"New reps for {args.test}: {modelName}")

                # Get file list and save
                img_names = os.listdir(ddir)
                img_names.sort()
                # Save file list as text file
                with open(
                    image_name_path,
                    "w",
                ) as f:
                    for file in img_names:
                        f.write(file + "\n")

                # Adjust batch size based on number of parameters
                if "num_params" not in modelData.keys():
                    batch_size = args.batch_size
                else:
                    batch_size = int(
                        args.batch_size
                        * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                    )
                    batch_size = 2 if batch_size < 2 else batch_size

                try:
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )
                except Exception as e:
                    print(f"Error making reps for {modelName}: {e}")
                    print(f"Attempting fallback batch_size")
                    batch_size = 2
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )

                # Flatten reps
                reps = reps.reshape(reps.shape[0], -1)
                np.save(rep_path, reps)

            print(f"New results for {args.test}: {modelName}")
            modelResults = three_afc(
                modelName,
                img_names,
                reps,
                "../data_storage/three_AFC_trials.csv",
                noise=args.noise,
                encoding_noise=args.encoding_noise,
            )
            results = pd.concat([results, modelResults])

            # Save results
            results.to_csv(resultsPath, index=False)

    elif args.test == "many_odd":
        # Setup results file
        # Check if results already exists
        resultsPath = f"../data_storage/results/results_{args.test}"
        if args.noise != 0:
            resultsPath += f"_noise-{args.noise}"

        if args.encoding_noise != 0:
            raise NotImplementedError("Encoding not implemented")
            resultsPath += f"_noise-{args.encoding_noise}"

        resultsPath += ".csv"
        if os.path.exists(resultsPath):
            results = pd.read_csv(resultsPath)
        else:
            results = pd.DataFrame(
                columns=[
                    "ModelName",
                    "Response",
                    "Corr",
                    "Trial",
                    "CorrRes",
                    "Duration",
                    "Feedback",
                ]
            )

        # Load model
        missingModels = []
        for modelName in modelList:
            modelData = hubModels[modelName]
            modelFile = modelData["modelFile"]

            rep_path = f"../data_storage/results/moo_reps/{modelName.replace('/', '-')}-moo.npy"
            image_name_path = f"../data_storage/results/moo_reps/{modelName.replace('/', '-')}-moo.txt"
            ddir = "../data_storage/standalone/MOO_set"
            if os.path.exists(rep_path):
                print(f"Already have reps for {args.test} from {modelName}")

                if modelName in results["ModelName"].values:
                    print(f"Already have results for moo: {modelName}")
                    # Get the accuracy for this model
                    acc = results[results["ModelName"] == modelName]["Corr"].mean()
                    print(f"Accuracy for {modelName}: {acc}")
                    continue
                else:
                    reps = np.load(rep_path)
                    img_names = []
                    with open(image_name_path, "r") as f:
                        for line in f:
                            img_names.append(line.strip())
            else:
                if modelName in badModels:
                    print(f"Skipping {modelName}")
                    missingModels.append(modelName)

                    continue
                try:
                    model = get_model(modelName, modelFile, hubModels)
                except Exception as e:
                    # Echo exception
                    print(f"Error loading model {modelName}: {e}")
                    missingModels.append(modelName)

                    continue

                print(f"New reps for {args.test}: {modelName}")

                # Get file list and save
                img_names = os.listdir(ddir)
                img_names.sort()
                # Save file list as text file
                with open(
                    image_name_path,
                    "w",
                ) as f:
                    for file in img_names:
                        f.write(file + "\n")

                # Adjust batch size based on number of parameters
                if "num_params" not in modelData.keys():
                    batch_size = args.batch_size
                else:
                    batch_size = int(
                        args.batch_size
                        * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                    )
                    batch_size = 2 if batch_size < 2 else batch_size

                try:
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )
                except Exception as e:
                    print(f"Error making reps for {modelName}: {e}")
                    print(f"Attempting fallback batch_size")
                    batch_size = 2
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )

                # Flatten reps
                reps = reps.reshape(reps.shape[0], -1)
                np.save(rep_path, reps)

            print(f"New results for {args.test}: {modelName}")
            modelResults = many_oddball(
                modelName,
                img_names,
                reps,
                "../data_storage/many_odd_trials.csv",
                noise=args.noise,
                encoding_noise=args.encoding_noise,
            )
            results = pd.concat([results, modelResults])

            # Save results
            results.to_csv(resultsPath, index=False)

    elif args.test == "learn_exemp":
        # Setup results file
        # Check if results already exists
        resultsPath = f"../data_storage/results/results_{args.test}"
        if args.noise != 0:
            resultsPath += f"_noise-{args.noise}"
        if args.memory_decay != 0:
            raise NotImplementedError("Memory decay not implemented")
            resultsPath += f"_memDecay-{args.memory_decay}"
        resultsPath += ".csv"
        if os.path.exists(resultsPath):
            results = pd.read_csv(resultsPath)
        else:
            results = pd.DataFrame(
                columns=[
                    "ModelName",
                    "Response",
                    "Corr",
                    "Trial",
                    "Img",
                    "Target",
                    "CorrRes",
                    "FoilLevel",
                    "View",
                    "Noise",
                    "Foil1",
                    "Foil2",
                ]
            )

        missingModels = []
        for modelName in modelList:
            modelData = hubModels[modelName]
            modelFile = modelData["modelFile"]

            rep_path = f"../data_storage/results/le_reps/{modelName.replace('/', '-')}-learnExemp.npy"
            image_name_path = f"../data_storage/results/le_reps/{modelName.replace('/', '-')}-learnExemp.txt"
            ddir = "../data_storage/standalone/LE_set"
            if os.path.exists(rep_path):
                print(f"Already have reps for {args.test} from {modelName}")

                if modelName in results["ModelName"].values:
                    print(f"Already have results for le: {modelName}")
                    # Get the accuracy for this model
                    acc = results[results["ModelName"] == modelName]["Corr"].mean()
                    print(f"Accuracy for {modelName}: {acc}")
                    continue
                else:
                    reps = np.load(rep_path)
                    img_names = []
                    with open(image_name_path, "r") as f:
                        for line in f:
                            img_names.append(line.strip())
            else:
                try:
                    if modelName in badModels:
                        print(f"Skipping {modelName}")
                        missingModels.append(modelName)

                        continue

                    model = get_model(modelName, modelFile, hubModels)
                except Exception as e:
                    # Echo exception
                    print(f"Error loading model {modelName}: {e}")
                    missingModels.append(modelName)

                    continue

                print(f"New reps for {args.test}: {modelName}")

                # Get file list and save
                img_names = os.listdir(ddir)
                img_names.sort()
                # Save file list as text file
                with open(
                    image_name_path,
                    "w",
                ) as f:
                    for file in img_names:
                        f.write(file + "\n")

                # Adjust batch size based on number of parameters
                if "num_params" not in modelData.keys():
                    batch_size = args.batch_size
                else:
                    batch_size = int(
                        args.batch_size
                        * (1 / 2 ** int(np.log10(int(modelData["num_params"])) - 3))
                    )
                    batch_size = 2 if batch_size < 2 else batch_size

                try:
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )
                except Exception as e:
                    print(f"Error making reps for {modelName}: {e}")
                    print(f"Attempting fallback batch_size")
                    batch_size = 2
                    reps = rep_maker(
                        ddir,
                        modelFile,
                        modelName,
                        modelData,
                        model,
                        batch_size,
                    )

                # Flatten reps
                reps = reps.reshape(reps.shape[0], -1)
                np.save(rep_path, reps)

                print(f"New results for {args.test}: {modelName}")
                modelResults = learning_exemplar(
                    modelName,
                    img_names,
                    reps,
                    "../data_storage/learning_exemplar_trials.csv",
                    noise=args.noise,
                    memory_decay=args.memory_decay,
                )
                results = pd.concat([results, modelResults])

                # Save results
                results.to_csv(resultsPath, index=False)

    else:
        print(f"Invalid test: {args.test}")

    print("Missing models:", missingModels)
