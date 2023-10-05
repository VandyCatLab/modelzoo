import numpy as np
import pandas as pd

def accuracy_correlation(accuracies_array, names):

    df = pd.DataFrame(columns=tuple(names))
    for i in range(1, len(accuracies_array), 2):
        new_dat = [accuracies_array[0][i], accuracies_array[1][i], accuracies_array[2][i]]
        print(new_dat, type(new_dat))
        df[names[i]] = tuple(new_dat)
    corr_mat = df.corr()

    return corr_mat


def get_accuracies(arr1, arr2, arr3):

    names_lists = [
        np.array([arr1[i] for i, string in enumerate(arr1[0::2])], dtype=object),
        np.array([arr2[i] for i, string in enumerate(arr2[0::2])], dtype=object),
        np.array([arr3[i] for i, string in enumerate(arr3[0::2])], dtype=object)
    ]
    accuracies_array = [
        tuple(np.array([item[1][2] for i, item in enumerate(arr1[1::2])], dtype=object)),
        tuple(np.array([item[2] for item in arr2[1::2]], dtype=object)),
        tuple(np.array([item[2] for item in arr3[1::2]], dtype=object))
    ]


    return accuracies_array, names_lists
    

def array_shaver(arr1, arr2, arr3):
    # Function written with the help og ChatGPT
    # Extract strings from the arrays
    strings_arr1 = arr1[0::2]
    strings_arr2 = arr2[0::2]
    strings_arr3 = arr3[0::2]
    
    # Identify common strings
    common_strings = np.intersect1d(np.intersect1d(strings_arr1, strings_arr2), strings_arr3)
    
    # Create new arrays with only the common strings and their subsequent arrays
    new_arr1 = np.array([item for i, item in enumerate(arr1) if i % 2 == 0 and item in common_strings or i % 2 == 1 and arr1[i-1] in common_strings], dtype=object)
    new_arr2 = np.array([item for i, item in enumerate(arr2) if i % 2 == 0 and item in common_strings or i % 2 == 1 and arr2[i-1] in common_strings], dtype=object)
    new_arr3 = np.array([item for i, item in enumerate(arr3) if i % 2 == 0 and item in common_strings or i % 2 == 1 and arr3[i-1] in common_strings], dtype=object)
    
    return new_arr1, new_arr2, new_arr3


def test_arrays(arr1, arr2, arr3):
    # Function written with the help og ChatGPT
    # Extract strings from the arrays
    strings_arr1 = arr1[0::2]
    strings_arr2 = arr2[0::2]
    strings_arr3 = arr3[0::2]
    
    # Check if the strings are the same and in the same order across all arrays
    return np.array_equal(strings_arr1, strings_arr2) and np.array_equal(strings_arr2, strings_arr3)


l1 = np.load('../data_storage/many_odd_results_new.npy', allow_pickle=True)
l2 = np.load('../data_storage/threeACF_results_new.npy', allow_pickle=True)
l3 = np.load('../data_storage/learn_exemp_results_new.npy', allow_pickle=True)
#print(len(l1), len(l2), len(l3))

shaved_l1, shaved_l2, shaved_l3 = array_shaver(l1, l2, l3)

##print(len(shaved_l1), len(shaved_l2), len(shaved_l3))

test_result = test_arrays(shaved_l1, shaved_l2, shaved_l3)
#print('\n\n', shaved_l3[1::2][0])

if test_result:
    accuracies, names = get_accuracies(shaved_l1, shaved_l2, shaved_l3)
    correlation = np.corrcoef(accuracies)
    print(correlation)

