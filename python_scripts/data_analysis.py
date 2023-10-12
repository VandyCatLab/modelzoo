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


def get_names_lists(arr1, arr2, arr3):
    
    names_lists = [
        np.array([arr1[i] for i, string in enumerate(arr1[0::2])], dtype=object),
        np.array([arr2[i] for i, string in enumerate(arr2[0::2])], dtype=object),
        np.array([arr3[i] for i, string in enumerate(arr3[0::2])], dtype=object)
    ]

    return names_lists

def get_accuracies(arr1, arr2, arr3):

    names_lists = get_names_lists(arr1, arr2, arr3)
    accuracies_array = [
        tuple(np.array([item[1][2] for i, item in enumerate(arr1[1::2])], dtype=object)),
        tuple(np.array([item[2] for item in arr2[1::2]], dtype=object)),
        tuple(np.array([item[2] for item in arr3[1::2]], dtype=object))
    ]

    return accuracies_array, names_lists


def test_arrays(arr1, arr2, arr3):
    # Function written with the help og ChatGPT
    # Extract strings from the arrays
    strings_arr1 = arr1[0::2]
    strings_arr2 = arr2[0::2]
    strings_arr3 = arr3[0::2]
    
    # Check if the strings are the same and in the same order across all arrays
    return np.array_equal(strings_arr1, strings_arr2) and np.array_equal(strings_arr2, strings_arr3)

def check_same_value(arr, index):
    # Get the value of the first element in the specified column
    first_value = arr[0][index]
    
    # Check if all other elements in the specified column have the same value
    for item in arr[1:]:
        if item[index] != first_value:
            return False
    
    return True


def clean_cross_array_acc(arr1, arr2, arr3):

    names_lists = get_names_lists(arr1, arr2, arr3)

    arr1_list, arr2_list, arr3_list = [], [], []
    same_answer_count = 0
    for i in range(len(arr1)):
        if i % 2 != 0:
            for j in range(len(arr1[i][1][3])):
                if arr1[i][1][3][j][0] == arr2[i][3][j][0] == arr3[i][3][j][0]:
                    same_answer_count += 1
            arr1_corr, arr1_tot = int(arr1[i][1][0]), int(arr1[i][1][1])
            arr2_corr, arr2_tot = int(arr2[i][0]), int(arr2[i][1])
            arr3_corr, arr3_tot = int(arr3[i][0]), int(arr3[i][1])
            arr1_per = (arr1_corr - same_answer_count) / (arr1_tot - same_answer_count)
            arr2_per = (arr2_corr - same_answer_count) / (arr2_tot - same_answer_count)
            arr3_per = (arr3_corr - same_answer_count) / (arr3_tot - same_answer_count)
            arr1_list.append(arr1_per)
            arr2_list.append(arr2_per)
            arr3_list.append(arr3_per)
            
    accuracies_array = [
        tuple(arr1_list),
        tuple(arr2_list),
        tuple(arr3_list)
    ]

    return accuracies_array, names_lists


def old_method_start():
    l1 = np.load('../data_storage/data_backup/many_odd_results_new.npy', allow_pickle=True)
    l2 = np.load('../data_storage/data_backup/threeACF_results_new.npy', allow_pickle=True)
    l3 = np.load('../data_storage/data_backup/learn_exemp_results_new.npy', allow_pickle=True)
    print(len(l1), len(l2), len(l3))
    #print(l2)

    shaved_l1, shaved_l2, shaved_l3 = array_shaver(l1, l2, l3)

    ##print(len(shaved_l1), len(shaved_l2), len(shaved_l3))

    test_result = test_arrays(shaved_l1, shaved_l2, shaved_l3)
    #print('\n\n', shaved_l3[1::2][0])
    if test_result:
        accuracies, names = get_accuracies(shaved_l1, shaved_l2, shaved_l3)
        correlation = np.corrcoef(accuracies)
        print(correlation)

# 'SbjID', 'response', 'TrialN', 'CorrRes', 'AnsCatagory'
def analyze_trials_filtered(df):
    
    percent_correct_before = df.groupby('SbjID')['AnsCatagory'].apply(lambda x: (x == 'correct').mean() * 100)
    #print(len(percent_correct_before))
    trials_to_remove = []
    unique_trials = df['TrialN'].unique()
    #print(unique_trials)
    for trial in unique_trials:
        trial_df = df[df['TrialN'] == trial]
        if len(trial_df['AnsCatagory'].unique()) == 1:
            trials_to_remove.append(trial)
    #print(trials_to_remove)

    df_filtered = df[~df['TrialN'].isin(trials_to_remove)]
    #print('\n\n\n\n\n',len(df_filtered) / (len(unique_trials) - len(trials_to_remove)))
    percent_correct_after = df_filtered.groupby('SbjID')['AnsCatagory'].apply(lambda x: (x == 'correct').mean() * 100)
    #print(len(percent_correct_after))
    #filtered_path = file_path[:-5] + 'filtered' + file_path[-5:]
    #df_filtered.to_csv(filtered_path, index=False)

    return percent_correct_before, percent_correct_after


def ensure_networks(dfs):

    common_networks = set(dfs[0]['SbjID'].unique())
    for df in dfs[1:]:
        common_networks.intersection_update(df['SbjID'].unique())
    
    networks_to_remove = set()
    for network in common_networks:
        all_incorrect_for_trials = any(all(df[df['SbjID'] == network]['AnsCatagory'] == 'incorrect') for df in dfs)

        if all_incorrect_for_trials:
            networks_to_remove.add(network)
    #print(networks_to_remove)
    final_networks = common_networks - networks_to_remove

    #print(len(final_networks))
    filtered_dfs = [df[df['SbjID'].isin(final_networks)] for df in dfs]
    
    return filtered_dfs

def compare_dfs(data_before, data_after):
    
    comparison_df = pd.DataFrame({
        'Before Filtering': data_before,
        'After Filtering': data_after
    }, index=[0])
    
    # Calculate the differences
    comparison_df['Difference'] = comparison_df['After Filtering'] - comparison_df['Before Filtering']
    
    final_compare = comparison_df[comparison_df['Difference'] != 0]
    return final_compare['Difference'].to_list()



def csv_accuracies(csv_files=['../data_storage/results/csv_data_maker_threeACF.csv', '../data_storage/results/csv_data_maker_many_odd.csv', '../data_storage/results/csv_data_maker_learn_exemp.csv']):
    #bad_network_list_learn_exemp = ['inception_resnet_v2', 'xception', 'inception_v3']

    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(file))
    
    filtered_df_list = ensure_networks(dfs)

    cross_data_before = []
    cross_data_after = []
    for filtered_df in filtered_df_list:
        per_corr_before, per_corr_after = analyze_trials_filtered(filtered_df)
        cross_data_before.append(per_corr_before)
        cross_data_after.append(per_corr_after)

    accuracies_before = [
        list(cross_data_before[0]),
        list(cross_data_before[1]),
        list(cross_data_before[2])
    ]
    accuracies_after = [
        list(cross_data_after[0]),
        list(cross_data_after[1]),
        list(cross_data_after[2])
    ]

    correlation_before = np.corrcoef(accuracies_before)
    correlation_after = np.corrcoef(accuracies_after)
    print('\n#######################################')
    print('Correlation before removing trials:\n', correlation_before, '\n\nCorrelation after removing trials:\n', correlation_after)
    print('#######################################\n')
    #print(list(cross_data_before[2]))
    #print('\n\n\n', list(cross_data_after[2]))
    '''''
    for i in range(3):
        cross_data_before[i]['']
    
    
    compare_list = [[],[],[]]
    for i in range(len(cross_data_before)):
        compare_list[i].append(compare_dfs(cross_data_before[i], cross_data_after[i]))

    print(compare_list)

    #print(cross_data_before.equals(cross_data_after))
    
    i = 1
    print('Cross Data Before')
    print(cross_data_before[i])
    print('Cross Data After\n', cross_data_after[i])
    print(len(cross_data_before), len(cross_data_after))
    '''''

csv_accuracies()

