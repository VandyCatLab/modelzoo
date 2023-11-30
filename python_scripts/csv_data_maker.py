import pandas as pd
import numpy as np
from csv import writer

def old_format():
    data_type = '3ACF'
    npa = np.load(f'modelnet/python_scripts/data_storage/{data_type}_results.npy', allow_pickle=True)
    pda = pd.DataFrame(columns=['SbjID', 'response', 'TrialN', 'CorrRes', 'Correct'])
    pda.to_csv(f'modelnet/python_scripts/data_storage/csv_data_maker_{data_type}.csv', index=False)
    if data_type != '3ACF':
        for i in range(len(npa)-1):
            if isinstance(npa[i][0],str) and isinstance(npa[i+1][0], int):
                print(0)
                for j in range(len(npa[i+1][2])):
                    if npa[i+1][2][j][0] == 'incorrect':
                        CorrRes = f'diffAnswer'
                    else:
                        CorrRes = npa[i+1][2][j][1]

                    if npa[i+1][2][j][0] == 'incorrect':
                        CorrAns = False
                    else:
                        CorrAns = True
                    TrialN = j + 1
                    data_list = [npa[i], npa[i+1][2][j][1], TrialN, CorrRes, CorrAns]
                    with open(f'modelnet/python_scripts/data_storage/csv_data_maker_{data_type}.csv', 'a') as f_object:

                        writer_object = writer(f_object)

                        writer_object.writerow(data_list)

                        f_object.close()
                

    else:
        print(0)
        for i in range(len(npa)-1):
            if i%2 == 0:
                for j in range(1, len(npa[i+1][1][2])):
                    if npa[i+1][1][2][j][0] == 'incorrect':
                        CorrRes = f'diffAnswer'
                    else:
                        CorrRes = npa[i+1][1][2][j][1]

                    if npa[i+1][1][2][j][0] == 'incorrect':
                        CorrAns = False
                    else:
                        CorrAns = True
                    TrialN = j + 1
                    data_list = [npa[i], npa[i+1][1][2][j][1], TrialN, CorrRes, CorrAns]
                    with open(f'modelnet/python_scripts/data_storage/csv_data_maker_{data_type}.csv', 'a') as f_object:

                        writer_object = writer(f_object)

                        writer_object.writerow(data_list)

                        f_object.close()

def make_csv_normal(data_types=['many_odd', 'learn_exemp', 'threeACF'], noise=False):
    print('ok')
    for data_type in data_types:
        if noise:
            npa = np.load(f'../data_storage/results/{data_type}_results-noise.npy', allow_pickle=True)
            pda = pd.DataFrame(columns=['SbjID', 'response', 'TrialN', 'CorrRes', 'AnsCatagory'])
            csv_path = f'../data_storage/results/csv_data_maker_{data_type}-noise.csv'
            pda.to_csv(csv_path, index=False)
        else:
            npa = np.load(f'../data_storage/results/{data_type}_results.npy', allow_pickle=True)
            pda = pd.DataFrame(columns=['SbjID', 'response', 'TrialN', 'CorrRes', 'AnsCatagory'])
            csv_path = f'../data_storage/results/csv_data_maker_{data_type}.csv'
            pda.to_csv(csv_path, index=False)

        for row in npa[1::2]:
            i = 0
            for item in row[4]:
                new_data_input = [row[0], item[2], i, item[1], item[0]]
                with open(csv_path, 'a') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(new_data_input)
                    f_object.close()
                i += 1

def add_to_csv(filetype, data_list, column_name):
    file_paths = []
    file_paths.append('../data_storage/results/csv_data_maker_' + filetype + '.csv')
    file_paths.append('../data_storage/results/csv_data_maker_' + filetype + '-noise.csv')
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Ensure 'TrialN' column exists
        if 'TrialN' not in df.columns:
            raise ValueError("The CSV file must contain a 'TrialN' column.")

        # Ensure the column exists or create it if not
        if column_name not in df.columns:
            df[column_name] = float('nan')

        # Insert data based on TrialN
        for index, row in df.iterrows():
            trial_n = row['TrialN']
            # Check for index bounds in the list and insert data
            if trial_n < len(data_list):
                df.at[index, column_name] = data_list[trial_n]

        # Save the modified DataFrame back to CSV
        df.to_csv(file_path, index=False)

data_types=['many_odd', 'learn_exemp', 'threeACF']
threeACF_view_list = ['s'] * 11 + ['d'] * 16 + ['s'] * 12 + ['d'] * 12
threeACF_noise_list = ['n'] * 27 + ['y'] * 24
learn_exemp_view_list = ['s'] * 25 + ['d'] * 23
learn_exemp_noise_list = ['n'] * 18 + ['y'] * 6 + ['n'] * 12 + ['y'] * 12
many_odd_view_list = ['d'] * 45
many_odd_noise_list = ['n'] * 12 + ['y'] * 10 + ['n'] * 3 + ['y'] + ['n'] * 4 + ['y'] + ['n'] * 2 + ['y'] + ['n'] * 2 + ['y'] + ['n'] * 8
many_odd_size_list = ['y'] * 45

add_to_csv(data_types[2], threeACF_view_list, "View")
add_to_csv(data_types[2], threeACF_noise_list, "Noise")
add_to_csv(data_types[1], learn_exemp_view_list, "View")
add_to_csv(data_types[1], learn_exemp_noise_list, "Noise")
add_to_csv(data_types[0], many_odd_view_list, "View")
add_to_csv(data_types[0], many_odd_noise_list, "Noise")



#make_csv_normal(noise=True)