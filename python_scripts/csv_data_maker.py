import pandas as pd
import numpy as np
from csv import writer

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
    
                    # Pass this file object to csv.writer()
                    # and get a writer object
                    writer_object = writer(f_object)
                
                    # Pass the list as an argument into
                    # the writerow()
                    writer_object.writerow(data_list)
                
                    # Close the file object
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
    
                    # Pass this file object to csv.writer()
                    # and get a writer object
                    writer_object = writer(f_object)
                
                    # Pass the list as an argument into
                    # the writerow()
                    writer_object.writerow(data_list)
                
                    # Close the file object
                    f_object.close()