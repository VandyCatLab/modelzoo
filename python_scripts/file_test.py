import numpy as np


name1 = '/data/brustdm/modelnet/python_scripts/data_storage/many_odd_results.npy'
name2 = '/data/brustdm/modelnet/python_scripts/data_storage/learn_exemp_results.npy'
name3 = '/data/brustdm/modelnet/python_scripts/data_storage/3ACF_results.npy'
test1 = np.load(name1, allow_pickle=True)
test2 = np.load(name2, allow_pickle=True)
test3 = np.load(name3, allow_pickle=True)
print('Lengths:', len(test1)/2, len(test2)/2, len(test3)/2)
print(test3[1][0])   
#print(test2[0:10])   
#print(test3[0:10])EXIT()
print(test3[2][1])
