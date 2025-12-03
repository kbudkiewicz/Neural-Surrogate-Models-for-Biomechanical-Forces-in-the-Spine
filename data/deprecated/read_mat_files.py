#%%
#from scipy.io import loadmat
import numpy as np
import os   
path = "/mnt/8tb_slot8/jonas/datasets/CT_nofx/models/simpack/Results/matlab/sub-basilarstroke0012_ses-20170825-Optim_10kg_25cm.mat"

path = "/mnt/8tb_slot8/jonas/datasets/CT_nofx/models/simpack/Results/matlab/sub-basilarstroke0012_ses-20170825-Optim_10kg_55cm.mat"
datapath = "/mnt/8tb_slot8/jonas/datasets/CT_nofx/models/simpack/Results/matlab/"

# %%
import h5py

def hdf5_to_dict(h5file):
    def recursively_load(name, obj):
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        elif isinstance(obj, h5py.Group):
            return {k: recursively_load(k, obj[k]) for k in obj.keys()}
    return {key: recursively_load(key, h5file[key]) for key in h5file.keys()}


def forPatGet_yout(path,):
# Usage
    with h5py.File(path, 'r') as f:
        data_dict = hdf5_to_dict(f)

    data_dict.keys()  # Check the keys in the dictionary
    data_dict['timeInt']  # Access a specific variable

    return data_dict['timeInt']["yout"]#.keys()  # Check the keys in the 'timeInt' variable

#%%
listPath = os.listdir(datapath)
selectedPat = []
for pat in listPath:
    if "Optim_0" in pat: #if "25cm" in pat: #if "Optim_0" in pat:
        selectedPat.append(pat)


#%%
allResults = {}
for i in range(len(selectedPat)):
    print(selectedPat[i])
    dict = forPatGet_yout(os.path.join(datapath, selectedPat[i]))
    allResults[selectedPat[i]] = dict

    #if i > 50:
    #    break
# save the results to a file as npz
import os
import numpy as np
output_path = os.path.join( 'allResults_0cm.npz')
np.savez(output_path, **allResults)


#%% read the dict again
import numpy as np
allResultsRead = np.load('allResults_0cm.npz', allow_pickle=True)
allResultsRead = {k: v.item() for k, v in allResultsRead.items()}


#%%

L1, L2, L3, L4, L5 = [], [], [], [], []
for pat in allResults.keys():
    L1.append(np.mean(allResults[pat]['SG_Y_Force__G_Y_compr__Y_L1']["values"]))
    L2.append(np.mean(allResults[pat]['SG_Y_Force__G_Y_compr__Y_L2']["values"]))
    L3.append(np.mean(allResults[pat]['SG_Y_Force__G_Y_compr__Y_L3']["values"]))
    L4.append(np.mean(allResults[pat]['SG_Y_Force__G_Y_compr__Y_L4']["values"]))
    L5.append(np.mean(allResults[pat]['SG_Y_Force__G_Y_compr__Y_L5']["values"]))
    print("")

#%%
import matplotlib.pyplot as plt

plt.plot(L1)
plt.plot(L2)
plt.plot(L3)
plt.plot(L4)
plt.plot(L5)
plt.legend(['L1', 'L2', 'L3', 'L4', 'L5'])
plt.xlabel('Patient Index')
plt.ylabel('Mean Compression Force')
plt.title('Mean Compression Force for Each Lumbar Vertebra')
plt.show()

# %%
datapoints =  data_dict['timeInt']["yout"].keys() 
print(datapoints)
# %%
dict_keys(['SG_Y_Ang__G_Y_ga__Y_L1', 'SG_Y_Ang__G_Y_ga__Y_L2', 'SG_Y_Ang__G_Y_ga__Y_L3', 'SG_Y_Ang__G_Y_ga__Y_L4', 'SG_Y_Ang__G_Y_ga__Y_L5', 'SG_Y_Ang__G_Y_ga__Y_SacrumDummy', 'SG_Y_Ang__G_Y_ga__Y_T12', 'SG_Y_Force__G_Y_compr__Y_L1', 'SG_Y_Force__G_Y_compr__Y_L2', 'SG_Y_Force__G_Y_compr__Y_L3', 'SG_Y_Force__G_Y_compr__Y_L4', 'SG_Y_Force__G_Y_compr__Y_L5', 'SG_Y_Force__G_Y_compr__Y_T12', 'SG_Y_Force__G_Y_shear_AP__Y_L1', 'SG_Y_Force__G_Y_shear_AP__Y_L2', 'SG_Y_Force__G_Y_shear_AP__Y_L3', 'SG_Y_Force__G_Y_shear_AP__Y_L4', 'SG_Y_Force__G_Y_shear_AP__Y_L5', 'SG_Y_Force__G_Y_shear_AP__Y_T12', 'SG_Y_Force__G_Y_shear_ML__Y_L1', 'SG_Y_Force__G_Y_shear_ML__Y_L2', 'SG_Y_Force__G_Y_shear_ML__Y_L3', 'SG_Y_Force__G_Y_shear_ML__Y_L4', 'SG_Y_Force__G_Y_shear_ML__Y_L5', 'SG_Y_Force__G_Y_shear_ML__Y_T12', 'SG_Y_Torque__G_Y_al__Y_L1', 'SG_Y_Torque__G_Y_al__Y_L2', 'SG_Y_Torque__G_Y_al__Y_L3', 'SG_Y_Torque__G_Y_al__Y_L4', 'SG_Y_Torque__G_Y_al__Y_L5', 'SG_Y_Torque__G_Y_al__Y_T12', 'SG_Y_Torque__G_Y_be__Y_L1', 'SG_Y_Torque__G_Y_be__Y_L2', 'SG_Y_Torque__G_Y_be__Y_L3', 'SG_Y_Torque__G_Y_be__Y_L4', 'SG_Y_Torque__G_Y_be__Y_L5', 'SG_Y_Torque__G_Y_be__Y_T12', 'SG_Y_Torque__G_Y_ga__Y_L1', 'SG_Y_Torque__G_Y_ga__Y_L2', 'SG_Y_Torque__G_Y_ga__Y_L3', 'SG_Y_Torque__G_Y_ga__Y_L4', 'SG_Y_Torque__G_Y_ga__Y_L5', 'SG_Y_Torque__G_Y_ga__Y_T12', 'SY_Mov_Arms', 'c_name', 'comment', 'description', 'name'])

dict_keys(['SG_Y_Ang__G_Y_ga__Y_L1', 'SG_Y_Ang__G_Y_ga__Y_L2', 'SG_Y_Ang__G_Y_ga__Y_L3', 'SG_Y_Ang__G_Y_ga__Y_L4', 'SG_Y_Ang__G_Y_ga__Y_L5', 'SG_Y_Ang__G_Y_ga__Y_SacrumDummy', 'SG_Y_Ang__G_Y_ga__Y_T12', 'SG_Y_Force__G_Y_compr__Y_L1', 'SG_Y_Force__G_Y_compr__Y_L2', 'SG_Y_Force__G_Y_compr__Y_L3', 'SG_Y_Force__G_Y_compr__Y_L4', 'SG_Y_Force__G_Y_compr__Y_L5', 'SG_Y_Force__G_Y_compr__Y_T12', 'SG_Y_Force__G_Y_shear_AP__Y_L1', 'SG_Y_Force__G_Y_shear_AP__Y_L2', 'SG_Y_Force__G_Y_shear_AP__Y_L3', 'SG_Y_Force__G_Y_shear_AP__Y_L4', 'SG_Y_Force__G_Y_shear_AP__Y_L5', 'SG_Y_Force__G_Y_shear_AP__Y_T12', 'SG_Y_Force__G_Y_shear_ML__Y_L1', 'SG_Y_Force__G_Y_shear_ML__Y_L2', 'SG_Y_Force__G_Y_shear_ML__Y_L3', 'SG_Y_Force__G_Y_shear_ML__Y_L4', 'SG_Y_Force__G_Y_shear_ML__Y_L5', 'SG_Y_Force__G_Y_shear_ML__Y_T12', 'SG_Y_Torque__G_Y_al__Y_L1', 'SG_Y_Torque__G_Y_al__Y_L2', 'SG_Y_Torque__G_Y_al__Y_L3', 'SG_Y_Torque__G_Y_al__Y_L4', 'SG_Y_Torque__G_Y_al__Y_L5', 'SG_Y_Torque__G_Y_al__Y_T12', 'SG_Y_Torque__G_Y_be__Y_L1', 'SG_Y_Torque__G_Y_be__Y_L2', 'SG_Y_Torque__G_Y_be__Y_L3', 'SG_Y_Torque__G_Y_be__Y_L4', 'SG_Y_Torque__G_Y_be__Y_L5', 'SG_Y_Torque__G_Y_be__Y_T12', 'SG_Y_Torque__G_Y_ga__Y_L1', 'SG_Y_Torque__G_Y_ga__Y_L2', 'SG_Y_Torque__G_Y_ga__Y_L3', 'SG_Y_Torque__G_Y_ga__Y_L4', 'SG_Y_Torque__G_Y_ga__Y_L5', 'SG_Y_Torque__G_Y_ga__Y_T12', 'SY_Mov_Arms', 'c_name', 'comment', 'description', 'name'])