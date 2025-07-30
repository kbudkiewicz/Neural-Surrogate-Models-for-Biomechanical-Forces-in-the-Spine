#%%
from scipy.io import loadmat
path = "/mnt/8tb_slot8/jonas/datasets/CT_nofx/models/simpack/Results/matlab/sub-basilarstroke0012_ses-20170825-Optim_10kg_25cm.mat"

path = "/mnt/8tb_slot8/jonas/datasets/CT_nofx/models/simpack/Results/matlab/sub-basilarstroke0012_ses-20170825-Optim_10kg_55cm.mat"

# %%
import h5py

def hdf5_to_dict(h5file):
    def recursively_load(name, obj):
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        elif isinstance(obj, h5py.Group):
            return {k: recursively_load(k, obj[k]) for k in obj.keys()}
    return {key: recursively_load(key, h5file[key]) for key in h5file.keys()}

# Usage
with h5py.File(path, 'r') as f:
    data_dict = hdf5_to_dict(f)

data_dict.keys()  # Check the keys in the dictionary
data_dict['timeInt']  # Access a specific variable



data_dict['timeInt']["yout"]["SG_Y_Force__G_Y_compr__Y_L1"]["values"]#.keys()  # Check the keys in the 'timeInt' variable

# %%
datapoints =  data_dict['timeInt']["yout"].keys() 
print(datapoints)
# %%
dict_keys(['SG_Y_Ang__G_Y_ga__Y_L1', 'SG_Y_Ang__G_Y_ga__Y_L2', 'SG_Y_Ang__G_Y_ga__Y_L3', 'SG_Y_Ang__G_Y_ga__Y_L4', 'SG_Y_Ang__G_Y_ga__Y_L5', 'SG_Y_Ang__G_Y_ga__Y_SacrumDummy', 'SG_Y_Ang__G_Y_ga__Y_T12', 'SG_Y_Force__G_Y_compr__Y_L1', 'SG_Y_Force__G_Y_compr__Y_L2', 'SG_Y_Force__G_Y_compr__Y_L3', 'SG_Y_Force__G_Y_compr__Y_L4', 'SG_Y_Force__G_Y_compr__Y_L5', 'SG_Y_Force__G_Y_compr__Y_T12', 'SG_Y_Force__G_Y_shear_AP__Y_L1', 'SG_Y_Force__G_Y_shear_AP__Y_L2', 'SG_Y_Force__G_Y_shear_AP__Y_L3', 'SG_Y_Force__G_Y_shear_AP__Y_L4', 'SG_Y_Force__G_Y_shear_AP__Y_L5', 'SG_Y_Force__G_Y_shear_AP__Y_T12', 'SG_Y_Force__G_Y_shear_ML__Y_L1', 'SG_Y_Force__G_Y_shear_ML__Y_L2', 'SG_Y_Force__G_Y_shear_ML__Y_L3', 'SG_Y_Force__G_Y_shear_ML__Y_L4', 'SG_Y_Force__G_Y_shear_ML__Y_L5', 'SG_Y_Force__G_Y_shear_ML__Y_T12', 'SG_Y_Torque__G_Y_al__Y_L1', 'SG_Y_Torque__G_Y_al__Y_L2', 'SG_Y_Torque__G_Y_al__Y_L3', 'SG_Y_Torque__G_Y_al__Y_L4', 'SG_Y_Torque__G_Y_al__Y_L5', 'SG_Y_Torque__G_Y_al__Y_T12', 'SG_Y_Torque__G_Y_be__Y_L1', 'SG_Y_Torque__G_Y_be__Y_L2', 'SG_Y_Torque__G_Y_be__Y_L3', 'SG_Y_Torque__G_Y_be__Y_L4', 'SG_Y_Torque__G_Y_be__Y_L5', 'SG_Y_Torque__G_Y_be__Y_T12', 'SG_Y_Torque__G_Y_ga__Y_L1', 'SG_Y_Torque__G_Y_ga__Y_L2', 'SG_Y_Torque__G_Y_ga__Y_L3', 'SG_Y_Torque__G_Y_ga__Y_L4', 'SG_Y_Torque__G_Y_ga__Y_L5', 'SG_Y_Torque__G_Y_ga__Y_T12', 'SY_Mov_Arms', 'c_name', 'comment', 'description', 'name'])

dict_keys(['SG_Y_Ang__G_Y_ga__Y_L1', 'SG_Y_Ang__G_Y_ga__Y_L2', 'SG_Y_Ang__G_Y_ga__Y_L3', 'SG_Y_Ang__G_Y_ga__Y_L4', 'SG_Y_Ang__G_Y_ga__Y_L5', 'SG_Y_Ang__G_Y_ga__Y_SacrumDummy', 'SG_Y_Ang__G_Y_ga__Y_T12', 'SG_Y_Force__G_Y_compr__Y_L1', 'SG_Y_Force__G_Y_compr__Y_L2', 'SG_Y_Force__G_Y_compr__Y_L3', 'SG_Y_Force__G_Y_compr__Y_L4', 'SG_Y_Force__G_Y_compr__Y_L5', 'SG_Y_Force__G_Y_compr__Y_T12', 'SG_Y_Force__G_Y_shear_AP__Y_L1', 'SG_Y_Force__G_Y_shear_AP__Y_L2', 'SG_Y_Force__G_Y_shear_AP__Y_L3', 'SG_Y_Force__G_Y_shear_AP__Y_L4', 'SG_Y_Force__G_Y_shear_AP__Y_L5', 'SG_Y_Force__G_Y_shear_AP__Y_T12', 'SG_Y_Force__G_Y_shear_ML__Y_L1', 'SG_Y_Force__G_Y_shear_ML__Y_L2', 'SG_Y_Force__G_Y_shear_ML__Y_L3', 'SG_Y_Force__G_Y_shear_ML__Y_L4', 'SG_Y_Force__G_Y_shear_ML__Y_L5', 'SG_Y_Force__G_Y_shear_ML__Y_T12', 'SG_Y_Torque__G_Y_al__Y_L1', 'SG_Y_Torque__G_Y_al__Y_L2', 'SG_Y_Torque__G_Y_al__Y_L3', 'SG_Y_Torque__G_Y_al__Y_L4', 'SG_Y_Torque__G_Y_al__Y_L5', 'SG_Y_Torque__G_Y_al__Y_T12', 'SG_Y_Torque__G_Y_be__Y_L1', 'SG_Y_Torque__G_Y_be__Y_L2', 'SG_Y_Torque__G_Y_be__Y_L3', 'SG_Y_Torque__G_Y_be__Y_L4', 'SG_Y_Torque__G_Y_be__Y_L5', 'SG_Y_Torque__G_Y_be__Y_T12', 'SG_Y_Torque__G_Y_ga__Y_L1', 'SG_Y_Torque__G_Y_ga__Y_L2', 'SG_Y_Torque__G_Y_ga__Y_L3', 'SG_Y_Torque__G_Y_ga__Y_L4', 'SG_Y_Torque__G_Y_ga__Y_L5', 'SG_Y_Torque__G_Y_ga__Y_T12', 'SY_Mov_Arms', 'c_name', 'comment', 'description', 'name'])