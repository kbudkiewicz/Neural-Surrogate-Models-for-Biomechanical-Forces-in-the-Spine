#%%
import numpy as np
import nibabel as nib

datasetPath = "/mnt/8tb_slot8/jonas/datasets/CT_nofx/derivatives/"
scalarsToAddToCSV = ['SG_Y_Force__G_Y_compr__Y_L1', 'SG_Y_Force__G_Y_compr__Y_L2', 'SG_Y_Force__G_Y_compr__Y_L3', 'SG_Y_Force__G_Y_compr__Y_L4', 'SG_Y_Force__G_Y_compr__Y_L5']

Dict = np.load('allResults_0cm.npz', allow_pickle=True)
import os
Dict = {k: v.item() for k, v in Dict.items()}   

# %%
Dict.keys()  # Check the keys in the dictionary
len(Dict)  # Number of keys
# %%
ct_image_paths, allScalars, labels = [], [], []
for k, v in Dict.items():
    #print(k)#, v.keys())  # Print keys of each item in the dictionary

    pat = k.split('_')[0]  # Extract patient ID from the key
    ses = "ses-" + k.split('_')[1].split('-')[1]
    #print(f"Patient: {pat}, Session: {ses}")  # Print patient and session information

    patFolder = os.path.join(datasetPath, pat, ses)
    #find ct nifti file
    ct_images = []
    for f in os.listdir(patFolder):
        #print(f)
        if "ct.nii.gz" in f:
            ct_images.append(f)
    if not len(ct_images) == 1:
        print(f"Expected one CT image, found {len(ct_images)} in {patFolder}")
        continue

    
    ctImgPath = os.path.join(datasetPath, pat, ses, ct_images[0])

    try:
        img = nib.load(ctImgPath, mmap=False)  # Load the NIfTI file
        img.get_fdata().shape
        img = nib.load(ctImgPath.replace("_ct.nii.gz", "_seg-tissue_msk.nii.gz"), mmap=False)  # Load the NIfTI file
        img.get_fdata().shape
        img = nib.load(ctImgPath.replace("_ct.nii.gz", "_seg-spine_msk.nii.gz"), mmap=False)  # Load the NIfTI file
        img.get_fdata().shape

    except Exception as e:
        print(f"Error loading CT image {ctImgPath}: {e}")
        continue

    ct_image_paths.append(ctImgPath)
    #print(f"CT Image Path: {ctImgPath}")  # Print the path to the CT image


    scalars = []
    for scalar in scalarsToAddToCSV:
        scalars.append(np.mean(v[scalar]["values"]))
    allScalars.append(scalars)
    #labels.append(v["labels"])

    
# %% create CSV with id, nifti_path, and scalars
import pandas as pd
df = pd.DataFrame({
    "id": list(range(len(ct_image_paths))),
    "nifti_path": ct_image_paths,
    "y0": [s[0] for s in allScalars],
    "y1": [s[1] for s in allScalars],
    "y2": [s[2] for s in allScalars],
    "y3": [s[3] for s in allScalars],
    "y4": [s[4] for s in allScalars],
})
df.to_csv("ct_nifti_scalars_full.csv", index=False)
# %%
