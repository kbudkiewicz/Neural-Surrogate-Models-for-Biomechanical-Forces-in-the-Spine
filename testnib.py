#%%
import nibabel as nib
img = nib.load("/mnt/8tb_slot8/jonas/datasets/CT_nofx/derivatives/sub-ctsr00555/ses-20060209/sub-ctsr00555_ses-20060209_sequ-7_space-aligASL_seg-spine_msk.nii.gz")

# %%
img 
# %%
img.get_fdata().shape
# %%
import matplotlib.pyplot as plt
plt.imshow(img.get_fdata()[..., 0, 0],           cmap="gray")  
# %%
import gzip
import nibabel as nib
from io import BytesIO

with gzip.open("/mnt/8tb_slot8/jonas/datasets/CT_nofx/derivatives/sub-ctsr00555/ses-20060209/ sub-ctsr00555_ses-20060209_sequ-7_space-aligASL_seg-spine_msk.nii.gz", "rb") as f:
    data = f.read()
img = nib.load(BytesIO(data))


# %%
