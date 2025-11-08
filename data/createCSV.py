import os
import argparse
import pandas as pd
import nibabel as nib

from utils import npz_to_dict, get_pat_ses, get_img_paths, generate_npz, get_scalars, unpack_scalar_from_dict

# pathMatfiles = pathDataset = os.curdir
pathMatfiles = '/mnt/models/simpack/Results/matlab'
pathDataset = '/mnt/derivatives/'
FORCES = 'compr', 'shear', 'muscles'
# timeInt.forceOv -> SG_muscles*


def create_csv(csv_name, npz_file):
    pathNpz = os.path.join(pathMatfiles, npz_file)
    if not os.path.exists(pathNpz):
        print(f'{pathNpz} does not exist! Generating {npz_file}...')
        generate_npz(datapath=pathMatfiles)
        print('Done.')

    npz_dict = npz_to_dict(pathNpz)
    csv_dict = {}
    scalars = get_scalars(*FORCES, npz=npz_dict)    # 121 values if all forces selected

    print('Parsing dict...')
    for idx, (k, v) in enumerate(npz_dict.items()):
        temp = {}
        pat, ses = get_pat_ses(k)
        pathFolder = os.path.join(pathDataset, pat, ses)
        ctImgPath = get_img_paths(pathFolder)
        try:
            img = nib.load(ctImgPath, mmap=False)  # Load the NIfTI file
            img = nib.load(ctImgPath.replace("_ct.nii.gz", "_seg-tissue_msk.nii.gz"), mmap=False)  # Load the NIfTI file
            img = nib.load(ctImgPath.replace("_ct.nii.gz", "_seg-spine_msk.nii.gz"), mmap=False)  # Load the NIfTI file
        except Exception as e:
            print(f"Error loading CT image {ctImgPath}: {e}. Not added to csv.")
            continue

        temp['id'] = idx
        temp['nifti_path'] = ctImgPath
        for scalar in scalars:
            n_scalar, value = unpack_scalar_from_dict(v, scalar)
            temp[n_scalar] = value
        csv_dict[idx] = temp

    print(f'Saving {csv_name}...')
    df = pd.DataFrame.from_dict(csv_dict, 'index')
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='default.csv')
    parser.add_argument('--npz_file', type=str, default='default.npz')
    args = parser.parse_args()

    create_csv(args.csv_file, args.npz_file)
