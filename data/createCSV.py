import os
import argparse
import pandas as pd
import nibabel as nib

from typing import Optional
from utils import npz_to_dict, get_pat_ses, get_img_paths, generate_npz, get_scalars, unpack_scalar_from_dict

# pathMatfiles = pathDataset = os.curdir
pathMatfiles: str = '/mnt/models/simpack/Results/matlab'
pathDataset: str = '/mnt/derivatives/'
SCALARS: tuple = 'compr', 'shear', 'muscles', 'Ang'


def create_csv(csv_name: str, npz_file: str, n: Optional[int] = None):
    pathNpz = os.path.join(pathMatfiles, npz_file)
    if not os.path.exists(pathNpz):
        print(f'{pathNpz} does not exist! Generating {npz_file}...')
        generate_npz(datapath=pathMatfiles, n=n)
        print('Done.')

    npz_dict = npz_to_dict(pathNpz)
    csv_dict = {}
    scalars = get_scalars(*SCALARS, npz=npz_dict)    # 121 values if all forces selected

    print('Parsing dict...')
    for idx, (k, v) in enumerate(npz_dict.items()):
        temp = {}
        pat, ses, weight = get_pat_ses(k)
        pathFolder = os.path.join(pathDataset, pat, ses)
        ctImgPath = get_img_paths(pathFolder)
        try:
            # Load the NIfTI files
            img = nib.load(ctImgPath, mmap=False)
            _ = img.get_fdata().shape
            img = nib.load(ctImgPath.replace("_ct.nii.gz", "_seg-tissue_msk.nii.gz"), mmap=False)
            _ = img.get_fdata().shape
            img = nib.load(ctImgPath.replace("_ct.nii.gz", "_seg-spine_msk.nii.gz"), mmap=False)
            _ = img.get_fdata().shape
            temp['id'] = idx
            temp['nifti_path'] = ctImgPath
            for scalar in scalars:
                n_scalar, value = unpack_scalar_from_dict(v, scalar)
                temp[n_scalar] = value
            temp['weight'] = weight  # in kg
            csv_dict[idx] = temp
        except EOFError:
            print(f"ERROR: Corrupted CT image along {ctImgPath}. Not adding to CSV.")
            continue

    print(f'Saving {csv_name}...')
    df = pd.DataFrame.from_dict(csv_dict, 'index')
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='default.csv')
    parser.add_argument('--npz_file', type=str, default='default.npz')
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    create_csv(args.csv_file, args.npz_file, args.n)
