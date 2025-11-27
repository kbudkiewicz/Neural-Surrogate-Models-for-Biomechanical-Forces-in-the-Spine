import os
import argparse
import pandas as pd
import nibabel as nib

from utils import npz_to_dict, get_pat_ses, get_img_paths, generate_npz, get_scalars, unpack_scalar_from_dict

# pathMatfiles = pathDataset = os.curdir
pathMatfiles: str = '/mnt/models/simpack/Results/matlab'
pathDataset: str = '/mnt/derivatives/'
SCALARS: tuple = 'compr', 'shear', 'muscles', 'Ang'


def create_csv(csv_name: str, npz_filename: str, n: int, tasks: list[str] = None):
    """
    Args:
        csv_name (str): name of the csv file to which the data is saved.
        npz_filename (str): name of the npz file from which matlab structs are loaded.
        n (int): number of samples contained in the csv file.
        tasks (list[str]): list of tasks. Must contain string literals.
    """
    pathNpz = os.path.join(pathMatfiles, npz_filename)
    if not os.path.exists(pathNpz):
        print(f'{pathNpz} does not exist! Generating {npz_filename}...')
        generate_npz(n=n, npz_filename=npz_filename, datapath=pathMatfiles, tasks=tasks)
        print('Done.')

    npz_dict = npz_to_dict(pathNpz)
    csv_dict = {}
    scalars = get_scalars(*SCALARS, npz=npz_dict)    # 121 values if all forces selected

    print('Parsing dict...')
    for idx, (k, v) in enumerate(npz_dict.items()):
        temp = {}
        pat, ses, weight, current_task = get_pat_ses(k)
        # check the current_task is a substring contained in tasks. If True, unpack the data and save to csv
        if tasks is not None and isinstance(tasks, list):
            if not any(task in current_task for task in tasks):
                continue
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
            # save as a dict
            temp['task'] = current_task
            temp['nifti_path'] = ctImgPath
            for scalar in scalars:
                n_scalar, value = unpack_scalar_from_dict(v, scalar)
                temp[n_scalar] = value
            temp['weight'] = weight  # in kg
            csv_dict[idx] = temp
        except EOFError:
            print(f"ERROR: Corrupted CT image along {ctImgPath}.")
            continue

    print(f'Saving {csv_name}...')
    df = pd.DataFrame.from_dict(csv_dict, 'index')
    df.insert(0, 'id', range(len(df.index)))    # reset index
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_filename', required=True, type=str, default='default.csv')
    parser.add_argument('--npz_filename', required=False, type=str, default='default.npz')
    parser.add_argument('-s', '--samples', required=False, type=int, default=1000)
    parser.add_argument('-t', '--tasks', required=False, nargs='*', type=str)
    args = parser.parse_args()

    create_csv(args.csv_file, args.npz_file, args.samples, args.tasks)
