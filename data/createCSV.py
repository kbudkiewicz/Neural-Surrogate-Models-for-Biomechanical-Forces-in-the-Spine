import os
import argparse
import pandas as pd
import nibabel as nib

from utils import npz_to_dict, get_pat_ses, get_img_paths, generate_npz, get_scalars, unpack_scalar_from_dict, read_sto

SCALARS: tuple = 'compr', 'shear', 'muscles', 'Ang'


def check_nii_file(path: str) -> None:
    img = nib.load(path, mmap=False)
    _ = img.get_fdata().shape


def create_csv_from_sto(root: str, csv_name: str, desired: str = 'inverse_dynamics.sto'):
    """Save data from sto_files along a given root directory to a csv file.

    .. note:: data and segmentation paths below are symbolic links to the datasets.
    """
    # data_path = segmentation_path = './'  # DEBUG
    data_path = segmentation_path = './derivatives-sim/rawdata_stitched'
    df = pd.DataFrame([])

    # find all files containing desired along root
    for dirpath, dirname, filenames in os.walk(root):
        if desired in filenames:
            try:
                sto = read_sto(os.path.join(dirpath, desired))  # read values from sto
                suffix = dirpath.replace(root, '').replace('\\', '/').replace('/mbs/results/ID', '')
                nako_data = data_path + suffix

                if os.path.exists(nako_data):
                    nako_file = get_img_paths(nako_data, 'T2w.nii.gz')
                    nako_msk = nako_file.replace(data_path, segmentation_path).replace('/T2w', '/vibe')
                    vibe_inphase = nako_msk.replace("-sag_T2w.nii.gz", "-ax_part-inphase_vibe.nii.gz")
                    vibe_outphase = nako_msk.replace("-sag_T2w.nii.gz", "-ax_part-outphase_vibe.nii.gz")
                    print(f'Checking {nako_file}...')

                    # Load nako files
                    check_nii_file(nako_file)
                    check_nii_file(vibe_inphase)
                    check_nii_file(vibe_outphase)

                    # add values to existing DataFrame
                    sto['nako_path'] = os.path.abspath(nako_file)
                    df = pd.concat([df, sto])
                else:
                    print(f'No data found along {nako_data}')
            except OSError:
                raise
            except EOFError:
                print(f"EOFError: Corrupted image along {nako_file}")
                continue

    print(f'Saving {csv_name}...')
    df[df.isna()] = 0.  # set forces and moments to 0 for patients without L6
    nako_paths = df.pop('nako_path')
    df.insert(0, 'nako_path', nako_paths)
    df.insert(0, 'id', range(len(df.index)))  # reset index
    df.to_csv(csv_name, index=False)


def create_csv_from_npz(csv_name: str, npz_filename: str, n: int, tasks: list[str] = None):
    """
    Args:
        csv_name (str): name of the csv file to which the data is saved.
        npz_filename (str): name of the npz file from which matlab structs are loaded.
        n (int): number of samples contained in the csv file.
        tasks (list[str]): list of tasks. Must contain string literals.

    .. note:: data and matfiles paths below are symbolic links to the datasets.
    """
    # pathMatfiles = pathDataset = os.path.join(os.curdir, 'mat')
    pathMatfiles: str = 'nifti_matfiles'
    pathDataset: str = 'nifti_data'
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
            tissue_msk_path = ctImgPath.replace("_ct.nii.gz", "_seg-tissue_msk.nii.gz")
            spine_msk_path = ctImgPath.replace("_ct.nii.gz", "_seg-spine_msk.nii.gz")
            check_nii_file(ctImgPath)
            check_nii_file(tissue_msk_path)
            check_nii_file(spine_msk_path)
            # save as a dict
            temp['task'] = current_task
            temp['nifti_path'] = ctImgPath
            for scalar in scalars:
                n_scalar, value = unpack_scalar_from_dict(v, scalar)
                temp[n_scalar] = value
            temp['weight'] = weight  # in kg
            csv_dict[idx] = temp
        except EOFError:
            print(f"ERROR: Corrupted image along {ctImgPath}.")
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

    create_csv_from_npz(args.csv_file, args.npz_file, args.samples, args.tasks)
    # create_csv_from_sto('./derivatives-sim', 'test_nako.csv')
