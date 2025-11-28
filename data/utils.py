import os
import h5py
import numpy as np
from typing import Tuple, Optional

# see Tanja's presentation for more details:
AVAILABLE_TASKS: tuple = 'Optim_0', 'Optim_30', 'Optim_10kg_25cm', 'Optim_10kg_55cm'
PREFIXES: tuple = '__G_Y_ga', 'SG_Y_Force__G_Y_', 'SG_muscles__', 'SG_Y_', '_Y', '__Y'
VALS_OF_INTEREST: tuple = 'shear', 'compr', 'muscle', 'Ang'


def rename_if_exists(path: str) -> str:
    idx = 0
    basename = os.path.basename(path)
    _, extension = os.path.splitext(basename)
    dirname = os.path.dirname(path)
    a = os.path.exists(path)
    while os.path.exists(path):
        path = os.path.join(dirname, basename + f'_{idx}' + extension)
        idx += 1
    return path


def prepare_dirs(eval_path: str, dataset_name: str, eval_name: str = 'eval.csv'):
    eval_path = os.path.join(eval_path, dataset_name)
    if not os.path.isdir(eval_path):
        print(f'Path "{eval_path}" does not exist. Creating directory...')
        os.mkdir(eval_path)
    else:
        eval_path = rename_if_exists(eval_path)
        os.mkdir(eval_path)
    eval_name = os.path.join(eval_path, eval_name)
    model_name = os.path.join(eval_path, dataset_name + '.pth')
    return eval_name, model_name


def get_available_paths(*tasks, datapath: str) -> iter:
    """List .mat files that contain a specified task identifier"""
    paths = []
    for path in os.listdir(datapath):
        for task in tasks:
            if task in path and '.npz' not in path and '_pre' not in path:
                paths.append(path)
    return iter(paths)


def hdf5_to_dict(h5file):
    def recursively_load(name, obj):
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        elif isinstance(obj, h5py.Group):
            return {k: recursively_load(k, obj[k]) for k in obj.keys()}
    return {key: recursively_load(key, h5file[key]) for key in h5file.keys()}


def get_yout(path: str) -> dict:
    """
    path (str): absolute path to .mat file
    """
    with h5py.File(path, 'r') as f:
        data_dict = hdf5_to_dict(f)['timeInt']
    return {**data_dict['yout'], **data_dict['forceOv']}


def generate_npz(n: int, npz_filename: str, datapath: str = os.curdir, tasks: Optional[list] = None):
    """
    n (int): number of samples to save in .npz.
    datapath (str, optional): Absolute path to directory containing .mat files.
    """
    results_dict = {}
    print(f'Finding paths along {datapath}...')
    if tasks is None:
        tasks = AVAILABLE_TASKS
    paths = get_available_paths(*tasks, datapath=datapath)
    npz_path = os.path.join(datapath, npz_filename)
    for idx, path in enumerate(paths):
        if idx == n:
            break
        filepath = os.path.join(datapath, path)
        print(f'{idx}: {filepath}')
        results_dict[path] = get_yout(filepath)
    np.savez(npz_path, **results_dict)
    print(f'DONE: {npz_filename} saved along {npz_filename}.')
    del results_dict


# --- CSV manipulation ---
def npz_to_dict(path: str) -> dict:
    print(f'Loading "{path}" to dict...')
    data = np.load(path, allow_pickle=True)
    data = {k: v.item() for k, v in data.items()}
    return data


def get_scalars(*forces: str, npz: dict) -> tuple:
    scalars = []
    for force in forces:
        for v in npz.values():
            for k in v.keys():
                if force in k and k not in scalars and 'Dummy' not in k:
                    scalars.append(k)
    return tuple(scalars)


def remove_prefix(scalar_name: str) -> str:
    for prefix in PREFIXES:
        while prefix in scalar_name:
            scalar_name = scalar_name.replace(prefix, '')
    return scalar_name


def unpack_scalar_from_dict(v, scalar) -> Tuple[str, np.array]:
    if 'compr' in scalar or 'shear' in scalar or 'Ang' in scalar:
        value = np.mean(v[scalar]['values'])
    elif 'muscle' in scalar:
        # ov_001, ov_002 -> force, torque
        value = np.mean(v[scalar]['ov_001']['values'])
    else:
        raise ValueError(f'Unknown scalar type "{scalar}".')
    return remove_prefix(scalar), value


def get_pat_ses(x: str) -> Tuple[str, str, float, str]:
    """
    Retrieve patient, session, and weight information from filename. If no weight is specified in the filename, the
    weight saved in the CSV will default to 0.0.
    """
    y = x.split('_')
    patient, format_ = y[0], y[-1]
    ses = 'ses-' + y[1].split('-')[1]
    task = x.replace(ses + '-', '').replace(patient + '_', '').replace('.mat', '')
    # get weight
    if 'kg' in task:
        t = task.split('_')[1].replace('kg', '')
        weight = float(t)
    else:
        weight = 0.0
    return patient, ses, weight, task


def get_img_paths(folder: str) -> str:
    """Find ct nifti files"""
    try:
        ct_images = [f for f in os.listdir(folder) if "ct.nii.gz" in f]
        if not len(ct_images) == 1:
            print(f"Expected one CT image, found {len(ct_images)} in {folder}")
        return os.path.join(folder, ct_images[0])
    except FileNotFoundError:
        raise FileNotFoundError(f"No ct.nii.gz files found along {folder}")
