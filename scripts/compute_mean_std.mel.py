import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def check_for_nans_or_infs(array, name):
    tensor = torch.tensor(array)
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        return True
    return False

if __name__ == "__main__":

    feat_type = 'mel'
    exp_dir = './training_data/exp_data/'

    out_dir = exp_dir

    # ### Process ###

    dataset_fp = os.path.join(exp_dir, f'dataset.pkl')
    feat_dir = os.path.join(exp_dir, feat_type)

    out_fp_mean = os.path.join(out_dir, f'mean.{feat_type}.npy')
    out_fp_std = os.path.join(out_dir, f'std.{feat_type}.npy')

    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    in_fns = [fn for fn, _ in dataset]

    scaler = StandardScaler()

    for fn in in_fns:
        print(fn)
        in_fp = os.path.join(feat_dir, f'{fn}.npy')
        data = np.load(in_fp).T

        # Check for NaNs or Infs in data
        if check_for_nans_or_infs(data, f'data from {fn}'):
            continue

        scaler.partial_fit(data)

    mean = scaler.mean_
    std = scaler.scale_

    # Check for NaNs or Infs in mean and std
    if check_for_nans_or_infs(mean, 'mean array') or check_for_nans_or_infs(std, 'std array'):
        print("Mean or std array contains NaN or Inf. Exiting.")
        exit(1)

    np.save(out_fp_mean, mean)
    np.save(out_fp_std, std)

    print("Mean and std arrays have been saved successfully.")
