import numpy as np
import os
from pygrappa import grappa


if __name__ == '__main__':

    down_scale = 6
    ACS_line_ratio = 0.08

    file_path = ""
    save_path = ""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(file_path)
    for f in files:
        kspace = np.load(os.path.join(file_path, f))['fullysampled_kspace']  # [coil_num, height, width]
        # ACS
        idx_lower = int((1 - ACS_line_ratio) * kspace.shape[-1] / 2)
        idx_upper = int((1 + ACS_line_ratio) * kspace.shape[-1] / 2)
        ACS = kspace[..., idx_lower:idx_upper].copy()
        # undersampled kspace
        undersampled_kspace = np.zeros_like(kspace)
        undersampled_kspace[..., ::down_scale] = kspace[..., ::down_scale].copy()
        undersampled_kspace[..., idx_lower:idx_upper] = kspace[..., idx_lower:idx_upper]

        # grappa reconstruction
        recon_kspace = grappa(undersampled_kspace, ACS, coil_axis=0)

        np.savez(os.path.join(save_path, f), recon_kspace=recon_kspace)

