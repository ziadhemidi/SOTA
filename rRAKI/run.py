from rRAKI import *
import torch
import os
import numpy as np


if __name__ == '__main__':

    # settings
    device = "cuda:0"
    down_scale = 6
    ACS_line_ratio = 0.08
    load_weight = False
    kernel_1 = (5, 2)       # (x, y)
    kernel_2 = (1, 1)
    kernel_3 = (3, 2)
    iters = 2000
    lr = 1e-3

    # path definition
    weight_path = ""
    file_path = ""
    save_path = ""

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    files = os.listdir(file_path)
    for f in files:
        ckp_path = os.path.join(weight_path, f.replace("npz", "pth"))

        # load data
        kspace = np.load(os.path.join(file_path, f))['fullysampled_kspace']  # [coil_num, height, width]
        coil_num = kspace.shape[0]
        # ACS
        idx_lower = int((1 - ACS_line_ratio) * kspace.shape[-1] / 2)
        idx_upper = int((1 + ACS_line_ratio) * kspace.shape[-1] / 2)
        ACS = kspace[..., idx_lower:idx_upper].copy()
        # [1, coil_num * 2, height, ACS_line_num]
        ACS = np.concatenate([ACS.real, ACS.imag], axis=0)[None, ...]
        # undersampled kspace
        undersampled_kspace = np.zeros_like(kspace)
        undersampled_kspace[..., ::down_scale] = kspace[..., ::down_scale].copy()
        undersampled_kspace[..., idx_lower:idx_upper] = kspace[..., idx_lower:idx_upper].copy()
        # [1, coil_num * 2, height, width]
        undersampled_kspace = np.concatenate([undersampled_kspace.real, undersampled_kspace.imag], axis=0)[None, ...]
        # convert to tensor on GPU
        undersampled_kspace = torch.tensor(undersampled_kspace, dtype=torch.float32).to(device)

        # model
        model = rRAKI(ACS.shape[1], down_scale, kernel_1, kernel_2, kernel_3).to(device)
        loss = rRAKI_loss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if load_weight:
            ckp = torch.load(ckp_path, map_location=device)
            model.load_state_dict(ckp['model'])
        else:
            target_x_start = np.int32(np.ceil(kernel_1[0] / 2) + np.floor(kernel_2[0] / 2) + np.floor(kernel_3[0] / 2) - 1)
            target_x_end = ACS.shape[2] - target_x_start
            target_y_start = np.int32((np.ceil(kernel_1[1] / 2) - 1) + (np.ceil(kernel_2[1] / 2) - 1) + (np.ceil(kernel_3[1] / 2) - 1)) * down_scale
            target_y_end = ACS.shape[3] - np.int32((np.floor(kernel_1[1] / 2) + np.floor(kernel_2[1] / 2) + np.floor(kernel_3[1] / 2))) * down_scale
            # ground truth shape
            target_dim_X = target_x_end - target_x_start
            target_dim_Y = target_y_end - target_y_start
            target_dim_Z = down_scale - 1

            targets = np.zeros([2 * coil_num, 1, target_dim_Z, target_dim_X, target_dim_Y])
            for j in range(2 * coil_num):
                for k in range(down_scale - 1):
                    target_y_start = np.int32((np.ceil(kernel_1[1] / 2) - 1) + (np.ceil(kernel_2[1] / 2) - 1) + (np.ceil(kernel_3[1] / 2) - 1)) * down_scale + k + 1
                    target_y_end = ACS.shape[-1] - np.int32((np.floor(kernel_1[1] / 2) + (np.floor(kernel_2[1] / 2)) + np.floor(kernel_3[1] / 2))) * down_scale + k + 1
                    targets[j, 0, k, :, :] = ACS[0, j, target_x_start:target_x_end, target_y_start:target_y_end]

            # convert to tensor on GPU
            ACS = torch.tensor(ACS, dtype=torch.float32).to(device)
            targets = torch.tensor(targets, dtype=torch.float32).to(device)

            model.train()
            for _ in range(iters):
                optimizer.zero_grad()
                outputs = model(ACS)
                err = loss(outputs, targets)
                err.backward()
                optimizer.step()
            torch.save({"model": model.state_dict()}, ckp_path)

        # reconstruction
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        recon_kspace = undersampled_kspace.clone()
        target_x_start = np.int32(np.ceil(kernel_1[0] / 2) + np.floor(kernel_2[0] / 2) + np.floor(kernel_3[0] / 2) - 1)
        target_x_end_kspace = undersampled_kspace.shape[-2] - target_x_start
        with torch.no_grad():
            model.eval()
            raki, res = model(undersampled_kspace)
            outputs = raki + res
            for j in range(2 * coil_num):
                for k in range(down_scale - 1):
                    target_y_start = np.int32((np.ceil(kernel_1[1] / 2) - 1) + np.int32((np.ceil(kernel_2[1] / 2) - 1)) + np.int32(np.ceil(kernel_3[1] / 2) - 1)) * down_scale + k + 1
                    target_y_end_kspace = undersampled_kspace.shape[-1] - np.int32((np.floor(kernel_1[1] / 2)) + (np.floor(kernel_2[1] / 2)) + np.floor(kernel_3[1] / 2)) * down_scale + k + 1
                    recon_kspace[0, j, target_x_start:target_x_end_kspace, target_y_start:target_y_end_kspace:down_scale] = outputs[j, 0, k, :, ::down_scale]

        recon_kspace[..., idx_lower:idx_upper] = ACS
        recon_kspace = recon_kspace.squeeze().cpu().numpy()
        recon_kspace = recon_kspace[:coil_num, ...] + 1j * recon_kspace[coil_num:, ...]

        np.savez(os.path.join(save_path, f), recon_kspace=recon_kspace)
        print(f"Complete file {f} Reconstruction...")

    print("All Done!")


