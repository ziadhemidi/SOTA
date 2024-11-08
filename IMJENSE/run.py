import my_utils as utils
import numpy as np
import IMJENSE
import torch
import os

if __name__ == '__main__':

    down_scale = 5
    center_ratio = 0.08
    dataset = 'brain'
    w0 = 31
    lamda = 3.8
    DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))

    save_path = f"G:\\dataset\\fastMRI_{dataset}\\Reconstruction\\restored\\IMJENSE\\scale={down_scale}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datapath = f"G:\\dataset\\fastMRI_{dataset}\\Reconstruction\\test"
    files = os.listdir(datapath)

    SSIM, PSNR = np.zeros(len(files)), np.zeros(len(files))
    for i in range(len(files)):
        # load data
        kspace = np.load(os.path.join(datapath, files[i]))['fullysampled_kspace']  # [coil_num, height, width]
        image = utils.normalize(np.sqrt(np.sum([utils.ifft(k) ** 2 for k in kspace], axis=0)), 2e-6, 1e-8)
        # mask
        idx_lower = int((1 - center_ratio) * kspace.shape[-1] / 2)
        idx_upper = int((1 + center_ratio) * kspace.shape[-1] / 2)
        mask = np.zeros_like(kspace[0], dtype=np.float32)
        mask[..., ::down_scale] = 1
        mask[..., idx_lower:idx_upper] = 1
        # undersampled kspace
        kspace = np.transpose(kspace, [1, 2, 0])
        mask = np.tile(mask[..., None], [1, 1, kspace.shape[-1]])
        under_kspace = kspace * mask / 2e-6

        pre_img, pre_Csm, pre_img_dc, pre_img_sos, pre_ksp = IMJENSE.IMJENSE_Recon(under_kspace, mask, DEVICE, w0=w0,
                                                                                   TV_weight=lamda, PolyOrder=15,
                                                                                   MaxIter=1500, LrImg=1e-4,
                                                                                   LrCsm=0.1)
        # pre_img_sos = utils.normalize(pre_img_sos, 2e-6, 1e-8)
        pre_img_sos = pre_img_sos.astype(np.float32)
        np.savez(os.path.join(save_path, files[i]), pred_image=pre_img_sos)

        image_tensor = torch.tensor(image[None, None, :, :])
        pre_img_sos_tensor = torch.tensor(pre_img_sos[None, None, :, :])

        SSIM[i] = utils.ssim(image_tensor, pre_img_sos_tensor).item()
        PSNR[i] = utils.psnr(image_tensor, pre_img_sos_tensor).item()
        print(f"Complete the {i+1}-th file reconstruction! SSIM={SSIM[i]}, PSNR={PSNR[i]}")

    print(f"Evaluation: downscale={down_scale} ssim={np.mean(SSIM):.4f} and psnr={np.mean(PSNR):.4f}")
