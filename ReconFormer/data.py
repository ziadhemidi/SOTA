import os
import utils
import numpy as np
import transforms
from torch.utils import data
import torch


class Dataset(data.Dataset):

    def __init__(self, datapath, down_scale, keep_center):

        self.down_scale = down_scale
        self.filepath, self.filename = utils.get_filepath(datapath)
        self.center_ratio = keep_center
        self.mask = self.create_mask()

    def create_mask(self):

        fullysampled_kspace = np.load(self.filepath[0])['fullysampled_kspace']
        idx_lower = int((1 - self.center_ratio) * fullysampled_kspace.shape[-1] / 2)
        idx_upper = int((1 + self.center_ratio) * fullysampled_kspace.shape[-1] / 2)
        mask = np.zeros_like(fullysampled_kspace[0], dtype=np.float32)
        mask[..., ::self.down_scale] = 1
        mask[..., idx_lower:idx_upper] = 1

        return np.stack((mask, mask), axis=-1)

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, item):
        # fully sampled kspace
        fullysampled_kspace = np.load(self.filepath[item])['fullysampled_kspace']
        # fully sampled image
        fully_image = utils.normalize(np.sqrt(np.sum([utils.ifft(k) ** 2 for k in fullysampled_kspace], axis=0)), 2e-6, 1e-8)
        fullysampled_kspace = utils.fft(fully_image).astype(np.complex64)

        kspace = transforms.to_tensor(fullysampled_kspace)
        # Apply mask
        mask = torch.from_numpy(self.mask)
        masked_kspace = kspace * mask
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)

        target = transforms.ifft2(kspace)

        image = image.permute(2, 0, 1)
        target = target.permute(2, 0, 1)
        masked_kspace = masked_kspace.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        return image, target, self.filename[item], mask, masked_kspace


def get_dataloader(config, evaluation=False, shuffle=True):

    if not evaluation:
        train_ds = Dataset(datapath=os.path.join(config.datapath, 'train_2Dslices'),
                           down_scale=config.down_scale,
                           keep_center=config.keep_center)
        val_ds = Dataset(datapath=os.path.join(config.datapath, 'val_2Dslices'),
                         down_scale=config.down_scale,
                         keep_center=config.keep_center)

        train_loader = cycle(data.DataLoader(dataset=train_ds, batch_size=config.batch_size, pin_memory=True, shuffle=shuffle))
        val_loader = data.DataLoader(dataset=val_ds, batch_size=config.batch_size, pin_memory=True, shuffle=shuffle)
        return train_loader, val_loader
    else:
        eval_ds = Dataset(datapath=os.path.join(config.datapath, 'test_2Dslices'),
                          down_scale=config.down_scale,
                          keep_center=config.keep_center)
        eval_loader = data.DataLoader(dataset=eval_ds, batch_size=1, pin_memory=True, shuffle=False)
        return eval_loader


def cycle(dl):

    while True:
        for data in dl:
            yield data
