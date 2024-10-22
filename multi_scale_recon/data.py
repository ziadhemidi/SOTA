import os
import utils
import numpy as np
from torch.utils import data


############################# fft and ifft ##################################
def fft(img):
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
    return kspace

def ifft(kspace):
    img = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace)))
    img = np.sqrt(img.real ** 2 + img.imag ** 2)
    return img


############################# data loader ##################################
class Dataset(data.Dataset):

    def __init__(self, datapath, down_scale, keep_center=0.08, is_pre_combine=True):

        self.down_scale = down_scale
        self.filepath, self.filename = utils.get_filepath(datapath)
        self.center_ratio = keep_center
        self.is_pre_combine = is_pre_combine

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, ind):
        # down scale choice
        scale = np.random.choice(self.down_scale, 1)
        # fully sampled kspace
        fullysampled_kspace = np.load(self.filepath[ind])['fullysampled_kspace']
        # fully sampled image
        image = utils.normalize(np.sqrt(np.sum([ifft(k) ** 2 for k in fullysampled_kspace], axis=0)), 2e-6, 1e-8)
        # coordinates
        coords = utils.create_grid(image.shape[0], image.shape[1])
        # undersampled kspace
        undersampled_kspace = np.zeros_like(fullysampled_kspace)
        idx_lower = int((1 - self.center_ratio) * fullysampled_kspace.shape[-1] / 2)
        idx_upper = int((1 + self.center_ratio) * fullysampled_kspace.shape[-1] / 2)
        undersampled_kspace[..., ::scale[0]] = fullysampled_kspace[..., ::scale[0]]
        undersampled_kspace[..., idx_lower:idx_upper] = fullysampled_kspace[..., idx_lower:idx_upper]
        # undersampled image
        if self.is_pre_combine:
            undersampled_image = utils.normalize(np.sqrt(np.sum([ifft(k) ** 2 for k in undersampled_kspace], axis=0)), 2e-6, 1e-8)[None, ...]
        else:
            # normalization range: 1.53e-06 4.03e-13 for brain (2.14e-06 1.21e-08)
            #                      3.60e-06 4.32e-13 for knee (4.22e-06 1.02e-08)
            undersampled_image = utils.normalize(np.array([ifft(k) for k in undersampled_kspace]), 2e-6, 1e-8)
        # integrate inputs and targets
        inputs = {'coords': coords.astype(np.float32),
                  'undersampled_image': undersampled_image,
                  'undersampled_kspace': undersampled_kspace,
                  'down_scale': scale}

        targets = {'image': np.expand_dims(image, axis=0),
                   'fullysampled_kspace': fullysampled_kspace,
                   'filename': self.filename[ind]}

        return inputs, targets

def get_dataloader(config, evaluation=False, shuffle=True, eval_scale=None):

    if not evaluation:
        train_ds = Dataset(datapath=os.path.join(config.datapath, 'train'),
                           down_scale=config.down_scale,
                           keep_center=config.keep_center,
                           is_pre_combine=config.is_pre_combine)
        val_ds = Dataset(datapath=os.path.join(config.datapath, 'val'),
                         down_scale=config.down_scale,
                         keep_center=config.keep_center,
                         is_pre_combine=config.is_pre_combine)

        train_loader = cycle(data.DataLoader(dataset=train_ds, batch_size=config.batch_size, pin_memory=True, shuffle=shuffle))
        val_loader = data.DataLoader(dataset=val_ds, batch_size=config.batch_size, pin_memory=True, shuffle=shuffle)
        return train_loader, val_loader
    else:
        assert eval_scale is not None
        eval_ds = Dataset(datapath=os.path.join(config.datapath, 'test'),
                          down_scale=(eval_scale, ),
                          keep_center=config.keep_center,
                          is_pre_combine=config.is_pre_combine)
        eval_loader = data.DataLoader(dataset=eval_ds, batch_size=1, pin_memory=True, shuffle=False)
        return eval_loader


def cycle(dl):

    while True:
        for data in dl:
            yield data


