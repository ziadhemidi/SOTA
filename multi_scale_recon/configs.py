import os
from pathlib import Path


class config(object):

    # hardware
    device = 'cuda'                  # device: cpu, cuda:0, cuda:1 and so on
    batch_size = 2

    is_train = False
    is_eval = True

    # path
    snap_path = Path.home() / 'storage' / 'staff' / 'ziadalhajhemid' / 'projects' / 'CFMRIxRecon' / 'SOTA' / 'multi_scale_recon' / 'Model_logs'
    datapath = Path.home() / 'storage' / 'datasets' / 'FastMRI' / 'knee' / 'multi_coil'
    eval_path = Path.home() / 'storage' / 'staff' / 'ziadalhajhemid' / 'projects' / 'CFMRIxRecon' / 'SOTA' / 'multi_scale_recon' / 'results'

    # data params
    down_scale = (4, 6, 8, 10, )     # MRI undersample scale
    eval_scale = (4, 6, 8, 10, )
    keep_center = 0.08               # -> 8% ACS lines

    # train params
    n_iters = 100000                 # train iterations
    log_step = 100                   # log file record step
    val_step = 1000                  # validation step
    load_checkpoint = True          # load checkpoint

    # optimizer params
    lr = 1e-3                        # learning rate
    beta1 = 0.9
    beta2 = 0.99
    weight_decay = 0.0
    scheduler_step = 100000

    # network params:
    image_out_dim = 1

    mlp_hidden_dim = 256
    mlp_num_layer = 8
    mlp_skips = (3, 6)                   # skip connection layer list

    enc_hidden_dim = 64
    enc_out_dim = 128
    enc_block_num = 5
    enc_kernel_size = (5, 5)

    # ablation settings
    use_encoder = True
    scale_embed = True
    pos_encoding = True
    is_pre_combine = False

    # positional encoding params:
    include_coord = False            # reserve original coordinates or not
    pos_fre_num = 128                # enconding frequency num
    pos_scale = 1                   # enconding coefficient scale


    def __init__(self, record=True):

        if self.is_pre_combine:
            self.image_in_dim = 1
        else:
            if 'brain' in self.datapath.as_posix():
                self.image_in_dim = 16
                self.image_size = 384
            else:
                self.image_in_dim = 15
                self.image_size = 320

        if record:
            self.record()

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")

    def record(self, subpath=None):

        if subpath is not None:
            path = os.path.join(self.snap_path, subpath)
        else:
            path = self.snap_path

        # create folder
        if not os.path.exists(path):
            os.makedirs(path)

        config_file = open(os.path.join(path, 'configs.txt'), 'w')
        for key, val in self.to_dict().items():
            config_file.write(f"{key:30} \t {val}\n")
        config_file.close()
