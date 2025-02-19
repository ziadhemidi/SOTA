import os
from pathlib import Path
import data
import utils
import torch
import numpy as np
import transforms
from models.Recurrent_Transformer import ReconFormer


class configs:

    device = 'cuda:0'

    n_iters = 100000
    val_step = 1000
    batch_size = 1
    lr = 0.0001

    data_type = 'knee'

    down_scale = (4, 6, 8, 10, )
    keep_center = 0.08

    def __init__(self):

        self.snap_path = ""
        self.datapath = ""
        self.eval_path = ""
        self.snap_path = Path.home() / 'storage' / 'staff' / 'ziadalhajhemid' / 'projects' / 'CFMRIxRecon' / 'SOTA' / 'ReconFormer' / 'Model_logs'
        self.datapath = Path.home() / 'storage' / 'datasets' / 'FastMRI' / 'knee' / 'multi_coil'
        self.eval_path = Path.home() / 'storage' / 'staff' / 'ziadalhajhemid' / 'projects' / 'CFMRIxRecon' / 'SOTA' / 'ReconFormer' / 'results'
        
        if not os.path.exists(self.snap_path):
            os.makedirs(self.snap_path)

        if self.data_type == "brain":
            self.img_size = 384
        else:
            self.img_size = 320

        self.log_step = 80 // self.batch_size


if __name__ == '__main__':

    config = configs()
    device = config.device
    print('using device: %s.' % device)

    # Create directories for experimental logs
    log_file = open(os.path.join(config.snap_path, 'log.txt'), 'a+')

    model = ReconFormer(
        in_channels=2, out_channels=2, num_ch=(96, 48, 24), num_iter=5, down_scales=(2, 1, 1.5),
        img_size=config.img_size, num_heads=(6, 6, 6), depths=(2, 1, 1),
        window_sizes=(8, 8, 8), mlp_ratio=2., resi_connection='1conv',
        use_checkpoint=(False, False, True, True, False, False)
    ).to(device)
    # loss
    loss_L1 = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    metric_ssim = 0
    checkpoint_dir = os.path.join(config.snap_path, 'checkpoint.pth')

    # data loader
    train_loader, val_loader = data.get_dataloader(config)

    for step in range(config.n_iters):

        model.train()
        optimizer.zero_grad()

        input, target, fname, mask, masked_kspace = next(train_loader)
        #with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(input.to(device), masked_kspace.to(device), mask.to(device))

        output = transforms.complex_abs(output)
        target = transforms.complex_abs(target.to(device))

        loss = loss_L1(output, target)
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

        if (step + 1) % config.log_step == 0:
            print('Step: [%d/%d], lr: [%.8f], loss=%.5f'
                % (step + 1, config.n_iters, optimizer.param_groups[0]['lr'], loss.item()))
            log_file.write('Step: [%d/%d], lr: [%.8f], loss=%.5f\n'
                % (step + 1, config.n_iters, optimizer.param_groups[0]['lr'], loss.item()))

        if (step + 1) % config.val_step == 0:
            model.eval()
            with torch.no_grad():
                ssim_eval, psnr_eval, loss_eval = 0, 0, 0
                for i, batch in enumerate(val_loader):
                    input, target, fname, mask, masked_kspace = batch
                    #with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = model(input.to(device), masked_kspace.to(device), mask.to(device))

                    output = transforms.complex_abs(output).unsqueeze(1)
                    target = transforms.complex_abs(target.to(device)).unsqueeze(1)

                    loss_eval += loss_L1(output, target).item()
                    ssim_eval += utils.ssim(output, target).item()
                    psnr_eval += utils.psnr(output, target).item()

                loss_eval = loss_eval / len(val_loader)
                ssim_eval = ssim_eval / len(val_loader)
                psnr_eval = psnr_eval / len(val_loader)
                print("Step: [%d/%d] Evaluation: loss=%.5f, ssim=%.5f, psnr=%.5f\n" % (step + 1, config.n_iters, loss_eval, ssim_eval, psnr_eval))
                log_file.write("Step: [%d/%d] Evaluation: loss=%.5f, ssim=%.5f, psnr=%.5f\n" % (step + 1, config.n_iters, loss_eval, ssim_eval, psnr_eval))

                if metric_ssim < ssim_eval:
                    metric_ssim = ssim_eval
                    torch.save({
                        "step": step,
                        "model": model.state_dict()
                    }, checkpoint_dir)
                    print('save weights of step %d' % (step + 1) + '\n')
                    log_file.write('save weights of step %d' % (step + 1) + '\n')

    # evaluation after training
    model.eval()
    if not os.path.exists(config.eval_path):
        os.makedirs(config.eval_path)
    eval_loader = data.get_dataloader(config, evaluation=True)
    with torch.no_grad():
        # load best model
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        model.load_state_dict(checkpoint['model'])
        eval_ssim, eval_psnr = 0, 0
        for i, batch in enumerate(eval_loader):
            input, target, fname, mask, masked_kspace = batch
            #with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(input.to(device), masked_kspace.to(device), mask.to(device))

            output = transforms.complex_abs(output).unsqueeze(1)
            target = transforms.complex_abs(target.to(device)).unsqueeze(1)

            # evaluation metrics
            eval_ssim += utils.ssim(output, target).item()
            eval_psnr += utils.psnr(output, target).item()
            # save output
            pred_image = output[0, 0].cpu().numpy()

            np.savez(os.path.join(config.eval_path, fname[0]), pred_image=pred_image)
        # record the mean metrics
        num = len(eval_loader)

        print('\nEvaluation: downscale=%d ssim=%.4f and psnr=%.4f' % (config.down_scale, eval_ssim / num, eval_psnr / num))
        log_file.write('\nEvaluation: downscale=%d ssim=%.4f and psnr=%.4f\n' % (config.down_scale, eval_ssim / num, eval_psnr / num))
    log_file.close()
